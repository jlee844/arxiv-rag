"""Figure extraction — pull charts, plots, and diagrams out of the PDFs.

WHY CLIP-RENDER INSTEAD OF `page.get_images()`:

The obvious implementation extracts embedded raster images. Measured on a
25-paper sample of this corpus, that finds 462 images — and **misses 73 pages of
vector figures entirely**. Matplotlib, TikZ, and pgfplots export figures as PDF
*drawing operators*, not bitmaps, so `get_images()` returns nothing for exactly
the scatter plots and ablation charts a reader most wants. It fails silently:
you get a plausible number of figures and no error.

So this module never asks "what bitmaps are embedded?". It asks "where is the
figure region on the page?" and renders that rectangle. Vector and raster come
out identically, because rendering is what a PDF viewer does anyway.

HOW A REGION IS FOUND, without a layout model:

Captions are the anchor. They are typographically regular ("Figure 3: ...",
"Table 1: ..."), they sit in the text layer already, and they are written by the
authors — free, high-signal text that no vision model has to reconstruct.

A FIGURE caption sits BELOW its graphic, so the region is the vertical band
between the body text above and the caption — bounded to the caption's own
column, because reading order lies on two-column papers (see `_column_ceiling`).

TABLES ARE DELIBERATELY EXCLUDED. Their caption sits above typeset text, so the
band is the whitespace between the two: the first version emitted 6 zero-byte
PNGs from exactly that. And table content already lives in the text layer and is
already indexed, so an image of it adds no retrievable signal.

VALIDATION IS LAYERED, and the last layer is the one that matters:

    caption regex   -> is this a caption or an "as Figure 3 shows" reference?
    column ceiling  -> where does the figure actually start?
    graphic extent  -> do drawings or images live in this band?
    INK FRACTION    -> did the render produce anything at all?

Only the last one inspects the output. Every check above it reasons about
metadata, and metadata can be confidently wrong — the Form XObject fallback
looked principled and produced a perfectly blank 212x171 crop. See
`_ink_fraction`.

KNOWN LIMIT: precision is good, recall is not measured. Some papers yield zero
figures (older single-column typesetting, figures without regular captions).
Extraction is caption-anchored, so a figure with no parseable caption does not
exist as far as this module is concerned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

import fitz

# Caption openers. Anchored to a block's start, because "see Figure 3" mid
# sentence is a reference, not a caption — matching those produced crops of
# body text.
_CAPTION_RE = re.compile(
    r"^\s*(?P<kind>Fig(?:ure)?\.?|Table)\s*(?P<num>\d+)\s*[:.]\s*(?P<rest>\S.*)",
    re.IGNORECASE | re.DOTALL,
)

# A caption shorter than this is usually a cross-reference fragment or a stray
# axis label that happens to start with "Fig".
_MIN_CAPTION_CHARS = 25

# Below this the crop is a rule, a logo, or a math glyph rather than a figure.
_MIN_REGION_PT = 40.0

# Vector figures are detected by drawing-operator density inside the band.
_MIN_DRAW_OPS = 8


@dataclass
class Figure:
    """One extracted figure or table, with the caption that anchored it."""

    figure_id: str          # f"{arxiv_id}-p{page}-{kind}{num}"
    arxiv_id: str
    page: int               # 0-based
    kind: str               # "figure" | "table"
    label: str              # "Figure 3"
    caption: str
    image_path: str
    bbox: tuple[float, float, float, float]
    n_drawings: int
    n_images: int
    ink: float = 0.0        # non-white pixel fraction; see _ink_fraction

    def to_dict(self) -> dict:
        return asdict(self)

    def index_text(self, description: str | None = None) -> str:
        """The text that represents this figure in the retrieval index.

        Caption first, because it is author-written and the highest-precision
        signal available. A VLM description is appended when one exists, which
        is what makes "caption-only vs caption+VLM" a measurable ablation rather
        than an assumption.
        """
        parts = [f"{self.label}: {self.caption}"]
        if description:
            parts.append(description)
        return "\n".join(parts)


def _blocks(page) -> list[tuple[fitz.Rect, str]]:
    """Text blocks as (rect, text), in reading order, non-empty only."""
    out = []
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
        if text and text.strip():
            out.append((fitz.Rect(x0, y0, x1, y1), text.strip()))
    out.sort(key=lambda rb: (round(rb[0].y0, 1), rb[0].x0))
    return out


def _is_prose(text: str) -> bool:
    """True for body-text blocks, false for figure-internal text.

    Charts carry their own text — axis ticks, legend entries, subplot labels
    ("(a)", "0.75", "Accuracy") — and PyMuPDF returns those as ordinary text
    blocks. Treating them as the ceiling clips the band to *inside* the figure:
    measured bands of 8-21 pt on a paper whose figures are half a page tall,
    dropping every one of them.

    Word count separates the two cleanly enough. Body prose runs long; axis
    labels are a word or two.
    """
    return len(text.split()) >= 8


def _column_ceiling(blocks, caption_rect: fitz.Rect, page_rect: fitz.Rect) -> float:
    """Bottom edge of the nearest text ABOVE the caption IN THE SAME COLUMN.

    THIS IS THE TWO-COLUMN FIX, and it was the single biggest defect in the
    first version. Taking "the previous block in reading order" is wrong on a
    two-column paper: reading order runs down column 1 then down column 2, so
    the block before a column-2 caption is physically *below* it in column 1.
    Measured, that produced `prev_bottom=240` for a caption whose top was
    `y=212` — a band of height 0, so the figure was silently dropped. Whole
    papers returned zero figures with no error.

    Overlap is required on the x-axis, which is what confines the search to one
    column without needing to detect the column layout itself.
    """
    ceiling = page_rect.y0
    for rect, text in blocks:
        if rect.y1 > caption_rect.y0:
            continue                       # not above the caption
        # Horizontal overlap => same column. A pure gap test would match the
        # neighbouring column and re-introduce the bug.
        if rect.x1 <= caption_rect.x0 or rect.x0 >= caption_rect.x1:
            continue
        # Only body prose bounds a figure. Axis labels and legend text sit
        # INSIDE it, and using them as the ceiling crops the figure away.
        if not _is_prose(text):
            continue
        ceiling = max(ceiling, rect.y1)
    return ceiling


def _graphic_extent(page, band: fitz.Rect) -> tuple[fitz.Rect | None, int, int]:
    """Union of drawing/image geometry inside `band`.

    Returns (rect, n_drawings, n_images). `rect` is None when the band holds no
    graphics — the signal that this "caption" was really a prose reference.
    """
    union: fitz.Rect | None = None
    n_draw = 0
    for d in page.get_drawings():
        r = d.get("rect")
        if r is None or not r.intersects(band):
            continue
        n_draw += 1
        union = r if union is None else (union | r)

    n_img = 0
    for info in page.get_image_info():
        r = fitz.Rect(info["bbox"])
        if not r.intersects(band):
            continue
        n_img += 1
        union = r if union is None else (union | r)

    return union, n_draw, n_img


def _ink_fraction(pix) -> float:
    """Fraction of non-white pixels in a rendered crop.

    THE FINAL GATE, and the only one that inspects the actual output.

    Every check before this reasons about *metadata* — caption regexes, drawing
    counts, block geometry — and metadata can be confidently wrong. The Form
    XObject fallback looked principled ("a tall band with no prose must be a
    figure") and rendered a **completely blank 212x171 crop**: a real PNG, a
    plausible file size, and nothing in it. No earlier check could have caught
    that, because from the metadata's point of view everything was fine.

    Asking the pixels is decisive and costs ~1 ms.
    """
    import numpy as np

    buf = np.frombuffer(pix.samples, dtype=np.uint8)
    if buf.size == 0:
        return 0.0
    # Treat near-white as blank: anti-aliased page background is not exactly 255.
    return float((buf < 250).mean())


# A crop below this is blank page, a hairline rule, or a stray margin mark.
_MIN_INK = 0.005


def extract_figures(
    pdf_path: Path | str,
    arxiv_id: str,
    out_dir: Path | str,
    dpi: int = 150,
) -> list[Figure]:
    """Extract every captioned figure/table from one PDF.

    Args:
        pdf_path: Source PDF.
        arxiv_id: Used for ids and the output subdirectory.
        out_dir: Root directory for rendered PNGs.
        dpi: Render resolution. 150 keeps axis labels legible to a VLM while
            holding a full-width figure near ~200 KB.

    Returns:
        Figures in page order. Empty when the PDF has no captioned graphics —
        which is a real answer, not a failure.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir) / arxiv_id
    doc = fitz.open(pdf_path)
    figures: list[Figure] = []
    seen: set[str] = set()

    try:
        for pno in range(len(doc)):
            page = doc[pno]
            blocks = _blocks(page)
            page_rect = page.rect

            for i, (rect, text) in enumerate(blocks):
                m = _CAPTION_RE.match(text)
                if not m:
                    continue
                # Store the caption WITHOUT its "Figure 3:" prefix — `label`
                # already carries that. Keeping both duplicated it into every
                # indexed chunk ("Figure 2: Figure 2: Plot of accuracy..."),
                # which wastes tokens and doubles "figure"/the number for BM25.
                caption = " ".join(m.group("rest").split())
                if len(caption) < _MIN_CAPTION_CHARS:
                    continue

                # TABLES ARE DELIBERATELY SKIPPED. Their caption sits above
                # typeset text, so the "band" is the whitespace between caption
                # and table body — the first version rendered 6 zero-byte PNGs
                # from exactly that. And a table's content is already in the
                # text layer and already indexed, so an image of it adds no
                # retrievable signal. Captions are still matched so the regex
                # stays honest about what it saw.
                if m.group("kind").lower().startswith("tab"):
                    continue

                kind = "figure"
                label = f"Figure {m.group('num')}"
                fid = f"{arxiv_id}-p{pno}-{kind}{m.group('num')}"
                if fid in seen:
                    continue

                # Band = from the text above (same column) down to the caption.
                ceiling = _column_ceiling(blocks, rect, page_rect)
                # Horizontal extent is the CAPTION's column, not the page. A
                # full-width band on a two-column paper crops the neighbour.
                pad = 6.0
                band = fitz.Rect(
                    max(page_rect.x0, rect.x0 - pad), ceiling,
                    min(page_rect.x1, rect.x1 + pad), rect.y0,
                )
                if band.height < _MIN_REGION_PT:
                    continue

                extent, n_draw, n_img = _graphic_extent(page, band)
                has_graphics = extent is not None and (n_draw >= _MIN_DRAW_OPS or n_img > 0)

                # FALLBACK for Form XObjects. An included EPS/PDF figure is a
                # single form object: `get_drawings()` does not descend into it
                # and `get_image_info()` does not list it, so both report zero.
                # Measured on 0112004v1 — a healthy 212x171 pt band, draw=0,
                # img=0, and a real figure sitting in it.
                #
                # A tall band in this column containing NO body prose is a
                # figure by elimination: prose would have set the ceiling.
                prose_inside = any(
                    _is_prose(t) and r.intersects(band) for r, t in blocks
                )
                if not has_graphics and not (band.height >= 80.0 and not prose_inside):
                    # Require actual graphics. Without this, every "As Figure 3
                    # shows..." that begins a paragraph crops a slab of prose.
                    continue

                # Intersect the graphic extent with the band: the extent alone
                # can spill across the page (a full-width rule inside a
                # one-column figure), the band alone can include leading
                # whitespace.
                region = (extent & band) if extent is not None else band
                if region.is_empty:
                    region = band

                if region.height < _MIN_REGION_PT or region.width < _MIN_REGION_PT:
                    continue

                pix = page.get_pixmap(clip=region, dpi=dpi)
                ink = _ink_fraction(pix)
                if ink < _MIN_INK:
                    # Rendered, inspected, empty. Discard rather than ship a
                    # blank PNG that every downstream stage would treat as a
                    # real figure.
                    continue

                out_dir.mkdir(parents=True, exist_ok=True)
                img_path = out_dir / f"{fid}.png"
                pix.save(img_path)

                figures.append(Figure(
                    figure_id=fid, arxiv_id=arxiv_id, page=pno, kind=kind,
                    label=label, caption=caption, image_path=str(img_path),
                    bbox=(region.x0, region.y0, region.x1, region.y1),
                    n_drawings=n_draw, n_images=n_img, ink=round(ink, 4),
                ))
                seen.add(fid)
    finally:
        doc.close()

    return figures
