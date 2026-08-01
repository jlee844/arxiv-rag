# Eval case review

58 clean · 1 ambiguous · 3 missed · 62 total candidates.

## How to use

Edit the `[KEEP]` / `[DROP]` marker on each `###` heading. Everything
defaults to KEEP. When done:

```bash
.venv/bin/python scripts/gen_eval_cases.py --promote
```

**Judge a MISS by reading the question and paper — NOT by whether
retrieval found it.** Dropping cases because the current system fails
them strips out exactly what the eval exists to catch, and quietly
inflates recall. Some misses SHOULD survive.

If other retrieved papers are genuinely relevant too, add their ids to
the `Also relevant` line — an incomplete label reads as a false miss.

---

### [KEEP] paraphrase-2310.14025

- **verdict:** `MISS`  ·  **source rank:** `None`  ·  **tag:** `paraphrase`
- **query:** method for selecting images that match ambiguous words in specific contexts
- **labeled:** `2310.14025` — Large Language Models and Multimodal Retrieval for Visual Word Sense Disambiguation
- **retrieved top-5:**
  1. `2507.22398` On the Reliability of Vision-Language Models Under Adversarial Frequen 
  2. `2409.11353` THaMES: An End-to-End Tool for Hallucination Mitigation and Evaluation 
  3. `2603.23521` Chitrakshara: A Large Multilingual Multimodal Dataset for Indian langu 
  4. `2302.14383` Linear Spaces of Meanings: Compositional Structures in Vision-Language 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2109.01134

- **verdict:** `MISS`  ·  **source rank:** `None`  ·  **tag:** `paraphrase`
- **query:** method for improving vision language models through context optimization
- **labeled:** `2109.01134` — Learning to Prompt for Vision-Language Models
- **retrieved top-5:**
  1. `2404.07214` Exploring the Frontier of Vision-Language Models: A Survey of Current  
  2. `2604.00086` Hierarchical Pre-Training of Vision Encoders with Large Language Model 
  3. `2507.07104` Vision-Language-Vision Auto-Encoder: Scalable Knowledge Distillation f 
  4. `2302.11713` Can Pre-trained Vision and Language Models Answer Visual Information-S 
- **Also relevant (add ids, space-separated):** 

---

### [DROP] rare-2404.07214

- **verdict:** `MISS`  ·  **source rank:** `None`  ·  **tag:** `rare`
- **query:** benchmark datasets for vision-language models
- **labeled:** `2404.07214` — Exploring the Frontier of Vision-Language Models: A Survey of Current Methodologies and Future Directions
- **retrieved top-5:**
  1. `2205.15237` VLUE: A Multi-Task Benchmark for Evaluating Vision-Language Models 
  2. `2406.12384` VRSBench: A Versatile Vision-Language Benchmark Dataset for Remote Sen 
  3. `2411.19103` VARCO-VISION: Expanding Frontiers in Korean Vision-Language Models 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2307.12114

- **verdict:** `AMBIG`  ·  **source rank:** `4`  ·  **tag:** `rare`
- **query:** evaluation of chatgpt on clinical biomedical nlp tasks
- **labeled:** `2307.12114` — A Zero-shot and Few-shot Study of Instruction-Finetuned Large Language Models Applied to Clinical and Biomedical Tasks
- **retrieved top-5:**
  1. `2403.13369` Clinical information extraction for Low-resource languages with Few-sh 
  2. `2402.15010` How Important Is Tokenization in French Medical Masked Language Models 
  3. `2306.12174` OphGLM: Training an Ophthalmology Large Language-and-Vision Assistant  
  4. `2307.12114` A Zero-shot and Few-shot Study of Instruction-Finetuned Large Language **<-- labeled**
- **Also relevant (add ids, space-separated):** 2403.13369 2306.12174

---

### [KEEP] paraphrase-1911.12377

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** how to improve navigation agents using multimodal attention for low-level tasks
- **labeled:** `1911.12377` — Multimodal Attention Networks for Low-Level Vision-and-Language Navigation
- **retrieved top-5:**
  1. `1911.12377` Multimodal Attention Networks for Low-Level Vision-and-Language Naviga **<-- labeled**
  2. `2412.09082` Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark a 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-1909.10225

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** report on gender diversity initiatives at computer vision conferences
- **labeled:** `1909.10225` — WiCV 2019: The Sixth Women In Computer Vision Workshop
- **retrieved top-5:**
  1. `1909.10225` WiCV 2019: The Sixth Women In Computer Vision Workshop **<-- labeled**
  2. `2104.08666` Worst of Both Worlds: Biases Compound in Pre-trained Vision-and-Langua 
  3. `2607.26886` Hearsay: Vision-Language Medical Diagnoses Without an Image 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2307.12114

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** how do leading instruction-tuned language models perform in clinical and biomedical nlp tasks without task-specific training?
- **labeled:** `2307.12114` — A Zero-shot and Few-shot Study of Instruction-Finetuned Large Language Models Applied to Clinical and Biomedical Tasks
- **retrieved top-5:**
  1. `2307.12114` A Zero-shot and Few-shot Study of Instruction-Finetuned Large Language **<-- labeled**
  2. `2403.13369` Clinical information extraction for Low-resource languages with Few-sh 
  3. `2312.10793` Demystifying Instruction Mixing for Fine-tuning Large Language Models 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2404.07214

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** review of visual language models capabilities and future research directions
- **labeled:** `2404.07214` — Exploring the Frontier of Vision-Language Models: A Survey of Current Methodologies and Future Directions
- **retrieved top-5:**
  1. `2404.07214` Exploring the Frontier of Vision-Language Models: A Survey of Current  **<-- labeled**
  2. `2504.09480` Vision-Language Model for Object Detection and Segmentation: A Review  
  3. `2411.19103` VARCO-VISION: Expanding Frontiers in Korean Vision-Language Models 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2209.07098

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** how does multi-modal learning improve medical image and text understanding using masked autoencoders?
- **labeled:** `2209.07098` — Multi-Modal Masked Autoencoders for Medical Vision-and-Language Pre-Training
- **retrieved top-5:**
  1. `2209.07098` Multi-Modal Masked Autoencoders for Medical Vision-and-Language Pre-Tr **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2104.04167

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** how to improve navigation using object-aware language models in indoor environments
- **labeled:** `2104.04167` — The Road to Know-Where: An Object-and-Room Informed Sequential BERT for Indoor Vision-Language Navigation
- **retrieved top-5:**
  1. `2104.04167` The Road to Know-Where: An Object-and-Room Informed Sequential BERT fo **<-- labeled**
  2. `2508.05838` Integrating Vision Foundation Models with Reinforcement Learning for E 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-0407028

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** how does improving language modeling affect speech-based question answering systems
- **labeled:** `0407028` — Effects of Language Modeling on Speech-driven Question Answering
- **retrieved top-5:**
  1. `0407028` Effects of Language Modeling on Speech-driven Question Answering **<-- labeled**
  2. `2403.19838` Multi-Frame, Lightweight & Efficient Vision-Language Models for Questi 
  3. `1312.3005` One Billion Word Benchmark for Measuring Progress in Statistical Langu 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2505.21089

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** dataset for assessing damage from various disasters using satellite imagery
- **labeled:** `2505.21089` — DisasterM3: A Remote Sensing Vision-Language Dataset for Disaster Damage Assessment and Response
- **retrieved top-5:**
  1. `2505.21089` DisasterM3: A Remote Sensing Vision-Language Dataset for Disaster Dama **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2408.09720

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** new large pedestrian attribute recognition dataset with cross-domain images
- **labeled:** `2408.09720` — Pedestrian Attribute Recognition: A New Benchmark Dataset and A Large Language Model Augmented Framework
- **retrieved top-5:**
  1. `2408.09720` Pedestrian Attribute Recognition: A New Benchmark Dataset and A Large  **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2307.03135

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** how to improve small vision-language models for better out-of-distribution performance
- **labeled:** `2307.03135` — Distilling Large Vision-Language Model with Out-of-Distribution Generalizability
- **retrieved top-5:**
  1. `2307.03135` Distilling Large Vision-Language Model with Out-of-Distribution Genera **<-- labeled**
  2. `2404.07214` Exploring the Frontier of Vision-Language Models: A Survey of Current  
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2410.00982

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** method for improving vision-language models to better understand and describe critical traffic events
- **labeled:** `2410.00982` — ScVLM: Enhancing Vision-Language Model for Safety-Critical Event Understanding
- **retrieved top-5:**
  1. `2410.00982` ScVLM: Enhancing Vision-Language Model for Safety-Critical Event Under **<-- labeled**
  2. `2505.21089` DisasterM3: A Remote Sensing Vision-Language Dataset for Disaster Dama 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2403.17811

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** how do different compression techniques affect the fairness towards underrepresented groups in BERT models
- **labeled:** `2403.17811` — Are Compressed Language Models Less Subgroup Robust?
- **retrieved top-5:**
  1. `2403.17811` Are Compressed Language Models Less Subgroup Robust? **<-- labeled**
  2. `2503.21676` How do language models learn facts? Dynamics, curricula and hallucinat 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2502.09927

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** novel lightweight multimodal system for enterprise visual document analysis
- **labeled:** `2502.09927` — Granite Vision: a lightweight, open-source multimodal model for enterprise Intelligence
- **retrieved top-5:**
  1. `2502.09927` Granite Vision: a lightweight, open-source multimodal model for enterp **<-- labeled**
  2. `2310.14025` Large Language Models and Multimodal Retrieval for Visual Word Sense D 
  3. `2404.07214` Exploring the Frontier of Vision-Language Models: A Survey of Current  
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2010.01177

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** method for enhancing neural networks with adaptive frequency filtering
- **labeled:** `2010.01177` — Global Adaptive Filtering Layer for Computer Vision
- **retrieved top-5:**
  1. `2010.01177` Global Adaptive Filtering Layer for Computer Vision **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2309.04041

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** how do large vision-language models connect language with real-world objects and concepts
- **labeled:** `2309.04041` — Evaluation and Enhancement of Semantic Grounding in Large Vision-Language Models
- **retrieved top-5:**
  1. `2309.04041` Evaluation and Enhancement of Semantic Grounding in Large Vision-Langu **<-- labeled**
  2. `2404.07214` Exploring the Frontier of Vision-Language Models: A Survey of Current  
  3. `2507.07104` Vision-Language-Vision Auto-Encoder: Scalable Knowledge Distillation f 
  4. `2303.10093` Investigating the Role of Attribute Context in Vision-Language Models  
  5. `2302.11713` Can Pre-trained Vision and Language Models Answer Visual Information-S 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2510.22370

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** novel reinforcement learning approach combining visual language understanding with geometric states for autonomous lane keeping
- **labeled:** `2510.22370` — BLIP-FusePPO: A Vision-Language Deep Reinforcement Learning Framework for Lane Keeping in Autonomous Vehicles
- **retrieved top-5:**
  1. `2510.22370` BLIP-FusePPO: A Vision-Language Deep Reinforcement Learning Framework  **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2504.06925

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** how effective are modern vision-language models in identifying food items from images
- **labeled:** `2504.06925` — Are Vision-Language Models Ready for Dietary Assessment? Exploring the Next Frontier in AI-Powered Food Image Recognition
- **retrieved top-5:**
  1. `2504.06925` Are Vision-Language Models Ready for Dietary Assessment? Exploring the **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2103.04037

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** reviewing progress and future directions for transformers in language and vision tasks
- **labeled:** `2103.04037` — Perspectives and Prospects on Transformer Architecture for Cross-Modal Tasks with Language and Vision
- **retrieved top-5:**
  1. `2103.04037` Perspectives and Prospects on Transformer Architecture for Cross-Modal **<-- labeled**
  2. `2504.09480` Vision-Language Model for Object Detection and Segmentation: A Review  
  3. `2210.09263` Vision-Language Pre-training: Basics, Recent Advances, and Future Tren 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2412.09082

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** new approach for long-term navigation planning and evaluation in complex environments
- **labeled:** `2412.09082` — Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method
- **retrieved top-5:**
  1. `2412.09082` Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark a **<-- labeled**
  2. `2411.18711` Evaluating Vision-Language Models as Evaluators in Path Planning 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2306.12174

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** method for improving medical image analysis using multimodal language models
- **labeled:** `2306.12174` — OphGLM: Training an Ophthalmology Large Language-and-Vision Assistant based on Instructions and Dialogue
- **retrieved top-5:**
  1. `2306.12174` OphGLM: Training an Ophthalmology Large Language-and-Vision Assistant  **<-- labeled**
  2. `2410.08397` VoxelPrompt: A Vision Agent for End-to-End Medical Image Analysis 
  3. `2604.00086` Hierarchical Pre-Training of Vision Encoders with Large Language Model 
  4. `2209.07118` Align, Reason and Learn: Enhancing Medical Vision-and-Language Pre-tra 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2506.18985

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** method for explaining visual attention in large vision-language models
- **labeled:** `2506.18985` — GLIMPSE: Holistic Cross-Modal Explainability for Large Vision-Language Models
- **retrieved top-5:**
  1. `2506.18985` GLIMPSE: Holistic Cross-Modal Explainability for Large Vision-Language **<-- labeled**
  2. `2503.21676` How do language models learn facts? Dynamics, curricula and hallucinat 
  3. `2604.00086` Hierarchical Pre-Training of Vision Encoders with Large Language Model 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2303.10093

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** how does additional information about objects affect vision-language models for image analysis tasks
- **labeled:** `2303.10093` — Investigating the Role of Attribute Context in Vision-Language Models for Object Recognition and Detection
- **retrieved top-5:**
  1. `2303.10093` Investigating the Role of Attribute Context in Vision-Language Models  **<-- labeled**
  2. `2605.22903` Seeing without Looking: Do Vision-Language Benchmarks Really Test Visi 
  3. `2404.07214` Exploring the Frontier of Vision-Language Models: A Survey of Current  
  4. `2201.05729` CLIP-TD: CLIP Targeted Distillation for Vision-Language Tasks 
  5. `2409.02228` Unforgettable Generalization in Language Models 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2603.20985

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** evaluating medical vision-language models for safety beyond paraphrase consistency
- **labeled:** `2603.20985` — Consistent but Dangerous: Per-Sample Safety Classification Reveals False Reliability in Medical Vision-Language Models
- **retrieved top-5:**
  1. `2603.20985` Consistent but Dangerous: Per-Sample Safety Classification Reveals Fal **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2508.05838

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** how to improve robot object interaction using vision models and reinforcement learning
- **labeled:** `2508.05838` — Integrating Vision Foundation Models with Reinforcement Learning for Enhanced Object Interaction
- **retrieved top-5:**
  1. `2508.05838` Integrating Vision Foundation Models with Reinforcement Learning for E **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2409.11353

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** methodology for assessing and reducing false information generated by large language models
- **labeled:** `2409.11353` — THaMES: An End-to-End Tool for Hallucination Mitigation and Evaluation in Large Language Models
- **retrieved top-5:**
  1. `2409.11353` THaMES: An End-to-End Tool for Hallucination Mitigation and Evaluation **<-- labeled**
  2. `2508.19294` Object Detection with Multimodal Large Vision-Language Models: An In-d 
  3. `2207.08179` End-to-End Spoken Language Understanding: Performance analyses of a vo 
  4. `2512.08480` Soft Inductive Bias Approach via Explicit Reasoning Perspectives in In 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2603.23521

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** dataset for improving vision language models with indian languages
- **labeled:** `2603.23521` — Chitrakshara: A Large Multilingual Multimodal Dataset for Indian languages
- **retrieved top-5:**
  1. `2603.23521` Chitrakshara: A Large Multilingual Multimodal Dataset for Indian langu **<-- labeled**
  2. `2404.07214` Exploring the Frontier of Vision-Language Models: A Survey of Current  
  3. `2411.19103` VARCO-VISION: Expanding Frontiers in Korean Vision-Language Models 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2603.15969

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** system for identifying regional dialects of a multilingual language
- **labeled:** `2603.15969` — Robust Language Identification for Romansh Varieties
- **retrieved top-5:**
  1. `2603.15969` Robust Language Identification for Romansh Varieties **<-- labeled**
  2. `1610.00031` Discriminating Similar Languages: Evaluations and Explorations 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2105.14897

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `paraphrase`
- **query:** how to use natural language descriptions for finding vehicles in traffic management
- **labeled:** `2105.14897` — Connecting Language and Vision for Natural Language-Based Vehicle Retrieval
- **retrieved top-5:**
  1. `2105.14897` Connecting Language and Vision for Natural Language-Based Vehicle Retr **<-- labeled**
  2. `2410.00982` ScVLM: Enhancing Vision-Language Model for Safety-Critical Event Under 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] paraphrase-2410.08397

- **verdict:** `OK`  ·  **source rank:** `2`  ·  **tag:** `paraphrase`
- **query:** method integrating language model and vision network for medical imaging tasks
- **labeled:** `2410.08397` — VoxelPrompt: A Vision Agent for End-to-End Medical Image Analysis
- **retrieved top-5:**
  1. `2209.07118` Align, Reason and Learn: Enhancing Medical Vision-and-Language Pre-tra 
  2. `2410.08397` VoxelPrompt: A Vision Agent for End-to-End Medical Image Analysis **<-- labeled**
  3. `2306.12174` OphGLM: Training an Ophthalmology Large Language-and-Vision Assistant  
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-1911.12377

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** PTA architecture for R2R and R4R benchmarks
- **labeled:** `1911.12377` — Multimodal Attention Networks for Low-Level Vision-and-Language Navigation
- **retrieved top-5:**
  1. `1911.12377` Multimodal Attention Networks for Low-Level Vision-and-Language Naviga **<-- labeled**
  2. `2411.16407` A Study on Unsupervised Domain Adaptation for Semantic Segmentation in 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-1909.10225

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** WiCV 2019 workshop report
- **labeled:** `1909.10225` — WiCV 2019: The Sixth Women In Computer Vision Workshop
- **retrieved top-5:**
  1. `1909.10225` WiCV 2019: The Sixth Women In Computer Vision Workshop **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2209.07098

- **verdict:** `OK`  ·  **source rank:** `3`  ·  **tag:** `rare`
- **query:** M^3AE medical vision-and-language pre-training
- **labeled:** `2209.07098` — Multi-Modal Masked Autoencoders for Medical Vision-and-Language Pre-Training
- **retrieved top-5:**
  1. `2209.07118` Align, Reason and Learn: Enhancing Medical Vision-and-Language Pre-tra 
  2. `2210.09263` Vision-Language Pre-training: Basics, Recent Advances, and Future Tren 
  3. `2209.07098` Multi-Modal Masked Autoencoders for Medical Vision-and-Language Pre-Tr **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2104.04167

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** ORIST indoor vision-language navigation
- **labeled:** `2104.04167` — The Road to Know-Where: An Object-and-Room Informed Sequential BERT for Indoor Vision-Language Navigation
- **retrieved top-5:**
  1. `2104.04167` The Road to Know-Where: An Object-and-Room Informed Sequential BERT fo **<-- labeled**
  2. `2507.07104` Vision-Language-Vision Auto-Encoder: Scalable Knowledge Distillation f 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-0407028

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** NTCIR QA test collection performance evaluation
- **labeled:** `0407028` — Effects of Language Modeling on Speech-driven Question Answering
- **retrieved top-5:**
  1. `0407028` Effects of Language Modeling on Speech-driven Question Answering **<-- labeled**
  2. `2307.12114` A Zero-shot and Few-shot Study of Instruction-Finetuned Large Language 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2505.21089

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** DisasterM3 remote sensing vision-language dataset
- **labeled:** `2505.21089` — DisasterM3: A Remote Sensing Vision-Language Dataset for Disaster Damage Assessment and Response
- **retrieved top-5:**
  1. `2505.21089` DisasterM3: A Remote Sensing Vision-Language Dataset for Disaster Dama **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2408.09720

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** LLM-PAR framework for pedestrian attribute recognition
- **labeled:** `2408.09720` — Pedestrian Attribute Recognition: A New Benchmark Dataset and A Large Language Model Augmented Framework
- **retrieved top-5:**
  1. `2408.09720` Pedestrian Attribute Recognition: A New Benchmark Dataset and A Large  **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2307.03135

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** open-vocabulary out-of-distribution generalization in model distillation
- **labeled:** `2307.03135` — Distilling Large Vision-Language Model with Out-of-Distribution Generalizability
- **retrieved top-5:**
  1. `2307.03135` Distilling Large Vision-Language Model with Out-of-Distribution Genera **<-- labeled**
  2. `2201.05729` CLIP-TD: CLIP Targeted Distillation for Vision-Language Tasks 
  3. `2402.09977` Fast Vocabulary Transfer for Language Model Compression 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2410.00982

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** ScVLM benchmark for safety-critical event understanding
- **labeled:** `2410.00982` — ScVLM: Enhancing Vision-Language Model for Safety-Critical Event Understanding
- **retrieved top-5:**
  1. `2410.00982` ScVLM: Enhancing Vision-Language Model for Safety-Critical Event Under **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2403.17811

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** benchmark for subgroup robustness in compressed language models
- **labeled:** `2403.17811` — Are Compressed Language Models Less Subgroup Robust?
- **retrieved top-5:**
  1. `2403.17811` Are Compressed Language Models Less Subgroup Robust? **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2502.09927

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** Granite Vision safety classification approach
- **labeled:** `2502.09927` — Granite Vision: a lightweight, open-source multimodal model for enterprise Intelligence
- **retrieved top-5:**
  1. `2502.09927` Granite Vision: a lightweight, open-source multimodal model for enterp **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2010.01177

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** Global Adaptive Filtering Layer
- **labeled:** `2010.01177` — Global Adaptive Filtering Layer for Computer Vision
- **retrieved top-5:**
  1. `2010.01177` Global Adaptive Filtering Layer for Computer Vision **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2310.14025

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** Chain-of-Thought prompting for VWSD
- **labeled:** `2310.14025` — Large Language Models and Multimodal Retrieval for Visual Word Sense Disambiguation
- **retrieved top-5:**
  1. `2310.14025` Large Language Models and Multimodal Retrieval for Visual Word Sense D **<-- labeled**
  2. `2412.09082` Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark a 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2109.01134

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** CoOp vision-language model adaptation
- **labeled:** `2109.01134` — Learning to Prompt for Vision-Language Models
- **retrieved top-5:**
  1. `2109.01134` Learning to Prompt for Vision-Language Models **<-- labeled**
  2. `2202.09061` VLP: A Survey on Vision-Language Pre-training 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2309.04041

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** fine-grained semantic grounding assessment LVLMs
- **labeled:** `2309.04041` — Evaluation and Enhancement of Semantic Grounding in Large Vision-Language Models
- **retrieved top-5:**
  1. `2309.04041` Evaluation and Enhancement of Semantic Grounding in Large Vision-Langu **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2510.22370

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** BLIP-FusePPO framework for autonomous vehicle lane keeping
- **labeled:** `2510.22370` — BLIP-FusePPO: A Vision-Language Deep Reinforcement Learning Framework for Lane Keeping in Autonomous Vehicles
- **retrieved top-5:**
  1. `2510.22370` BLIP-FusePPO: A Vision-Language Deep Reinforcement Learning Framework  **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2504.06925

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** Expert-Weighted Recall for VLMs in Food Recognition
- **labeled:** `2504.06925` — Are Vision-Language Models Ready for Dietary Assessment? Exploring the Next Frontier in AI-Powered Food Image Recognition
- **retrieved top-5:**
  1. `2504.06925` Are Vision-Language Models Ready for Dietary Assessment? Exploring the **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2103.04037

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** visuolinguistic cross-modal tasks transformer review
- **labeled:** `2103.04037` — Perspectives and Prospects on Transformer Architecture for Cross-Modal Tasks with Language and Vision
- **retrieved top-5:**
  1. `2103.04037` Perspectives and Prospects on Transformer Architecture for Cross-Modal **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2412.09082

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** LHPR-VLN benchmark vision-language navigation
- **labeled:** `2412.09082` — Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method
- **retrieved top-5:**
  1. `2412.09082` Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark a **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2306.12174

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** ophthalmic multimodal instruction-following dataset
- **labeled:** `2306.12174` — OphGLM: Training an Ophthalmology Large Language-and-Vision Assistant based on Instructions and Dialogue
- **retrieved top-5:**
  1. `2306.12174` OphGLM: Training an Ophthalmology Large Language-and-Vision Assistant  **<-- labeled**
  2. `2306.09265` LVLM-eHub: A Comprehensive Evaluation Benchmark for Large Vision-Langu 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2506.18985

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** GLIMPSE cross-modal explainability framework
- **labeled:** `2506.18985` — GLIMPSE: Holistic Cross-Modal Explainability for Large Vision-Language Models
- **retrieved top-5:**
  1. `2506.18985` GLIMPSE: Holistic Cross-Modal Explainability for Large Vision-Language **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2303.10093

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** attribute context in open-vocabulary object detection
- **labeled:** `2303.10093` — Investigating the Role of Attribute Context in Vision-Language Models for Object Recognition and Detection
- **retrieved top-5:**
  1. `2303.10093` Investigating the Role of Attribute Context in Vision-Language Models  **<-- labeled**
  2. `2504.09480` Vision-Language Model for Object Detection and Segmentation: A Review  
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2603.20985

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** per-sample safety taxonomy in medical VLMs
- **labeled:** `2603.20985` — Consistent but Dangerous: Per-Sample Safety Classification Reveals False Reliability in Medical Vision-Language Models
- **retrieved top-5:**
  1. `2603.20985` Consistent but Dangerous: Per-Sample Safety Classification Reveals Fal **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2508.05838

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** AI2-THOR environment with SAM and PPO
- **labeled:** `2508.05838` — Integrating Vision Foundation Models with Reinforcement Learning for Enhanced Object Interaction
- **retrieved top-5:**
  1. `2508.05838` Integrating Vision Foundation Models with Reinforcement Learning for E **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2409.11353

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** THaMES framework
- **labeled:** `2409.11353` — THaMES: An End-to-End Tool for Hallucination Mitigation and Evaluation in Large Language Models
- **retrieved top-5:**
  1. `2409.11353` THaMES: An End-to-End Tool for Hallucination Mitigation and Evaluation **<-- labeled**
  2. `2210.09263` Vision-Language Pre-training: Basics, Recent Advances, and Future Tren 
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2603.23521

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** Chitrakshara dataset series
- **labeled:** `2603.23521` — Chitrakshara: A Large Multilingual Multimodal Dataset for Indian languages
- **retrieved top-5:**
  1. `2603.23521` Chitrakshara: A Large Multilingual Multimodal Dataset for Indian langu **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2603.15969

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** Romansh Grischun classification
- **labeled:** `2603.15969` — Robust Language Identification for Romansh Varieties
- **retrieved top-5:**
  1. `2603.15969` Robust Language Identification for Romansh Varieties **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2105.14897

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** AIC2021 T5 CLV vehicle retrieval
- **labeled:** `2105.14897` — Connecting Language and Vision for Natural Language-Based Vehicle Retrieval
- **retrieved top-5:**
  1. `2105.14897` Connecting Language and Vision for Natural Language-Based Vehicle Retr **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

### [KEEP] rare-2410.08397

- **verdict:** `OK`  ·  **source rank:** `1`  ·  **tag:** `rare`
- **query:** VoxelPrompt benchmark for neuroimaging tasks
- **labeled:** `2410.08397` — VoxelPrompt: A Vision Agent for End-to-End Medical Image Analysis
- **retrieved top-5:**
  1. `2410.08397` VoxelPrompt: A Vision Agent for End-to-End Medical Image Analysis **<-- labeled**
- **Also relevant (add ids, space-separated):** 

---

