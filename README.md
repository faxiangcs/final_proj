---
annotations_creators:
- expert-generated
language:
- en
task_categories:
- object-detection
- text-generation
- zero-shot-classification
- image-to-text
task_ids:
- open-domain-qa
- image-captioning
- multi-class-classification
- multi-label-classification
- multi-input-text-classification
multimodal:
- image
- text
tags:
- vision-language
- open-world
- rare-diseases
- brain-mri
- medical-imaging
- out-of-distribution
- generalization
- anomaly-localization
- captioning
- diagnostic-reasoning
- zero-shot
---

<table>
  <tr>
    <td><img src="https://huggingface.co/datasets/c-i-ber/Nova/resolve/main/logo.png" alt="NOVA Logo" width="40"></td>
    <td><h1 style="margin:0; padding-left: 10px;">NOVA: A Benchmark for Anomaly Localization and Clinical Reasoning in Brain MRI</h1></td>
  </tr>
</table>

> **An open-world generalization benchmark under clinical distribution shift**

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC--BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)  
[Dataset on 🤗 Hugging Face](https://huggingface.co/datasets/c-i-ber/Nova)  
*For academic, non-commercial use only*

---

## 🔖 Citation

If you find this dataset useful in your work, please consider citing it:

  ```bibtex
  @article{bercea2025nova,
    title={NOVA: A Benchmark for Anomaly Localization and Clinical Reasoning in Brain MRI},
    author={Bercea, Cosmin I. and Li, Jun and Raffler, Philipp and Riedel, Evamaria O. and Schmitzer, Lena and Kurz, Angela and Bitzer, Felix and Roßmüller, Paula and Canisius, Julian and Beyrle, Mirjam L. and others},
    journal={arXiv preprint arxiv:2505.14064},
    year={2025},
    note={Preprint. Under review.}
  }
  ```

[Paper](https://huggingface.co/papers/2505.14064)
[Evaluation Scripts](https://huggingface.co/c-i-ber/Nova_Evaluation)

---

## 📦 Usage

```python
from datasets import load_dataset
ds = load_dataset("parquet", data_files=f"hf://datasets/c-i-ber/Nova/data/nova-v1.parquet", split="train")
```

## 🧪 Try it out in Colab

You can explore the NOVA dataset directly in your browser using the interactive notebook below:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://huggingface.co/datasets/c-i-ber/Nova/blob/main/notebooks/nova_demo.ipynb)

Or open the notebook directly:
👉 [example.ipynb](https://huggingface.co/datasets/c-i-ber/Nova/blob/main/notebooks/nova-demo.ipynb)

---
## 💡 Motivation

Machine learning models in real-world clinical settings must detect and reason about anomalies they have **never seen during training**. Current benchmarks mostly focus on known, curated categories—collapsing evaluation back into a **closed-set** problem and overstating model robustness.

**NOVA** is the first benchmark designed as a **zero-shot, evaluation-only** setting for assessing how well models:

- Detect **rare, real-world anomalies**
- Generalize across diverse MRI protocols and acquisition settings
- Perform **multimodal reasoning** from image, text, and clinical context

It challenges foundation models and vision-language systems with what they *were not trained for*: the **unexpected**.

---

## 🧠 Dataset Overview

- **906 brain MRI slices**
- **281 rare neurological conditions**, spanning neoplastic, vascular, metabolic, congenital, and other pathologies
- **Real-world clinical heterogeneity** (unprocessed, long-tailed distribution)
- **Radiologist-written captions** and **double-blinded bounding boxes**
- **Clinical histories** and diagnostics for reasoning tasks

🛠️ All cases are **2D PNG slices**, sized 480×480, and are available under CC BY-NC-SA 4.0.

---

## 📊 Benchmark Tasks

NOVA captures the clinical diagnostic workflow through **three open-world tasks**:

### 🔍 1. Anomaly Localization  
Detect abnormal regions via bounding box prediction. Evaluated with:
- `mAP@30`, `mAP@50`, `mAP@[50:95]`
- True/false positive counts per case

### 📝 2. Image Captioning  
Generate structured radiology-style descriptions.
- Evaluated with Clinical/Modality F1, BLEU, METEOR
- Also assesses normal/abnormal classification

### 🧩 3. Diagnostic Reasoning  
Predict the correct diagnosis based on clinical history + image caption.  
- Evaluated with Top-1, Top-5 accuracy
- Label coverage and entropy analysis for long-tail reasoning

---

## 🧪 Model Performance (Stress Test)

| Task                  | Top Model (2025)   | Top Metric         | Score    |
|-----------------------|--------------------|---------------------|----------|
| Anomaly Localization | Qwen2.5-VL-72B     | mAP@50              | 24.5%    |
| Image Captioning     | Gemini 2.0 Flash   | Clinical Term F1    | 19.8%    |
| Diagnostic Reasoning | GPT-4o             | Top-1 Accuracy      | 24.2%    |

Even top-tier foundation models fail under this **open-world generalization** benchmark.

---


## ⚠️ Intended Use

NOVA is intended **strictly for evaluation**. Each case has a unique diagnosis, preventing leakage and forcing **true zero-shot testing**.

Do **not** fine-tune on this dataset.  
Ideal for:
- Vision-language model benchmarking
- Zero-shot anomaly detection
- Rare disease generalization

---

## 📬 Contact

Stay tuned for the **public leaderboard** coming soon.

---