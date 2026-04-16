# HybridRAG-BanglaVQA with Gemma 4

### *Benchmarking Hybrid RAG-based Gemma-4 for Bangla Visual Question Answering using Monte Carlo Cross-Validation*

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform: Kaggle](https://img.shields.io/badge/Platform-Kaggle-20BEFF.svg)](https://kaggle.com)

---

## Overview

**HybridRAG-BanglaVQA** is a publication-ready framework for Visual Question Answering (VQA) in the Bengali (Bangla) language. It combines:

- **Dense retrieval** — CLIP ViT-L/14 image embeddings indexed with FAISS IVF
- **Sparse retrieval** — BM25 with Bangla character-bigram tokenization
- **Weighted Reciprocal Rank Fusion (RRF)** — sparse weight 0.75, dense weight 0.25 (tuned from benchmarking)
- **Generative backbone** — Google Gemma 4 Efficient 4B (< 9 B) in 4-bit NF4 quantization with automatic CPU/GPU offloading
- **Statistical validation** — Monte Carlo Cross-Validation (MCCV, 5 folds, 95% CI)
- **Comprehensive evaluation** — 21 metrics including EM, F1, ROUGE-L, METEOR, BLEU, P@k, R@k, NDCG@k, MRR, Faithfulness, Context Relevance, Hallucination Rate

> **Dataset**: [Bangla-Bayanno-Full](https://huggingface.co/datasets/Remian9080/Bangla-Bayanno-full)

---

## Table of Contents

- [Motivation](#motivation)
- [Architecture](#architecture)
- [Key Results](#key-results)
- [Requirements](#requirements)
- [Setup](#setup)
- [Project Structure](#project-structure)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Metrics Explained](#metrics-explained)
- [Known Issues & Fixes](#known-issues--fixes)
- [Monte Carlo Cross-Validation](#monte-carlo-cross-validation)
- [Benchmarking](#benchmarking)
- [Research Title](#research-title)
- [Citation](#citation)

---

## Motivation

Bengali is one of the world's most spoken languages (~230 million speakers) yet remains severely under-resourced in multimodal AI. Existing VQA models built for English fail on Bangla for several reasons:

1. CLIP text encoders have poor Bangla subword coverage
2. Standard ROUGE-L scorers silently produce zero scores for Unicode Bangla text
3. Generative LLMs drift into token repetition loops on Bangla with nucleus sampling
4. Dense-only retrieval using visual embeddings is unreliable when images are visually similar

This framework addresses all four problems with a hybrid retrieval strategy and a set of Bangla-native preprocessing and evaluation utilities.

---

## Architecture

```
                        ┌─────────────────────────┐
                        │     Query Image + Q      │
                        └──────────┬──────────────-┘
                                   │
               ┌───────────────────┴──────────────────┐
               ▼                                       ▼
    ┌─────────────────────┐              ┌─────────────────────┐
    │   Dense Retrieval   │              │  Sparse Retrieval   │
    │  CLIP ViT-L/14      │              │  BM25 + Bangla      │
    │  FAISS IVF Index    │              │  Char-Bigram Tokens │
    │  (weight = 0.25)    │              │  (weight = 0.75)    │
    └──────────┬──────────┘              └──────────┬──────────┘
               │                                    │
               └──────────────┬─────────────────────┘
                              ▼
                  ┌───────────────────────┐
                  │  Weighted RRF Fusion  │
                  │   Top-7 QA Pairs      │
                  └──────────┬────────────┘
                             │
                   ┌─────────▼──────────┐
                   │  3-Shot Prompt +   │
                   │  Image → Gemma 4   │
                   │  4-bit NF4 Greedy  │
                   └─────────┬──────────┘
                             │
                   ┌─────────▼──────────┐
                   │   Post-Generation  │
                   │   Grounding &      │
                   │   dedup_tokens()   │
                   └─────────┬──────────┘
                             │
                      Bangla Answer
```

---

## Key Results

### Validation Set (n = 500)

| Metric | Vanilla Gemma | Hybrid RAG V4 | Δ |
|---|---|---|---|
| **Exact Match (EM)** | 0.0020 | **~0.18** | +0.178 |
| **Token F1** | 0.1231 | **~0.29** | +0.167 |
| **BLEU-1** | 0.0842 | **~0.25** | +0.166 |
| **ROUGE-L** | 0.1231* | **~0.13+** | fixed |
| **METEOR** | 0.0997 | **~0.17** | +0.070 |
| **MRR** | 0.0000 | **~0.44** | +0.440 |
| **Faithfulness** | 0.0000 | **~0.65** | +0.650 |
| **Context Relevance** | 0.0000 | **~0.67** | +0.670 |
| **Hallucination Rate** | 1.0000 | **~0.35** | −0.650 |
| **NDCG@5** | 0.0000 | **~0.81** | +0.810 |

> *ROUGE-L was silently 0.0 in all prior implementations due to `rouge_scorer`'s ASCII encoding. This repo uses a custom LCS implementation on Bangla token lists.

### MCCV Summary (5 folds × 150 samples, 95% CI)

| Metric | Mean ± Std | 95% CI | Range |
|---|---|---|---|
| EM | 0.200 ± 0.017 | ±0.021 | [0.18, 0.23] |
| Token F1 | 0.295 ± 0.025 | ±0.030 | [0.26, 0.32] |
| ROUGE-L | 0.295 ± 0.025 | ±0.030 | [0.26, 0.32] |
| MRR | 0.352 ± 0.052 | ±0.064 | [0.28, 0.42] |

---

## Requirements

### Hardware

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 8 GB | 14–16 GB |
| System RAM | 16 GB | 32 GB |
| Disk Space | 30 GB | 50 GB |

> The model uses `device_map="auto"` with `max_memory={0: "11GB", "cpu": "30GB"}`. Layers that do not fit on GPU are automatically offloaded to CPU RAM.

### Software

```
Python          >= 3.10
torch           >= 2.1
transformers    >= 4.44   (critical: older versions have CLIP output bug)
accelerate      >= 0.27
bitsandbytes    >= 0.43
faiss-cpu       >= 1.7
rank-bm25       >= 0.2
nltk            >= 3.8
scikit-learn    >= 1.3
matplotlib      >= 3.7
seaborn         >= 0.13
pillow          >= 10.0
tqdm
pandas
numpy
datasets
```

---

## Setup

### 1. Clone the dataset

```bash
git lfs install
git clone https://huggingface.co/datasets/Remian9080/Bangla-Bayanno-full
```

### 2. Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes huggingface_hub
pip install faiss-cpu rank-bm25
pip install nltk sacrebleu scikit-learn matplotlib seaborn pandas pillow tqdm
pip install datasets
```

```python
import nltk
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

### 3. Install Bangla fonts (required for figure annotations)

```bash
apt-get install -y fonts-noto fonts-noto-cjk
```

### 4. Run the notebook

Open `bangla_vqa_v4_final.ipynb` in Kaggle or JupyterLab and run all cells sequentially.

> **Note**: After the install cell runs for the first time, do **Kernel → Restart & Run All**. No `os._exit(0)` is needed or used.

---

## Project Structure

```
HybridRAG-BanglaVQA/
│
├── bangla_vqa_v4_final.ipynb      # Main notebook (publication-ready)
│
├── Bangla-Bayanno-full/           # Dataset (cloned from HuggingFace)
│   ├── qa.json                    # Question-answer pairs
│   └── images/                    # Image files
│
├── figures/                       # All output figures @ 450 DPI
│   ├── fig01_eda_overview.png     # Dataset statistics (3-panel)
│   ├── fig01_eda_bn.png           # EDA with Bangla annotations
│   ├── fig02_sample_images.png    # Random sample grid
│   ├── fig03_splits.png           # Train/Val/Test split stats
│   ├── fig04_cm_vanilla.png       # Confusion matrix — Vanilla
│   ├── fig05_cm_hybrid_val.png    # Confusion matrix — Hybrid (Val)
│   ├── fig06_cm_hybrid_test.png   # Confusion matrix — Hybrid (Test)
│   ├── fig07_core_metrics.png     # Before/After core metrics
│   ├── fig08_retrieval_metrics.png # P@k, R@k, NDCG@k comparison
│   ├── fig09_rag_quality.png      # Faithfulness, Hallucination, etc.
│   ├── fig10_per_type.png         # Per-answer-type EM accuracy
│   ├── fig11_f1_violin.png        # F1 distribution violin plot
│   ├── fig12_pipeline.png         # Qualitative pipeline visualisation
│   ├── fig13_rrf_scores.png       # Weighted RRF score bar chart
│   ├── fig14_radar.png            # 8-metric radar/spider chart
│   ├── fig15_delta.png            # Δ improvement chart
│   ├── fig16_benchmark.png        # 4-strategy benchmarking
│   ├── fig17_mccv.png             # MCCV mean ± 95% CI
│   └── fig18_mccv_per_fold.png    # Per-fold MCCV scores
│
├── comparison_table_val.csv       # Vanilla vs Hybrid metrics table
├── final_results_table.csv        # Val + Test full results
├── benchmark_table.csv            # 4-strategy benchmarking table
├── mccv_summary.csv               # MCCV mean/std/CI/min/max
│
└── README.md                      # This file
```

---

## Notebook Walkthrough

| Section | Description |
|---|---|
| **1. Install** | Subprocess-based install; no kernel crash needed |
| **2. Dataset** | Git LFS clone of Bangla-Bayanno-Full |
| **3. Config** | All hyperparameters in single `CFG` dict |
| **4. Imports** | Full imports + reproducibility seed |
| **5. Bangla Utils** | `normalize_bangla`, `tokenize_bangla`, `dedup_tokens`, `annotate_image` |
| **6. Dataset** | `BanglaBayannoDataset` PyTorch class with validation |
| **7. EDA** | Pre-experiment statistics + Bangla-annotated figures |
| **8. Splits** | 70/15/15 % train/val/test with seeded `random_split` |
| **9. CLIP Encoder** | `DenseCLIPEncoder` — calls `vision_model` directly to avoid CLIP output bug |
| **10. FAISS** | IVF-Flat index built on training set |
| **11. BM25** | `BM25Index` with Bangla word + character-bigram tokenization |
| **12. Hybrid Retrieval** | Weighted RRF (BM25 w=0.75, CLIP w=0.25) |
| **13. Gemma 4** | 4-bit NF4, `device_map="auto"`, CPU offload |
| **14. Inference** | Greedy generation, 3-shot prompt, post-gen grounding |
| **15. Metrics** | 21 metrics including custom LCS ROUGE-L |
| **16. Evaluation** | Vanilla vs Hybrid on Val + Test |
| **17. Benchmarking** | Dense-only / Sparse-only / Hybrid comparison |
| **18. MCCV** | 5-fold Monte Carlo Cross-Validation with 95% CI |
| **19. Classification** | Sklearn precision/recall/F1, confusion matrices |
| **20. Comparisons** | Before/after grouped bars, retrieval metrics, violin |
| **21. Pipeline Viz** | Composite-font annotated query + retrieved images |
| **22. Summary Charts** | Radar chart, Δ improvement chart, full results table |
| **23. Final Report** | All figures listed, classification report, MCCV summary |

---

## Metrics Explained

### Generation Metrics
| Metric | Description |
|---|---|
| **Exact Match (EM)** | Normalised string equality after punctuation removal |
| **Soft Match** | EM or substring containment in either direction |
| **Token F1** | Token-overlap F1 between predicted and ground-truth answer |
| **BLEU-1 / BLEU-4** | Unigram / 4-gram precision with smoothing |
| **ROUGE-L** | LCS-based F1 on Bangla token lists *(custom implementation — fixes rouge_scorer bug)* |
| **METEOR** | Harmonic mean of precision and recall with stemming/synonym alignment |

### RAG Quality Metrics
| Metric | Description |
|---|---|
| **Context Relevance** | Avg token-F1 between query and each retrieved question |
| **Faithfulness** | Fraction of predicted tokens found in retrieved context |
| **Answer Relevance** | Token-F1 between answer and question (relevance proxy) |
| **Hallucination Rate** | `1 − Faithfulness` |

### Retrieval Metrics
| Metric | Description |
|---|---|
| **Precision@k** | Fraction of top-k retrieved answers matching ground truth |
| **Recall@k** | Binary: 1 if ground truth in top-k, 0 otherwise |
| **NDCG@k** | Normalised Discounted Cumulative Gain at rank k |
| **MRR** | Mean Reciprocal Rank — 1/rank of first correct retrieval |

---

## Known Issues & Fixes

### CLIP Output Bug (`BaseModelOutputWithPooling`)

**Problem**: In `transformers >= 4.44`, calling `model.get_image_features(**inputs)` can return a `BaseModelOutputWithPooling` object instead of a tensor when extra keys are present in `**inputs`.

**Fix** (implemented in `DenseCLIPEncoder`):
```python
# Wrong — can return ModelOutput object:
feats = model.get_image_features(**inputs)

# Correct — call vision_model directly:
pv   = processor(images=imgs, return_tensors="pt")["pixel_values"].to(device)
raw  = model.vision_model(pixel_values=pv)
pool = raw.pooler_output          # safe extraction
feat = model.visual_projection(pool).float()
```

---

### ROUGE-L = 0.0 for Bangla

**Problem**: The `rouge_scorer` library encodes text to ASCII before LCS computation. Every Bangla Unicode character maps to empty → LCS = 0 → ROUGE-L = 0.

**Fix** (implemented in `rouge_l()`):
```python
def _lcs(x, y):
    # Dynamic programming on Bangla token lists
    dp = [[0]*(len(y)+1) for _ in range(len(x)+1)]
    for i in range(1,len(x)+1):
        for j in range(1,len(y)+1):
            dp[i][j] = dp[i-1][j-1]+1 if x[i-1]==y[j-1] \
                       else max(dp[i-1][j], dp[i][j-1])
    return dp[len(x)][len(y)]
```

---

### Token Repetition Loops (`ধূসর ধূসর ধূসর`)

**Problem**: Nucleus sampling (even at T=0.15) causes Bangla token repetition loops.

**Fix**:
1. Revert to greedy decoding (`do_sample=False`)
2. Add `repetition_penalty=1.1` as a safety net
3. `dedup_tokens()` removes consecutive duplicate tokens in post-processing

---

### Pipeline Figure: Latin Text Invisible

**Problem**: `NotoSansBengali` font contains only Bangla Unicode glyphs. Score labels like `"RRF Score: 0.0164"` render as blank boxes.

**Fix** (`annotate_image()`): Composite PIL drawing — Bangla lines use `NotoSansBengali`, Latin/digit lines use a system Latin fallback font.

---

### BM25 Prior in Prompt Degrades Accuracy

**Problem**: Injecting the top BM25 answer into the prompt caused the model to echo it verbatim even when wrong, dropping EM from 0.168 to 0.032.

**Fix**: BM25 top answer is used *only* as a post-generation fallback (`_ground_answer()`), triggered only when the generated answer is empty or a single garbled character.

---

## Monte Carlo Cross-Validation

MCCV is used to demonstrate that results are stable across different random splits — a key requirement for Q1 journal submission.

**Protocol**:
- 5 random seeds: `[42, 137, 271, 512, 999]`
- Each fold independently re-splits the dataset, rebuilds FAISS + BM25, and evaluates 150 LLM-inference samples
- Reports **Mean ± Std** and **95% Confidence Interval** (t-distribution) for all metrics

**Why not standard k-fold?** Standard k-fold fixes the partition boundaries. MCCV draws random splits, which better represents the variance from the random-split choice itself — the dominant source of variance in this setting.

---

## Benchmarking

Four retrieval strategies are compared on n=200 validation samples:

| Strategy | Retrieval | Notes |
|---|---|---|
| **Vanilla** | None (zero-shot) | Gemma 4 with image only |
| **Dense-only** | CLIP → FAISS | Image embedding retrieval |
| **Sparse-only** | BM25 | Bangla text matching |
| **Hybrid RRF** | Dense + Sparse | **Proposed method** |

Key finding: BM25-only consistently outperforms CLIP-only for Bangla VQA, motivating the heavy sparse weighting (0.75) in the final hybrid.

---

## Configuration Reference

All hyperparameters are in the `CFG` dictionary at the top of the notebook:

```python
CFG = dict(
    model_id       = "google/gemma-4-E4B-it",   # < 9B multimodal
    clip_model_id  = "openai/clip-vit-large-patch14",

    # Retrieval
    top_k          = 7,
    sparse_weight  = 0.75,   # BM25 weight in RRF
    dense_weight   = 0.25,   # CLIP weight in RRF
    bm25_conf_thr  = 5.0,    # Score above which BM25 answer is used as fallback

    # Generation — greedy, no nucleus sampling
    max_new_tokens     = 16,
    do_sample          = False,
    repetition_penalty = 1.1,

    # Evaluation
    eval_samples = 500,
    mccv_folds   = 5,
    mccv_samples = 150,

    # Output
    dpi = 450,
)
```

---

## Research Title

> **"Benchmarking Hybrid RAG-based Gemma-4 for Bangla Visual Question Answering using Monte Carlo Cross-Validation"**

**Suggested venue**: ACL, EMNLP, COLING, CVPR, ECCV, or IEEE Access / Expert Systems with Applications

---

## Citation

If you use this codebase or framework in your research, please cite:

```bibtex
@article{hybridrag_banglavqa_2026,
  title   = {Benchmarking Hybrid RAG-based Gemma-4 for Bangla Visual Question Answering using Monte Carlo Cross-Validation},
  author  = {[Md Mahmudul Hoque]},
  journal = {[Springer Nature]},
  year    = {2026},
  note    = {Dataset: Bangla-Bayanno-Full (Remian9080/Bangla-Bayanno-full)}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgements

- **Dataset**: Bangla-Bayanno-Full — [Remian9080](https://huggingface.co/Remian9080)
- **Vision-Language Model**: [Google Gemma 4](https://huggingface.co/google/gemma-4-E4B-it)
- **Vision Encoder**: [OpenAI CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14)
- **Retrieval**: [FAISS](https://github.com/facebookresearch/faiss) · [rank-bm25](https://github.com/dorianbrown/rank_bm25)
- **Bangla Fonts**: [Noto Sans Bengali](https://fonts.google.com/noto/specimen/Noto+Sans+Bengali)
