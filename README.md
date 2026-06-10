# LLM Hallucination Reduction & Cost Optimisation

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![GCP](https://img.shields.io/badge/Google_Cloud-MLOps-blue?logo=googlecloud)
![License](https://img.shields.io/badge/License-MIT-green)

An MLOps pipeline on Google Cloud Platform for fine-tuning large language models using LoRA and QLoRA, reducing GPU VRAM requirements from 28GB to ~10GB through 8-bit quantisation, enabling deployment on hardware costing roughly 70–75% less while maintaining hallucination detection accuracy on the HaluEval benchmark.

---

## Table of Contents
- [Overview](#overview)
- [Results](#results)
- [Models Evaluated](#models-evaluated)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Usage](#usage)

---

## Overview

Deploying large language models at scale is expensive. Naive inference with full-precision models consumes significant GPU resources, making production deployment economically unviable for many organisations.

This project investigates whether quantisation and parameter-efficient fine-tuning (LoRA/QLoRA) can reduce inference costs without meaningfully degrading hallucination detection accuracy — evaluating across three task types and multiple model variants.

**Core questions:**
1. How does 4-bit and 8-bit quantisation affect hallucination detection accuracy on the HaluEval benchmark?
2. Does fine-tuning on in-domain and out-of-domain datasets improve accuracy over the base model?
3. How do different model sizes compare across QA, dialogue, and summarisation tasks?

**Evaluation benchmark:** HaluEval — 30,000 task-specific examples across question answering, knowledge-grounded dialogue, and text summarisation.

---

## Results

### Base Model Comparison

Evaluated on HaluEval across QA, dialogue, and summarisation tasks (10,000 samples per task).

| Model | QA Accuracy | Dialogue Accuracy | Summarisation Accuracy |
|-------|-------------|-------------------|------------------------|
| Llama-2-7B-chat | **55.3%** | **57.7%** | **46.8%** |
| Opt-1.3B | 52.0% | 48.0% | — |
| ChatGLM-6B | 50.7% | 49.8% | — |
| Falcon-7B | 50.2% | 40.0% | — |

> Llama-2-7B demonstrated the highest accuracy across all three tasks. Smaller models (ChatGLM, Falcon, OPT) performed near chance level (~50%), attributed to the subtlety of hallucinated answers in the HaluEval dataset. Summarisation evaluation was omitted for smaller models due to token length truncation significantly increasing evaluation time.

---

### QLoRA Fine-Tuning Results (Llama-2-7B, QA Task)

| Configuration | QA Accuracy | Notes |
|---------------|-------------|-------|
| Llama-2-7B base (FP16) | 55.3% | Baseline |
| Llama-2-7B Q8 (no fine-tuning) | 56.4% | 8-bit quantisation only |
| Llama-2-7B Q8 + QLoRA (HotpotQA) | **57.0%** | Best result — in-domain fine-tuning |
| Llama-2-7B Q8 + QLoRA (Google HAR) | 56.5% | Out-of-domain fine-tuning |
| Llama-2-7B Q4 (no fine-tuning) | 51.2% | 4-bit quantisation only |
| Llama-2-7B Q4 + QLoRA (HotpotQA) | 50.9% | Regression vs base |

**Key finding:** 8-bit quantisation provided the best balance between cost efficiency and performance — maintaining accuracy above the FP16 baseline with significantly reduced hardware requirements. 4-bit quantisation introduced regression, making it unsuitable for this evaluation task despite its greater resource savings.

---

### Cost vs Performance Trade-off

| Configuration | GPU VRAM Required | Accuracy (QA) |
|---------------|-------------------|---------------|
| FP16 (full precision) | 28GB+ | 55.3% |
| 8-bit quantisation | ~10GB | 56.4–57.0% |
| 4-bit quantisation | ~6GB | 50.9–51.2% |

> 8-bit QLoRA reduces VRAM requirements from 28GB to ~10GB, enabling deployment on hardware costing roughly 70–75% less than the A100-class GPUs required for full-precision inference — while matching or slightly exceeding the FP16 baseline on the HaluEval benchmark.

---

## Models Evaluated

| Model | Parameters | Quantisation Tested |
|-------|------------|---------------------|
| Llama-2-7B-chat | 7B | FP16, 8-bit, 4-bit |
| ChatGLM-6B | 6B | FP16 |
| Falcon-7B | 7B | FP16 |
| OPT-1.3B | 1.3B | FP16 |

All models accessed via Hugging Face Transformers. Llama-2 requires access approval at [huggingface.co/meta-llama](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

---

## Datasets

| Dataset | Task | Usage |
|---------|------|-------|
| HotpotQA | QA | In-domain fine-tuning |
| Google HAR (CFTriviaQA) | QA | Out-of-domain fine-tuning |
| OpenDialKG | Dialogue | Evaluation |
| CNN/DailyMail | Summarisation | Evaluation |

---

## Project Structure

```
Llama-2-HallucinationReduction-CostOptimization/
│
├── LLM-Halucination_/          # Evaluation module
│   └── evaluate.py             # HaluEval benchmark evaluation across tasks and models
│
├── analysis/                   # Analysis module
│   └── analyze.py              # Result analysis and reporting
│
├── scripts/                    # Pre-configured fine-tuning scripts per model
│
├── qlora.py                    # QLoRA/LoRA fine-tuning entry point
├── requirements.txt
└── README.md
```

---

## Architecture

```
Datasets (HotpotQA / Google HAR / OpenDialKG / CNN-DailyMail)
           ↓
  LoRA / QLoRA Fine-tuning (GCP — NVIDIA L4 GPU, CUDA 11.8)
           ↓
  Quantisation (FP16 → 8-bit → 4-bit via bitsandbytes)
           ↓
  HaluEval Benchmark Evaluation (QA / Dialogue / Summarisation)
           ↓
  Cross-model Accuracy Comparison & Cost Analysis
```

**Infrastructure:** Google Cloud Platform Compute Engine — NVIDIA L4 GPU, g2-standard-4 (4 vCPU, 16GB RAM), Deep Learning VM with CUDA 11.8.

---

## Getting Started

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 10GB+ VRAM for 8-bit, 6GB+ for 4-bit (NVIDIA L4 recommended)
- Google Cloud account with GPU quota, or equivalent local hardware
- Hugging Face account with Llama-2 access approved

### Installation

```bash
# Clone the repository
git clone https://github.com/callsomeoneelse/Llama-2-HallucinationReduction-CostOptimization.git
cd Llama-2-HallucinationReduction-CostOptimization

# Install dependencies
pip install -r requirements.txt

# Fix numpy compatibility (required)
pip uninstall numpy -y
pip install numpy==1.26.4

# Download required NLP modules
python -m nltk.downloader stopwords
python -m spacy download en_core_web_sm
```

### Hugging Face Authentication

Llama-2 requires access approval and authentication:

```bash
huggingface-cli login
# Enter your Hugging Face access token when prompted
# Only required once per machine
```

---

## Usage

### 1. Evaluation

```bash
cd LLM-Halucination_
python evaluate.py --task qa --model Llama2
```

**Task options:** `qa`, `dialogue`, `summarization`
**Model options:** `Llama2`, `ChatGLM`, `Falcon`

### 2. Analysis

```bash
cd analysis
python analyze.py --task qa --result ../evaluation/qa/qa_Llama-2-7b-chat-hf_results.json --category all
```

### 3. Fine-tuning

Pre-configured fine-tuning scripts for each model are in the `scripts/` directory.

```bash
# Run QLoRA fine-tuning with a model script
python qlora.py <insert script parameters>

# For long-running jobs on cloud instances (prevents session timeout)
nohup python qlora.py <insert script parameters>
```

Once complete, the fine-tuned adapter and checkpoints are saved to their own directory.

---

## GCP Setup

| Setting | Value |
|---------|-------|
| Region | asia-east-1 (a or c) |
| GPU | NVIDIA L4 |
| Machine type | g2-standard-4 (4 vCPU, 16GB RAM) |
| Boot disk | Deep Learning VM with CUDA 11.8 M125 |
| Disk size | 150–200GB balanced persistent |

> A GPU quota increase may be required on first use — typically approved within minutes via the GCP console. The model is deployed at half precision (FP16) as full precision requires 28GB+ VRAM, exceeding the L4's 24GB capacity.