# Imaginative Perception Token

## Imaginative Perception Tokens (IPT) for Spatial Reasoning

> *We train a unified VLM to generate **Imaginative Perception Tokens** — intermediate visual representations of what the model would perceive under an unseen spatial configuration — and show that imagination supervision consistently beats text-based chain-of-thought, even when no image is generated at inference.*

<p align="center">
  <img src="assets/overview.png" width="90%" alt="Imaginative Perception Token — tasks and data overview"/>
</p>

This repository extends the [ThinkMorph](https://github.com/ThinkMorph/ThinkMorph) codebase with training and evaluation for three spatial imagination tasks:

| Task | What's imagined | Question form |
|------|-----------------|---------------|
| **Perspective Taking (PET)** | Novel-viewpoint scene | *"If you move to the marked position and turn left, will the chair be on your left or right?"* |
| **Path Tracing (PT)** | Sideview along a path | *"If you walk along the marked path, which object will you see?"* |
| **Multiview Counting (MVC)** | Top-down BEV map | *"How many objects are in the scene given these views?"* |

<p align="center">
  <a href="https://huggingface.co/collections/weikaih/imaginative-perception-token-mvc-pet-pt-datasets-6a15f80e0fcef43bd0c50aba">
    <img src="https://img.shields.io/badge/IPT-Datasets-yellow?logo=huggingface&logoColor=yellow" alt="IPT Datasets"/>
  </a>
  <a href="https://github.com/weikaih04/Imaginative-Perception-Token-Eval">
    <img src="https://img.shields.io/badge/IPT-Eval-purple?logo=eval&logoColor=black" alt="IPT Eval"/>
  </a>
</p>

---

## Datasets at a glance

All datasets live in one [collection](https://huggingface.co/collections/weikaih/imaginative-perception-token-mvc-pet-pt-datasets-6a15f80e0fcef43bd0c50aba). Each task ships three training variants and its own eval sets:

```
Imaginative Perception Token  (weikaih/imaginative-perception-token-mvc-pet-pt-datasets)
│
├── MVC · Multiview Counting
│   ├── train ─ mvc-ipt · mvc-answeronly · mvc-textcot
│   └── eval  ─ multiview_eval (AI2-THOR HV) · messytable · scannet_counting
│
├── PET · Perspective Taking
│   ├── train ─ pet-ipt · pet-answeronly · pet-textcot
│   └── eval  ─ pet-eval-ai2thor · habitat_perspective_eval (HV) · vlmevalkit_tsv (SAT)
│
└── PT · Path Tracing
    ├── train ─ pt-ipt · pt-answeronly · pt-textcot
    └── eval  ─ pt-eval-ai2thor · pt-eval-real
```

**Training variants** — `ipt`: Imaginative Perception Tokens (Visual CoT, generates intermediate images); `textcot`: text-only chain-of-thought; `answeronly`: direct-answer data, serves both the label-only baseline and the answer-only half of Mixed training.

---

## Quick Start (IPT training)

### 1. Environment

```bash
git clone https://github.com/weikaih04/Imaginative-Perception-Token.git
cd Spatial-Imaginative-Token
conda create -n thinkmorph python=3.10 -y
conda activate thinkmorph
pip install -r requirements.txt
```

### 2. Download datasets

All 8 paper training datasets (≈30 GB after JPEG optimization) are available in a single [HuggingFace collection](https://huggingface.co/collections/weikaih/imaginative-perception-token-mvc-pet-pt-datasets-6a15f80e0fcef43bd0c50aba):

```bash
python scripts/download_spatial_datasets.py              # all 8 datasets (MVC + PET)
python scripts/download_spatial_datasets.py --task mvc   # MVC only (4 datasets)
python scripts/download_spatial_datasets.py --task pet   # PET only (4 datasets)
```

Parquets land in `data/training/<dataset_name>/`; `data/dataset_info.py` already points there.

### 3. Train — 4 end-to-end variants per task, matching paper main results

All variants start from `BAGEL-7B-MoT` and train for 25,000 steps. The mixed variants combine IPT and answer-only data 50/50 via dataloader-level mixing.

|  | **Label-only** | **+ Text CoT** | **+ IPT** *(Visual CoT)* | **+ Mixed Training** |
|---|---|---|---|---|
| **MVC** | `train_mvc_no_thought.sh` | `train_mvc_textcot.sh` | `train_mvc_ipt.sh` | `train_mvc_mixed.sh` |
| **PET** | `train_pet_no_thought.sh` | `train_pet_textcot.sh` | `train_pet_ipt.sh` | `train_pet_mixed.sh` |

```bash
bash scripts/train_mvc_ipt.sh    # MVC with imaginative perception tokens
bash scripts/train_pet_mixed.sh  # PET with 50/50 IPT + answer-only mix
```

VAE is automatically enabled for IPT / Mixed variants (which generate intermediate images) and disabled for Label-only / Text CoT (text outputs only) — controlled by the `enable_vae` flag in each YAML config.

### 4. Evaluate

Benchmarks are supported by our eval repo [SpatialReasoning_Eval](https://github.com/weikaih04/Imaginative-Perception-Token-Eval):

- **In-domain (AI2-THOR)**: `MVC_AI2Thor_ImaginativePerceptionToken`, `PET_AI2Thor_ImaginativePerceptionToken`
- **Different environment**: `PET_Habitat_ImaginativePerceptionToken`
- **OOD (similar tasks)**: `PET_SAT_ImaginativePerceptionToken`, `MVC_MessyTable_ImaginativePerceptionToken`, `MVC_ScanNet_ImaginativePerceptionToken`
- **OOD (other spatial)**: `MindCube_ImaginativePerceptionToken`, `AllAngles_ImaginativePerceptionToken`

All linked in the [HuggingFace collection](https://huggingface.co/collections/weikaih/imaginative-perception-token-mvc-pet-pt-datasets-6a15f80e0fcef43bd0c50aba).

---

## Headline Results

| Method | MVC (AI2-THOR) | PET (AI2-THOR) | PET (Habitat) |
|---|:---:|:---:|:---:|
| GPT-5 (zero-shot) | 53.5 | 79.8 | 69.3 |
| Bagel (label-only) | 63.9 | 97.5 | 82.0 |
| + Text CoT | 62.3 | 83.1 | 70.3 |
| **+ IPT** | **67.3** | 96.8 | 87.0 |
| **+ Mixed Training** | 62.3 | **97.8** | **87.7** |

IPT models are evaluated in *answer-only* mode — no image is generated at inference, yet the imagination targets during training strengthen internal spatial representations that transfer across environments.

---

<details>
<summary><b> Built on ThinkMorph (click to expand)</b></summary>

This repository is forked from [ThinkMorph](https://github.com/ThinkMorph/ThinkMorph). The original training infrastructure for interleaved text–image reasoning is preserved below.

<p align="center">
  <a href="https://thinkmorph.github.io/"><img src="https://img.shields.io/badge/ThinkMorph-Website-0A66C2?logo=safari&logoColor=white" alt="ThinkMorph Website"/></a>
  <a href="https://arxiv.org/abs/2510.27492"><img src="https://img.shields.io/badge/ThinkMorph-Paper-red?logo=arxiv&logoColor=red" alt="ThinkMorph Paper on arXiv"/></a>
  <a href="https://huggingface.co/ThinkMorph/ThinkMorph-7B"><img src="https://img.shields.io/badge/ThinkMorph-Model-yellow?logo=huggingface&logoColor=yellow" alt="ThinkMorph Model"/></a>
  <a href="https://huggingface.co/ThinkMorph"><img src="https://img.shields.io/badge/ThinkMorph-Dataset-yellow?logo=huggingface&logoColor=yellow" alt="ThinkMorph Dataset"/></a>
</p>

### ThinkMorph training data preparation

Original ThinkMorph trains on four interleaved-reasoning tasks: **Jigsaw Assembly**, **Spatial Navigation**, **Visual Search**, and **Chart Refocus** — all hosted at [huggingface.co/ThinkMorph](https://huggingface.co/ThinkMorph).

```python
from datasets import load_dataset
dataset = load_dataset("ThinkMorph/Jigsaw_Assembly", split="train")
dataset = load_dataset("ThinkMorph/Spatial_Navigation", split="train")
dataset = load_dataset("ThinkMorph/Visual_Search", split="train")
dataset = load_dataset("ThinkMorph/Chart_Refocus", split="train")
```

Edit `data/dataset_info.py` with your own data paths, edit `data/configs/example.yaml`, then launch the relevant `scripts/train_*.sh`. Example configs for the three ThinkMorph settings (`interleaved_reasoning`, `text_reasoning`, `thinkmorph`) live in `data/configs/`.

### ThinkMorph evaluation

Evaluation is in [VLMEvalKit_Thinkmorph](https://github.com/hychaochao/VLMEvalKit_Thinkmorph), supporting VSP, VisPuzzle, ChartQA, VStar, BLINK-J, MMVP, SAT, BLINK, and CV-Bench.

### ThinkMorph benchmarks

| Model | Size | VSP | VisPuzzle | ChartQA | VStar | BLINK-J | MMVP | SAT | BLINK | CV-Bench |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-4o | – | 33.50 | 43.75 | 76.34 | 61.78 | 72.67 | 84.67 | 28.00 | 60.28 | 75.61 |
| GPT-5 | – | 57.33 | 78.00 | 80.85 | 71.73 | 77.33 | 86.33 | 73.30 | 69.86 | 85.46 |
| Bagel | 7B | 0.83 | 35.00 | 61.82 | 55.49 | 67.33 | 70.33 | 44.67 | 47.66 | 76.03 |
| **ThinkMorph** | **7B** | **75.83** | **79.00** | 78.10 | 67.02 | 72.00 | 80.33 | 52.67 | 60.07 | 80.82 |

</details>

---

## Citation

```bibtex
@article{gu2025thinkmorph,
  title={ThinkMorph: Emergent Properties in Multimodal Interleaved Chain-of-Thought Reasoning},
  author={Gu, Jiawei and Hao, Yunzhuo and Wang, Huichen Will and Li, Linjie and Shieh, Michael Qizhe and Choi, Yejin and Krishna, Ranjay and Cheng, Yu},
  journal={arXiv preprint arXiv:2510.27492},
  year={2025}
}
```
