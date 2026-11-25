# GRPO with Verifiable Clinical Rewards for Structured Brain MRI Reporting

**Course:** IOTA 5201: RL for Intelligent Decision Making (Fall 2026) — **Path 2: Apply RL to a Problem of Your Choice**  
**Author:** Yan Faxiang · HKUST(GZ)  
**Report:** [paper/main.pdf](paper/main.pdf) — 6 pages, IEEE double-column conference format  
**Presentation video:** [PRESENTATION.md](PRESENTATION.md) — recording outline; add the final video link before submitting

---

## 1. Overview

Structured brain MRI reporting requires three coupled decisions: **where** the abnormality is, **what** it looks like, and **what diagnosis** to propose. This project asks whether Group Relative Policy Optimization (GRPO) with deterministic, verifiable clinical rewards improves this task beyond supervised fine-tuning (SFT), and how the strength of the SFT initialization changes the answer.

The policy is a vision-language model (Qwen3.5-9B or LLaVA-1.5-7B) with a LoRA adapter. A complete report is generated token by token; a terminal reward scores the finished report. The reward combines a one-to-one box-matching localization score at IoU 0.5 with a diagnosis score based on the final diagnosis, the differential list, and a small format check. No learned reward model is used.

**Main finding.** GRPO can refine an under-trained SFT checkpoint, but it does not replace stronger SFT or better data. The one positive result (early-start GRPO checkpoint 100) was selected after inspecting test results; the final checkpoint from the same run has a confidence interval that includes zero. Starting from the stronger SFT-201 checkpoint, GRPO shows no test improvement. Diagnosis and captioning do not improve reliably.

## 2. Results at a glance

Test split: 111 images from 36 held-out cases. Localization is raw (no post-hoc box calibration); diagnosis uses keyword alias matching. Full table, confidence intervals, and caveats are in [paper/main.pdf](paper/main.pdf).

| Policy | Loc F1@0.5 ↑ | Mean IoU ↑ | Diag Top-1 ↑ | Diag Top-5 ↑ |
|---|---:|---:|---:|---:|
| Center prior (heuristic) | 0.077 | 0.195 | — | — |
| Qwen3.5-9B, zero-shot | 0.068 | 0.218 | 0.117 | 0.180 |
| Qwen3.5-9B, SFT-50 (weak start) | 0.146 | 0.300 | 0.090 | 0.198 |
| Qwen3.5-9B, GRPO from SFT-50, ck100 | **0.216** | 0.344 | **0.135** | 0.234 |
| Qwen3.5-9B, GRPO from SFT-50, final ck266 | 0.191 | 0.328 | 0.117 | **0.243** |
| Qwen3.5-9B, SFT-201 (strong start) | **0.401** | **0.403** | 0.108 | 0.180 |
| Qwen3.5-9B, GRPO from SFT-201, ck266 | 0.390 | **0.403** | 0.099 | 0.171 |
| LLaVA-1.5-7B, SFT-50 | 0.054 | 0.152 | 0.072 | 0.099 |
| LLaVA-1.5-7B, GRPO from SFT-50, ck50 | 0.054 | 0.165 | 0.072 | 0.090 |

Key numbers:

- Weak start: F1@0.5 **0.146 → 0.216** (+0.071; nominal paired case-bootstrap 95% CI [+0.015, +0.125]) at ck100, but ck100 was chosen retrospectively. The final ck266 reaches 0.191 (CI [−0.028, +0.113]).
- Strong start: SFT-201 **0.401** vs. GRPO ck266 **0.390** (Δ −0.011; CI [−0.088, +0.078]). No detected gain, not proof of equivalence.
- 93.7% of test images (104/111) have a final diagnosis absent from the fine-tuning split.
- The mean fraction of GRPO groups with identical rewards is 23.7% (weak start), 17.7% (strong start), and 53.8% (LLaVA). Zero-variance groups contribute no reward-advantage gradient.

## 3. Method at a glance

- **Observation:** image `I`, clinical history `H`, and a fixed instruction `P`.
- **State:** `(I, H, P, y_<t)`, where `y_<t` is the generated prefix.
- **Action:** the next token. **Transition:** append the token. **Episode:** ends at EOS or 512 tokens.
- **Reward (terminal, deterministic):**
  - `R_loc = TP / (TP + FP + FN)` after coordinate auto-detection, clipping, NMS, and a three-box cap. This is a box-set Jaccard score; evaluation uses F1.
  - `R_diag = 0.95 * (0.7 * s_final + 0.3 * s_diff) + 0.05 * s_fmt`.
  - Total `R = R_loc + R_diag` in [0, 2].
- **Optimization:** GRPO with `G = 4` sampled completions, learning rate `5e-6`, `β = 0.001`, 266 optimizer steps, four RTX 3090 GPUs. SFT uses LoRA `r=16`, `α=32`, learning rate `5e-5`, three epochs.

## 4. Repository layout

```text
.
├── README.md                  # this file
├── requirements.txt           # core Python dependencies
├── PRESENTATION.md            # suggested 7-minute video outline
├── inference.sh               # swift infer wrapper
├── medical_reward.py          # GRPO reward plugin (localization + diagnosis)
├── eval_all_metric.py         # compatibility entry point for src/eval_nova.py
├── src/
│   ├── build_dataset.py       # build case-level train/val/test JSONL files
│   ├── audit_data.py          # structural and leakage audits
│   ├── nova_schema.py         # prompt, label, parser, box/diagnosis schema
│   ├── eval_nova.py           # localization, caption, diagnosis metrics
│   ├── infer_qwen_baseline.py # zero-shot Qwen baseline via OpenAI API
│   └── diagnosis_aliases.json # keyword aliases for diagnosis evaluation
├── scripts/
│   ├── README.md              # detailed experiment log and command reference
│   ├── setup_swift_env.sh     # reproduce the ms-swift environment
│   ├── common.sh              # shared defaults
│   ├── run_sft.sh             # SFT
│   ├── run_grpo.sh            # GRPO
│   ├── train.sh               # one entry point: sft | grpo
│   ├── run_eval.sh            # evaluation wrapper
│   ├── eval_grpo_checkpoints.sh
│   ├── run_baseline_qwen.sh
│   ├── fit_box_scale.py       # validation-only box calibration
│   ├── calibration_ci.py      # paired case-cluster bootstrap
│   ├── box_bias_report.py     # box-area bias diagnostics
│   ├── compare_checkpoints.py
│   ├── monitor_dashboard.py   # lightweight local training dashboard
│   ├── start_monitor.sh       # start/stop/status helper
│   └── make_submission_zip.sh # build the course .zip
├── tests/                     # 87 unit tests (unittest, no pytest required)
└── paper/
    ├── main.tex               # IEEEtran source
    ├── main.pdf               # compiled 6-page report
    ├── refs.bib
    ├── REPRODUCE.md           # exact overrides and artifact mapping
    ├── make_figure.py
    ├── IEEEtran.cls
    ├── IEEEtran.bst
    └── figures/
```

## 5. Quick start

The reported runs used Python 3.11.18, PyTorch 2.6.0 (CUDA 12.4), Transformers 5.20.2, and ms-swift 4.6.1 on four RTX 3090 GPUs (24 GB each). A single GPU works for inference and smoke tests.

```bash
# 1) Environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Alternative: reproduce the conda prefix used for the reported runs
# (set ENV_DIR to choose the install location)
bash scripts/setup_swift_env.sh
```

`transformers` must be 5.x: 4.x does not register the `qwen3_5` architecture. `ms-swift` 4.x also renamed `--train_type` to `--tuner_type`; the scripts already use the new flag.

### 5.1 Data

NOVA is a public, **evaluation-only** benchmark released under **CC BY-NC-SA 4.0**. This project fine-tunes on a filtered subset, which departs from the benchmark's intended use; the paper states this limitation. Please obtain NOVA from the official release and follow its license:

- Dataset: <https://huggingface.co/datasets/c-i-ber/Nova>
- Paper: <https://arxiv.org/abs/2505.14064>

The builder expects the CSV annotation layout used in this project:

```text
Nova/
├── case_metadata.csv      # clinical history, differential, final diagnosis
├── captions.csv           # radiologist captions
├── bboxes_gold.csv        # expert boxes
├── bboxes_raters.csv      # reader boxes (part of the release; not used here)
└── images/                # 480x480 PNG slices
```

Place the release under `Nova/` (or set `NOVA_ROOT`), then build the case-level split:

```bash
python src/build_dataset.py \
  --nova_root Nova \
  --image_root Nova/images \
  --out_dir data/nova_sft \
  --seed 42

python src/audit_data.py --data_dir data/nova_sft
```

This produces 164/35/36 cases (533/135/111 images). The JSONL files store absolute image paths; if the images live elsewhere, re-run the builder with `--image_root` rather than editing the JSONL by hand.

### 5.2 SFT

```bash
NPROC_PER_NODE=4 \
MODEL="$HOME/models/Qwen3.5-9B" \
DATA_DIR="$PWD/data/nova_sft" \
OUTPUT_DIR="$PWD/output/qwen_sft" \
LORA_RANK=16 LORA_ALPHA=32 FREEZE_VIT=true FREEZE_ALIGNER=false \
SFT_LR=5e-5 SFT_EPOCHS=3 SFT_BATCH_SIZE=1 SFT_GRAD_ACCUM=2 \
MAX_LENGTH=1024 ENABLE_THINKING=false SEED=42 \
SAVE_STEPS=25 EVAL_STEPS=25 LOGGING_STEPS=1 \
bash scripts/run_sft.sh
```

For LLaVA-1.5-7B, use `MODEL="$HOME/models/llava-1.5-7b-hf"` and `MAX_LENGTH=2048`. Use `DRY_RUN=1` to print the exact `swift` command without launching training.

### 5.3 GRPO

```bash
NPROC_PER_NODE=4 \
MODEL="$HOME/models/Qwen3.5-9B" \
ADAPTERS="$PWD/output/qwen_sft/checkpoint-50" \
DATA_DIR="$PWD/data/nova_sft" \
OUTPUT_DIR="$PWD/output/qwen_grpo" \
LORA_RANK=16 LORA_ALPHA=32 FREEZE_VIT=true FREEZE_ALIGNER=false \
GRPO_LR=5e-6 GRPO_EPOCHS=1 GRPO_BATCH_SIZE=1 GRPO_GRAD_ACCUM=2 \
NUM_GENERATIONS=4 TEMPERATURE=0.9 TOP_P=0.95 BETA=0.001 \
REWARD_WEIGHTS="1 1" MAX_LENGTH=2048 MAX_COMPLETION_LENGTH=512 \
ENABLE_THINKING=false GRPO_EVAL_STRATEGY=no SEED=42 \
SAVE_STEPS=25 LOGGING_STEPS=1 \
bash scripts/run_grpo.sh
```

The LoRA rank and alpha must match the SFT adapter. The reward plugin registers exactly two functions: `localization_reward` and `diagnosis_reward`.

### 5.4 Inference and evaluation

```bash
# Merge the adapter, or point inference.sh at the merged model
bash inference.sh

# Evaluate a prediction file (gold labels are read from the test file here)
python src/eval_nova.py \
  --pred result/qwen_grpo/checkpoint-100/pred.jsonl \
  --gold data/nova_sft/test.jsonl \
  --coord auto \
  --out result/qwen_grpo/checkpoint-100/report.json
```

`src/eval_nova.py` reports one-to-one multi-box precision/recall/F1 at IoU 0.3/0.5/0.75, mean IoU, a custom AP proxy, caption token F1 and BLEU-4, and diagnosis Top-1/Top-5 under strict, lenient, and keyword matching. The AP proxy is **not** standard COCO mAP.

### 5.5 Tests

```bash
python3 -m unittest discover -s tests -v
```

The suite contains 87 tests covering the schema, parser, rewards, evaluator, dataset format, and dashboard helpers. No GPU is needed.

### 5.6 Monitoring dashboard

The optional read-only dashboard shows baseline progress, recent outputs, GPU memory/utilization, and training loss/eval lines:

```bash
# On the GPU server (binds 127.0.0.1:8777 by default)
bash scripts/start_monitor.sh start

# On your laptop
ssh -N -L 8777:127.0.0.1:8777 <user>@<server>
# Open http://127.0.0.1:8777/
```

Use `bash scripts/start_monitor.sh status` to check the process and `/healthz` endpoint. The dashboard has no authentication and may show clinical text, so prefer the SSH tunnel over exposing the port publicly. See [scripts/README.md](scripts/README.md#监控面板) for direct-LAN access and troubleshooting.

## 6. Reproducibility and honest reporting

- The paper reports **one split and one training seed**. Case-cluster bootstrap CIs describe case-sampling uncertainty, not run-to-run variance.
- The weak-start GRPO ck100 result was selected **after** inspecting test results; the strong-start ck266 checkpoint was selected on validation. These are exploratory results, not an untouched-test confirmation.
- Box calibration is **post-hoc** and fitted on validation only. Raw and calibrated numbers are reported separately; calibration is never fitted on test.
- Saved `args.json` and `logging.jsonl` record the actual runs. Exact overrides, artifact mapping, and the figure/PDF build commands are in [paper/REPRODUCE.md](paper/REPRODUCE.md). The detailed experiment log is in [scripts/README.md](scripts/README.md).
- The public repository does **not** redistribute NOVA images/annotations, model weights, checkpoints, training logs, or per-sample predictions. These are large and/or contain clinical text; the aggregate results are in the paper.

## 7. What is not in this repository

| Excluded | Why |
|---|---|
| `Nova/` (images, CSVs, parquet) | Public benchmark data; evaluation-only license; do not redistribute |
| Model weights (Qwen3.5-9B, LLaVA-1.5-7B) | Large; download from the model publishers |
| `output/`, `result/`, `nova_sft_merged/` | Checkpoints, logs, and predictions; large and may contain clinical text |
| `IOTA_Final_Project.pdf` | Course material, not part of the project |
| Internal planning notes | Not needed for assessment; kept local |

## 8. Citation

If you refer to this course project:

```bibtex
@misc{yan2026grpo,
  author = {Yan, Faxiang},
  title  = {Reinforcement Learning with Verifiable Clinical Rewards for
            Structured Brain MRI Reporting},
  year   = {2026},
  note   = {IOTA 5201 course project, HKUST(GZ)}
}
```

Please also cite the NOVA benchmark if you use its data:

```bibtex
@article{bercea2025nova,
  author  = {Bercea, Cosmin I. and Li, Jun and Raffler, Philipp and others},
  title   = {{NOVA}: A Benchmark for Anomaly Localization and Clinical
             Reasoning in Brain {MRI}},
  journal = {arXiv preprint arXiv:2505.14064},
  year    = {2025}
}
```
