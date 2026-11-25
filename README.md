# Reward Shaping for Informative GRPO Updates in Brain MRI Localization

**Course:** IOTA 5201: RL for Intelligent Decision Making (Fall 2026) — **Path 2: Apply RL to a Problem of Your Choice**  
**Author:** Yan Faxiang · HKUST(GZ)  
**Report:** [paper/main.pdf](paper/main.pdf) — 5 pages, IEEE double-column conference format  
**中文版:** [paper/main_zh.pdf](paper/main_zh.pdf) · [HTML](paper/main_zh.html)  
**Presentation:** [PRESENTATION.md](PRESENTATION.md) — suggested recording outline; add the final video link before submitting  
**Project:** <https://github.com/faxiangcs/final_proj>

---

## 1. Overview

Structured brain MRI reports combine three coupled decisions: **where** the abnormality is,
**what** it looks like, and **what diagnosis** to propose. This project studies the
localization component: the model must generate coordinates that overlap an expert-annotated
abnormality. Supervised fine-tuning (SFT) learns reference coordinate tokens, but the task
metric depends on the geometry of the completed box set, so reinforcement learning (RL) can
optimize the geometric score of the model's own reports.

The algorithm is Group Relative Policy Optimization (GRPO). GRPO estimates advantages by
comparing several completions sampled for the same input. A hard localization threshold can
give boxes of different quality the same score; if the total rewards also tie, every
group-relative task advantage is zero and the group contributes no task-reward update.

**Research question.** Does continuous IoU shaping reduce uninformative reward ties in GRPO,
and does the resulting policy improve localization beyond a strong SFT starting point?

**Design.** Three reward configurations are trained for three seeds (42, 43, 44). All nine
runs share the same SFT-201 initialization, a separate frozen SFT KL reference, the same
data split, prompt, rollout budget, and decoding parameters.

- **Original:** hard localization reward plus diagnosis reward.
- **Shaped + Dx:** continuous shaped localization reward plus diagnosis reward. This changes
  only the localization term relative to Original.
- **Shaped only:** continuous shaped localization reward. This also removes the diagnosis
  component and is therefore a second, confounded contrast.

**Main finding.** Continuous shaping sharply reduces tied-reward groups during training. The
localization-only policy selected on validation has a small positive mean test difference over
SFT, but its confidence interval includes zero. The report therefore claims improved reward
discrimination for GRPO, not a reliable end-task improvement.

## 2. Results at a glance

Test split: 111 images from 36 held-out cases. Scores are raw per-image localization F1@0.5,
averaged over images. SD is across the three GRPO seeds. Intervals use 4,000 paired bootstrap
replicates that resample seeds and whole test cases, and compare each configuration with
SFT-201. Checkpoints and the configuration were selected on validation.

| Policy | Val. mean | Test seed 42 | Test seed 43 | Test seed 44 | Test mean ± SD | Δ vs SFT | 95% CI for Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT-201 (shared start) | 0.3746 | — | — | — | 0.4009 | — | — |
| Original | 0.4498 | 0.4474 | 0.3799 | 0.3739 | 0.4004 ± 0.0409 | −0.0005 | [−0.0792, +0.0761] |
| Shaped + Dx | 0.4630 | 0.4324 | 0.3799 | 0.4144 | 0.4089 ± 0.0267 | +0.0080 | [−0.0602, +0.0775] |
| **Shaped only** | **0.4936** | 0.4279 | 0.4099 | 0.4189 | **0.4189 ± 0.0090** | **+0.0180** | [−0.0566, +0.0908] |

Key numbers:

- **Reward informativeness.** With diagnosis reward held fixed, the mean fraction of training
  groups with zero total-reward standard deviation falls from **14.54%** (Original, seed range
  14.10%–15.04%) to **0.19%** (Shaped + Dx, 0.00%–0.38%). Shaped only averages 0.88%
  (0.56%–1.13%).
- **Localization.** Validation selects Shaped only. Its test F1@0.5 is **0.4189** versus
  **0.4009** for SFT-201: a mean gain of **+0.0180** (1.80 F1 points), with a nominal 95%
  interval of **[−0.0566, +0.0908]**. All three seeds are above SFT.
- **Common endpoint.** At the final step 266, mean test F1 is 0.3914 (Original), 0.4149
  (Shaped + Dx), and 0.4159 (Shaped only). The ranking is descriptive; for Shaped only, two
  seeds exceed SFT and one ties it.
- **Geometric control.** Mean test IoU rises from 0.4028 (SFT-201) to 0.4445 (selected Shaped
  only). Validation-fitted box scaling raises SFT to 0.4069 and leaves Shaped only unchanged
  at 1.00 scale; the remaining difference is +0.0120 ([−0.0613, +0.0850]).
- **Reference points.** A fixed central box scores F1@0.5 = 0.077 on the same test images.
- **Scope.** For selected Shaped only, diagnosis keyword Top-1 falls from 0.1081 to 0.0631
  and description token F1 from 0.2504 to 0.2427. The paper therefore restricts the claim to
  localization and does not infer better overall report quality.

## 3. Method at a glance

- **Observation:** MRI slice `I`, clinical history `H`, and fixed reporting instruction `P`.
- **State:** `(I, H, P, y_<t)`, where `y_<t` is the generated prefix.
- **Action:** the next token. **Transition:** append the token. **Episode:** ends at EOS or a
  512-token generation cap. Rewards are terminal only, with discount `γ = 1`.
- **Policy:** Qwen3.5-9B with a trainable LoRA adapter. The report contains up to three boxes
  in a 480-pixel frame, an image description, and differential and final diagnoses.
- **Rewards:**
  - Hard localization reward, a box-set Jaccard score:
    `R_hard = TP / (TP + FP + FN)`, where `TP` counts greedily matched pairs with IoU ≥ 0.5.
  - Continuous quality term:
    `Q_IoU = 2 * Σ IoU(p, g) / (n_p + n_g)` over positive-overlap greedy pairs; unmatched
    boxes remain in the denominator.
  - Shaped localization reward: `R_shape = 0.5 * R_hard + 0.5 * Q_IoU`.
  - Diagnosis reward:
    `R_Dx = 0.95 * (0.7 * s_final + 0.3 * s_diff) + 0.05 * s_fmt`.
- **Configurations:** Original `R_hard + R_Dx`; Shaped + Dx `R_shape + R_Dx`; Shaped only
  `R_shape`. Active reward weights are all 1.
- **Optimization:** GRPO with `G = 4` completions per input, group-standardized advantage,
  clipping `ε = 0.2`, KL coefficient `β = 0.001`, and a separate frozen SFT reference.
- **Why RL:** the localization score is non-differentiable with respect to the generated text,
  and a deterministic verifier supplies terminal rewards without a learned reward model.

## 4. Experimental setup

- **Data:** 235 cases and 779 images with valid localization and reporting annotations from
  NOVA. Seed 42 assigns 164/35/36 cases and 533/135/111 images to training, validation, and
  test. All images from a case stay in one split.
- **Baseline:** Qwen3.5-9B after 201 SFT optimizer steps (three epochs), which is also the
  common GRPO initialization.
- **Training:** ms-swift with LoRA rank 16 and `α = 32`, frozen vision tower, trainable
  aligner. SFT uses learning rate `5e-5`, maximum length 1024, effective batch size 8 on four
  RTX 3090 GPUs. Each GRPO run uses 266 optimizer steps, learning rate `5e-6`, maximum length
  2048, two RTX 3090 GPUs, per-device batch size 1, gradient accumulation 4, temperature 0.9,
  top-p 0.95, and at most 512 new tokens.
- **Selection:** each run selects among SFT step 0 and GRPO steps 50, 100, 175, and 266 by
  validation F1@0.5, breaking ties with mean IoU and then the earlier step. The configuration
  with the highest mean selected validation F1 across seeds is chosen. Choices are locked
  before new test inference.
- **Uncertainty:** the 4,000-replicate paired bootstrap resamples seeds and whole test cases.
  It describes variation over seeds and cases, not over SFT initializations.

## 5. Repository layout

```text
.
├── README.md                  # this file
├── requirements.txt           # core Python dependencies
├── PRESENTATION.md            # suggested presentation/video outline
├── inference.sh               # swift infer wrapper
├── medical_reward.py          # GRPO reward plugin (hard localization + diagnosis)
├── eval_all_metric.py         # compatibility entry point for src/eval_nova.py
├── src/
│   ├── build_dataset.py       # build case-level train/val/test JSONL files
│   ├── audit_data.py          # structural and leakage audits
│   ├── nova_schema.py         # prompt, labels, parser, box/diagnosis schema
│   ├── eval_nova.py           # localization, caption, and diagnosis metrics
│   ├── infer_qwen_baseline.py # zero-shot Qwen baseline via OpenAI API
│   └── diagnosis_aliases.json # keyword aliases for diagnosis evaluation
├── scripts/
│   ├── README.md              # detailed experiment log and command reference
│   ├── setup_swift_env.sh     # reproduce the ms-swift environment
│   ├── common.sh              # shared defaults
│   ├── run_sft.sh             # SFT
│   ├── run_grpo.sh            # GRPO (reward functions and plugin are overridable)
│   ├── train.sh               # one entry point: sft | grpo
│   ├── run_eval.sh            # evaluation wrapper
│   ├── eval_grpo_checkpoints.sh
│   ├── run_baseline_qwen.sh
│   ├── fit_box_scale.py       # validation-only box calibration
│   ├── calibration_ci.py      # paired seed-and-case bootstrap
│   ├── box_bias_report.py     # box-area bias diagnostics
│   ├── compare_checkpoints.py
│   ├── monitor_dashboard.py   # single-page training/evaluation cockpit
│   ├── start_monitor.sh       # start/stop/status helper
│   ├── dashboard.html         # static snapshot of the dashboard
│   └── make_submission_zip.sh # build the course .zip
├── tests/                     # unit tests (unittest; no GPU required)
└── paper/
    ├── main.tex               # IEEEtran source (English)
    ├── main.pdf               # compiled 5-page report
    ├── main_zh.html           # Chinese translation (HTML)
    ├── main_zh.pdf            # Chinese translation (PDF)
    ├── refs.bib
    ├── REPRODUCE.md           # exact overrides and artifact mapping
    ├── REVISION_NOTES.md      # claim-to-artifact map for this revision
    ├── make_figure.py         # regenerate the evidence-chain figure
    ├── IEEEtran.cls
    ├── IEEEtran.bst
    └── figures/
```

The multi-seed ablation controller and the continuous shaped-reward plugin used for the
paper's main comparison are kept with the local experiment artifacts and are excluded from
this repository by `.gitignore`; `paper/REPRODUCE.md` documents them and the artifact
mapping. The public reward plugin implements the Original configuration.

## 6. Quick start

The reported runs used Python 3.11.18, PyTorch 2.6.0 (CUDA 12.4), Transformers 5.20.2, and
ms-swift 4.6.1. A single GPU works for inference and smoke tests.

```bash
# 1) Environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Alternative: reproduce the conda prefix used for the reported runs
# (set ENV_DIR to choose the install location)
bash scripts/setup_swift_env.sh
```

`transformers` must be 5.x: 4.x does not register the `qwen3_5` architecture. `ms-swift` 4.x
also renamed `--train_type` to `--tuner_type`; the scripts already use the new flag.

### 6.1 Data

NOVA is a public, **evaluation-only** benchmark released under **CC BY-NC-SA 4.0**. This
project fine-tunes on a filtered subset, which departs from the benchmark's intended use; the
paper states this limitation. Obtain NOVA from the official release and follow its license:

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

This produces 164/35/36 cases (533/135/111 images). The JSONL files store absolute image
paths; if the images live elsewhere, re-run the builder with `--image_root` rather than
editing the JSONL by hand.

### 6.2 SFT (SFT-201 baseline)

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

Step 201 of this run is the shared initialization and reference for the paper's nine GRPO
runs. Use `DRY_RUN=1` to print the exact `swift` command without launching training.

### 6.3 GRPO (Original configuration)

```bash
NPROC_PER_NODE=2 \
MODEL="$HOME/models/Qwen3.5-9B" \
ADAPTERS="$PWD/output/qwen_sft/checkpoint-201" \
REF_ADAPTERS="$PWD/output/qwen_sft/checkpoint-201" \
DATA_DIR="$PWD/data/nova_sft" \
OUTPUT_DIR="$PWD/output/qwen_grpo_original_seed42" \
LORA_RANK=16 LORA_ALPHA=32 FREEZE_VIT=true FREEZE_ALIGNER=false \
GRPO_LR=5e-6 GRPO_MAX_STEPS=266 GRPO_BATCH_SIZE=1 GRPO_GRAD_ACCUM=4 \
NUM_GENERATIONS=4 TEMPERATURE=0.9 TOP_P=0.95 BETA=0.001 \
REWARD_FUNCS="localization_reward diagnosis_reward" \
MAX_LENGTH=2048 MAX_COMPLETION_LENGTH=512 \
ENABLE_THINKING=false GRPO_EVAL_STRATEGY=no SEED=42 \
SAVE_STEPS=25 LOGGING_STEPS=1 \
bash scripts/run_grpo.sh
```

`REF_ADAPTERS` makes the KL reference a frozen copy of the SFT adapter, matching the paper.
The Shaped + Dx and Shaped only arms replace `localization_reward` with
`shaped_localization_reward`, registered by the local plugin `src/refinement_rewards.py`
(see [paper/REVISION_NOTES.md](paper/REVISION_NOTES.md)). The LoRA rank and alpha must match
the SFT adapter.

### 6.4 Inference and evaluation

```bash
# Merge the adapter, or point inference.sh at the merged model
bash inference.sh

# Evaluate a prediction file (gold labels are read from the test file here)
python src/eval_nova.py \
  --pred result/qwen_grpo_original_seed42/pred.jsonl \
  --gold data/nova_sft/test.jsonl \
  --coord auto \
  --out result/qwen_grpo_original_seed42/report.json
```

`src/eval_nova.py` reports one-to-one multi-box precision/recall/F1 at IoU 0.3/0.5/0.75,
mean IoU, a custom AP proxy, caption token F1 and BLEU-4, and diagnosis Top-1/Top-5 under
strict, lenient, and keyword matching. The AP proxy is **not** standard COCO mAP.

### 6.5 Figures and PDF

```bash
python paper/make_figure.py
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The Chinese translation is built from HTML, for example:

```bash
cd paper
wkhtmltopdf --enable-local-file-access --page-size A4 \
  --margin-top 17mm --margin-bottom 19mm --margin-left 19mm --margin-right 19mm \
  --footer-center "[page]" main_zh.html main_zh.pdf
```

### 6.6 Tests

```bash
python3 -m unittest discover -s tests -v
```

The suite covers the schema, parser, reward definitions, evaluator, dataset format, and
dashboard helpers. No GPU is needed.

### 6.7 Monitoring dashboard

The read-only dashboard shows the ablation story, live controller/run status, training curves,
GPU state, and saved results:

```bash
# On the GPU server (binds 127.0.0.1:8777 by default)
bash scripts/start_monitor.sh start

# On your laptop
ssh -N -L 8777:127.0.0.1:8777 <user>@<server>
# Open http://127.0.0.1:8777/
```

Use `bash scripts/start_monitor.sh status` to check the process and the `/healthz` endpoint.
The dashboard has no authentication and may show clinical text, so prefer the SSH tunnel over
exposing the port publicly. See [scripts/README.md](scripts/README.md#监控面板) for
direct-LAN access and troubleshooting.

## 7. Reproducibility and honest reporting

- The paper reports **one split**; all three GRPO seeds share **one SFT initialization**. The
  bootstrap resamples seeds and cases, so it captures seed and case variation but not
  variation across SFT runs.
- Configuration and checkpoints were selected on validation and locked before new test
  inference. The same test split had been inspected during earlier development; the intervals
  are exploratory and do not correct for that history.
- All nine runs share one frozen SFT reference, so the experiments do not isolate the effect
  of the reference policy.
- The second contrast (Shaped + Dx to Shaped only) also removes the diagnosis reward and its
  format term. It does not identify whether any localization effect comes from objective
  competition, reward weighting, or another optimization factor.
- Localization F1 uses evaluation preprocessing, while the training rewards use a separate
  per-completion parser and duplicate suppression. These differences are constant across
  configurations but mean reward and metric are not identical.
- Saved `args.json`, `logging.jsonl`, `selection.json`, and aggregate reports record the
  reported runs. Exact overrides, artifact paths, and the claim-to-artifact map are in
  [paper/REPRODUCE.md](paper/REPRODUCE.md) and [paper/REVISION_NOTES.md](paper/REVISION_NOTES.md).
  The detailed experiment log is in [scripts/README.md](scripts/README.md).
- The public repository does **not** redistribute NOVA images/annotations, model weights,
  checkpoints, training logs, per-sample predictions, or the local multi-seed ablation
  controller. These are large and/or contain clinical text; the paper reports aggregate
  results.

## 8. What is not in this repository

| Excluded | Why |
|---|---|
| `Nova/` (images, CSVs, parquet) | Public benchmark data; evaluation-only license; do not redistribute |
| Model weights (Qwen3.5-9B) | Large; download from the model publisher |
| `output/`, `result/`, `experiments/`, `nova_sft_merged/` | Checkpoints, logs, selections, and predictions; large and may contain clinical text |
| `scripts/*refinement*.py`, `src/*refinement*.py` | Local controller, dashboard adapter, and shaped-reward plugin for the multi-seed ablation |
| `IOTA_Final_Project.pdf` | Course material, not part of the project |
| Internal planning notes | Not needed for assessment; kept local |

## 9. Citation

If you refer to this course project:

```bibtex
@misc{yan2026rewardshaping,
  author = {Yan, Faxiang},
  title  = {Reward Shaping for Informative GRPO Updates in Brain MRI Localization},
  year   = {2026},
  note   = {IOTA 5201 course project, HKUST(GZ)},
  url    = {https://github.com/faxiangcs/final_proj}
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
