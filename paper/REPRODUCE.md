# Reproducing the revised course report

> **Repository scope.** The public repository contains code and the report. It does
> **not** redistribute the NOVA images/annotations, model weights, checkpoints,
> training logs, or per-sample predictions. Paths such as `data/`, `output/`,
> `result/`, and `paper/review/` refer to local experiment artifacts; regenerate
> them with the commands below.

Run commands from the repository root. The report follows **Path 2: apply RL**. Its main evidence is the three-reward, three-seed GRPO comparison from SFT-201: reward shaping reduces tied-reward groups, followed by evaluation of localization on cases excluded from training. The earlier single-seed runs are retained below for provenance and are excluded from the paper's main comparison. See `paper/REVISION_NOTES.md` for the claim-to-artifact mapping.

## Environment and data

Training environment inspected on 2026-11-19: Python 3.11.18, torch 2.6.0 (CUDA 12.4 build), transformers 5.20.2, ms-swift 4.6.1, peft 0.20.0, qwen-vl-utils 0.0.14. Figure generation uses matplotlib 3.11.2. SFT and historical GRPO used four RTX 3090 GPUs. The main GRPO ablation used two, with two additional GPUs for evaluation. Zero-shot predictions were served separately with vLLM 0.30; do not install it into the training environment without checking compatibility.

```sh
# Existing environment on this machine
export PATH="/data/yanfaxiang/venvs/ms-swift/bin:$PATH"
# For a new environment, see scripts/setup_swift_env.sh and pin the versions above.
python src/audit_data.py --data_dir data/nova_sft
```

Obtain the public NOVA release from <https://huggingface.co/datasets/c-i-ber/Nova> and follow its CC BY-NC-SA 4.0 license. Main experiments use the saved `data/nova_sft/` files (seed 42, 164/35/36 cases, 533/135/111 images). These files are not redistributed; rebuild them and adjust absolute image paths for a new machine. To rebuild into a **separate** directory:

```sh
python src/build_dataset.py --nova_root Nova --image_root Nova/images \
  --out_dir data/nova_sft_rebuilt --seed 42
python src/audit_data.py --data_dir data/nova_sft_rebuilt
```

The builder and parser evolved during experiments; compare rebuilt files with the saved split before claiming exact reproduction. Do not enable `--integer_boxes`, `--clean_final`, or auxiliary samples for the main SFT recipe; these belong to the data variants. The local audit manifest (not included) recorded SHA-256 hashes of the datasets, predictions, and configurations used in this audit.

## SFT baseline

Replace model/checkpoint paths on a different machine. These explicit overrides matter: training script defaults differ from the reported runs.

```sh
NPROC_PER_NODE=4 MODEL="$HOME/models/Qwen3.5-9B" \
DATA_DIR="$PWD/data/nova_sft" OUTPUT_DIR="$PWD/output/reproduce_qwen_sft" \
LORA_RANK=16 LORA_ALPHA=32 FREEZE_VIT=true FREEZE_ALIGNER=false \
SFT_LR=5e-5 SFT_EPOCHS=3 SFT_BATCH_SIZE=1 SFT_GRAD_ACCUM=2 \
MAX_LENGTH=1024 ENABLE_THINKING=false SEED=42 \
SAVE_STEPS=25 EVAL_STEPS=25 LOGGING_STEPS=1 \
bash scripts/run_sft.sh
```

The original Qwen SFT checkpoints 50 and 201 are under
`output/qwen_sft_full/v0-20261117-192623/`. Use the regenerated checkpoint-201
when reproducing on a different machine. Generate the SFT baseline predictions
and validation report before running the ablation controller.

```sh
model="$HOME/models/Qwen3.5-9B"
sft="$PWD/output/qwen_sft_full/v0-20261117-192623/checkpoint-201"
for split in val test; do
  out="result/qwen_sft_${split}_checkpoint-201"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=0 swift infer \
    --model "$model" --adapters "$sft" \
    --val_dataset "data/nova_sft/$split.jsonl" --infer_backend pt \
    --temperature 0 --max_new_tokens 512 --enable_thinking false \
    --result_path "$out/pred.jsonl"
  python src/eval_nova.py --pred "$out/pred.jsonl" \
    --gold "data/nova_sft/$split.jsonl" --out "$out/report.json"
done
```

Use empty output directories for a fresh inference run. Fit the SFT bounding-box
scale on validation with the same grid used for the GRPO policies.

```sh
python scripts/fit_box_scale.py \
  --pred result/qwen_sft_val_checkpoint-201/pred.jsonl \
  --gold data/nova_sft/val.jsonl --out /tmp/iota_sft_scale.json
```

## Main GRPO experiment

The configuration is `experiments/grpo_refinement_20261119/plan.json`. The
controller `scripts/run_refinement.py` runs the three recipes below for seeds
42, 43, and 44. All runs use SFT-201 both as the initialization and as a
separate frozen KL reference.

| Paper label | Artifact prefix | Reward functions |
|---|---|---|
| Original | `reference` | `localization_reward diagnosis_reward` |
| Shaped + Dx | `dense` | `shaped_localization_reward diagnosis_reward` |
| Shaped only | `loc_only` | `shaped_localization_reward` |

Active reward weights are all 1. GRPO uses 266 optimizer steps with learning
rate `5e-6`, batch size 1 per device, two devices, four accumulation steps,
four completions per input, temperature 0.9, top-p 0.95, `beta=0.001`,
maximum sequence length 2048, and maximum completion length 512.

The following commands are for the original machine. In a fresh workspace on
a different machine, set `base_model`, `sft_adapter`, `train_gpus`, and
`eval_gpus` in the plan, and set `PYTHON` and `SWIFT` in the controller and
the Python path in the launcher. The aggregate code uses 1.20, the scale
selected by the original SFT-201 validation predictions. Substitute the
refitted scale if reproducing with a newly trained baseline.

```sh
bash scripts/start_refinement.sh
python3 scripts/refinement_status.py
```

The same command resumes from saved training checkpoints and completed
inference chunks after interruption. Use a fresh copy for a new experiment.
The existing run uses recorded source hashes and selection files to keep
resumption tied to the recorded configuration.

The controller evaluates SFT step 0 and GRPO steps 50, 100, 175, and 266 on
validation. It chooses by raw localization F1 at IoU 0.5, then mean IoU, then
earlier step. It locks checkpoint choices and the reward configuration before
running test inference for both the selected and final checkpoint per seed.
The calibrated evaluation chooses the scale on validation from 1.00 to 1.50
in 0.05 increments.

| Artifact under `experiments/grpo_refinement_20261119/` | Contents |
|---|---|
| `selection.json`, `configuration_selection.json` | Locked checkpoints and recipe |
| `test_results.json` | 18 evaluation rows with raw and calibrated metrics |
| `aggregate_results.json` | Seed means, sample SDs, and 95% bootstrap intervals |
| `{stage}_seed{seed}/training_done.json` | Saved policy checkpoint and frozen-reference audit |

Uncertainty uses 4,000 paired replicates resampling three GRPO seeds and
36 cases, with random seed 42. `paper/make_figure.py` reads those intervals,
the locked validation candidates in `selection.json`, and the nine formal
`logging.jsonl` files in `output/grpo_refinement_20261119/`. It generates
`figures/fig_evidence_chain.pdf` for the report (analytical reward curve,
measured tied-group fractions, and validation trajectories) and retains the
earlier `figures/fig_refinement.pdf` diagnostic. Validation shading spans the
minimum and maximum of the three seeds; it is not a confidence interval.

When assembling the code portion, include the local plan, `refinement_plugin.py`,
`src/refinement_rewards.py`, `src/grpo_reference_audit.py`,
`scripts/run_refinement.py`, `scripts/start_refinement.sh`, and
`scripts/refinement_status.py`. These experiment sources are present in the
workspace and excluded by the current Git ignore rules. Predictions, checkpoints,
raw logs, and clinical datasets are generated locally.

## Historical comparisons

The fixed-box prior uses `[120,120,360,360]` with the same localization evaluator.
Zero-shot Qwen uses `scripts/run_baseline_qwen.sh`. The historical single-seed
GRPO runs used the original reward sum, an adapter-disabled reference, four
training GPUs, and gradient accumulation 2. To reproduce these runs, use
`scripts/run_grpo.sh` with `ADAPTERS` pointing to SFT-50 or SFT-201, omit
`REF_ADAPTERS`, and use the GRPO training settings above with the historical GPU
configuration. Keep outputs separate from the nine formal runs.

| Historical result | Local directory under `result/` |
|---|---|
| Qwen zero-shot | `qwen35_baseline` |
| SFT-50 | `qwen_sft_ck50` |
| Weak-start GRPO steps 100 and 266 | `qwen_grpo_v1_checkpoint-100` and `qwen_grpo_v1_checkpoint-266` |
| Strong-start GRPO step 266 | `qwen_grpo_ck201_test_checkpoint-266` |

## Figure and PDF

```sh
python paper/make_figure.py
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The revised PDF is five IEEE double-column pages, including references, within the six-page limit. The submitted report is `paper/main.pdf`; the local review backup is not included. Regenerating the figures and PDF from existing artifacts does not run training or inference. The files present before this revision are backed up under `paper/review/before_evidence_revision_20261121_154237/`.
