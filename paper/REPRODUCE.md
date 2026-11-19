# Reproducing the revised course report

> **Repository scope.** The public repository contains code and the report. It does
> **not** redistribute the NOVA images/annotations, model weights, checkpoints,
> training logs, or per-sample predictions. Paths such as `data/`, `output/`,
> `result/`, and `paper/review/` refer to local experiment artifacts; regenerate
> them with the commands below.

Run commands from the repository root. The report is **Path 2: apply RL**. No training was rerun for the paper audit. Results come from saved predictions and run configurations.

## Environment and data

Training environment inspected on 2026-11-19: Python 3.11.18, torch 2.6.0 (CUDA 12.4 build), transformers 5.20.2, ms-swift 4.6.1, peft 0.20.0, qwen-vl-utils 0.0.14. Figure generation uses matplotlib 3.11.2. Training used four RTX 3090 GPUs. Zero-shot predictions were served separately with vLLM 0.30; do not install it into the training environment without checking compatibility.

```sh
# Existing environment on this machine
export PATH="$HOME/venvs/ms-swift/bin:$PATH"
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

## SFT and GRPO

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

LLaVA uses `MODEL="$HOME/models/llava-1.5-7b-hf"`, `MAX_LENGTH=2048`, and a separate output directory, with the same SFT settings. The original Qwen SFT directory is `output/qwen_sft_full/v0-20261117-192623/`; early and final checkpoints are 50 and 201. Original LLaVA SFT is `output/nova_sft_full_v2/v0-20261117-180551/checkpoint-50`.

```sh
# Repeat with checkpoint-201 and a separate OUTPUT_DIR for the stronger start.
NPROC_PER_NODE=4 MODEL="$HOME/models/Qwen3.5-9B" \
ADAPTERS="$PWD/output/qwen_sft_full/v0-20261117-192623/checkpoint-50" \
DATA_DIR="$PWD/data/nova_sft" OUTPUT_DIR="$PWD/output/reproduce_qwen_grpo_weak" \
LORA_RANK=16 LORA_ALPHA=32 FREEZE_VIT=true FREEZE_ALIGNER=false \
GRPO_LR=5e-6 GRPO_EPOCHS=1 GRPO_BATCH_SIZE=1 GRPO_GRAD_ACCUM=2 \
NUM_GENERATIONS=4 TEMPERATURE=0.9 TOP_P=0.95 BETA=0.001 \
REWARD_WEIGHTS="1 1" MAX_LENGTH=2048 MAX_COMPLETION_LENGTH=512 \
ENABLE_THINKING=false GRPO_EVAL_STRATEGY=no SEED=42 \
SAVE_STEPS=25 LOGGING_STEPS=1 bash scripts/run_grpo.sh
```

Actual GRPO reward is `localization_reward + diagnosis_reward` (range 0–2), not their average. The localization implementation is Jaccard over matched box counts, despite its Python name `localization_f1`. The saved runs use the adapter-disabled KL reference, not a frozen SFT reference adapter. These details describe the experiments; changing the code now would require new training to evaluate a different method.

## Prediction and evaluation

```sh
CUDA_VISIBLE_DEVICES=0 swift infer \
  --model "$HOME/models/Qwen3.5-9B" \
  --adapters output/qwen_grpo_v1/v0-20261117-221829/checkpoint-100 \
  --val_dataset data/nova_sft/test.jsonl --infer_backend pt \
  --temperature 0 --max_new_tokens 512 --enable_thinking false \
  --result_path /tmp/iota_reproduced_predictions.jsonl
python src/eval_nova.py --pred /tmp/iota_reproduced_predictions.jsonl \
  --gold data/nova_sft/test.jsonl --out /tmp/iota_reproduced_report.json
```

Repeat inference with the checkpoint paths in the mapping below. Main zero-shot runs use the same user prompt and 512-token greedy decoding, without the additional system-format hint. See `scripts/run_baseline_qwen.sh` and the saved baseline prediction files; the format-guided image/text comparison is a separate experiment, not the main-table zero-shot row. Predictions alone do not archive the full vLLM server configuration, which limits exact reproduction of those baselines.

| Main-table row | Local artifact directory under `result/` (not included) |
|---|---|
| LLaVA zero-shot | `llava_base` |
| LLaVA SFT-50 | `nova_sft_ck50` |
| LLaVA GRPO ck50 | `llava_grpo_v1_checkpoint-50` |
| Qwen zero-shot | `qwen35_baseline` |
| Qwen SFT-50 | `qwen_sft_ck50` |
| Qwen weak-start GRPO ck100 | `qwen_grpo_v1_checkpoint-100` |
| Qwen SFT-201 | `qwen_sft_test_checkpoint-201` |
| Qwen strong-start GRPO ck266 | `qwen_grpo_ck201_test_checkpoint-266` |

Each directory contains `pred.jsonl` and `report.json`. The final weak-start result is in `qwen_grpo_v1_checkpoint-266`. The center prior uses one fixed box `[120,120,360,360]` per image and the same evaluator. `map50_95` in JSON is the custom per-image AP proxy described in the revised paper, **not standard COCO mAP**.

```sh
# Recompute paired CIs from regenerated predictions (CPU only, standard library).
# The local audit script paper/review/verify_results.py is not included.
# Fit scale using validation only; then apply the selected value to test.
python scripts/fit_box_scale.py \
  --pred result/qwen_sft_val_checkpoint-50/pred.jsonl \
  --gold data/nova_sft/val.jsonl --out /tmp/iota_box_scale.json
python src/eval_nova.py --pred result/qwen_sft_ck50/pred.jsonl \
  --gold data/nova_sft/test.jsonl --box_scale 1.25 \
  --out /tmp/iota_calibrated_report.json
python scripts/calibration_ci.py --pred result/qwen_sft_ck50/pred.jsonl \
  --gold data/nova_sft/test.jsonl --scale 1.25
```

CIs resample 36 whole cases, 4000 times, with seed 42. They do not correct test-informed checkpoint/metric selection or estimate training-seed variance.

## Figure and PDF

```sh
python paper/make_figure.py
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The revised PDF must remain within six IEEE double-column pages, including references. The submitted report is `paper/main.pdf`; the local review backup is not included.
