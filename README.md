# Reward Shaping for Informative GRPO Updates in Brain MRI Localization

**IOTA 5201 · Fall 2026 · Path 2** — Yan Faxiang, HKUST(GZ)

[English paper (5 IEEE pages)](paper/main.pdf) ·
[Reproduction details](paper/REPRODUCE.md)

## Question and evidence

Does continuous IoU shaping reduce uninformative reward ties in GRPO, and does
this improve localization beyond a strong SFT starting policy? The model generates
structured reports from an MRI slice and history. We study the localization boxes,
using token actions, generated-prefix states and a verifiable terminal reward.
At the whole-report level this is also a contextual-bandit problem.

Three reward configurations × three seeds share one Qwen3.5-9B SFT-201 start,
a separate frozen SFT reference, data, decoding and training budget. Checkpoints
and configuration are selected on validation. Test has 111 images from 36 cases.

| Configuration | Reward | Tied groups | Selected validation F1 mean | Test F1 mean ± seed SD | Δ vs SFT; paired 95% CI |
|---|---|---:|---:|---:|---|
| SFT-201 | Supervised baseline | — | 0.3746 | 0.4009 | — |
| Original | Hard localization + diagnosis | 14.54% | 0.4498 | 0.4004 ± 0.0409 | −0.0005 [−0.0792, +0.0761] |
| Shaped + Dx | Shaped localization + diagnosis | 0.19% | 0.4630 | 0.4089 ± 0.0267 | +0.0080 [−0.0602, +0.0775] |
| **Shaped only** | Shaped localization | 0.88% | **0.4936** | **0.4189 ± 0.0090** | +0.0180 [−0.0566, +0.0908] |

F1 is localization F1 at IoU ≥ 0.5, averaged over images. For each GRPO seed,
the checkpoint is selected on validation; both columns evaluate that same
checkpoint. Validation values are the mean of the three selected validation
scores, not the score at a fixed final step. Test values are the mean and
**sample** SD across seeds 42, 43 and 44 (`ddof=1`); this SD is not a standard
error or a confidence interval. SFT is one shared baseline, so no seed SD is
estimated for it. All values are rounded from full precision to four decimals.
Sources: [validation candidates and selections](paper/results/selection.json),
[test results per run](paper/results/test_results.json) and
[seed summaries and paired intervals](paper/results/aggregate_results.json).

The strongest observation is fewer tied-reward groups. Localization has a small
positive point estimate with uncertainty including zero. These results do not
establish generalization or clinical benefit. One split, a shared SFT initialization,
historical test inspection and no compute-matched continued-SFT control limit the
claim. Removing diagnosis also removes its format term and does not isolate a
negative-transfer mechanism. NOVA's original use is evaluation-only; this is an
internal adaptation study, not a result on its official zero-shot leaderboard.

Rewards use the same canonicalization and duplicate suppression:

- `R_hard = TP / (TP + FP + FN)` at IoU ≥ .5: **box-set Jaccard**, despite the
  historical helper name `localization_f1`. Evaluation uses actual F1.
- `Q_IoU = 2 * sum(positive one-to-one matched IoUs) / (n_pred + n_gold)`.
- `R_shape = .5 * R_hard + .5 * Q_IoU`.
- `R_Dx = .95 * (.7 * s_final + .3 * s_diff) + .05 * s_format`.

The continuous term is terminal feedback, not a per-token intermediate reward.

## Repository contents

```text
configs/grpo_ablation.json       # all three rewards, seeds and paper hyperparameters
medical_reward.py               # hard localization and diagnosis rewards
refinement_plugin.py            # all rewards + frozen-reference audit registration
src/                            # schema, dataset builder, audit, evaluator, shaped reward
scripts/run_sft.sh               # common SFT start
scripts/run_grpo.sh              # one audited GRPO run
scripts/run_refinement.py        # complete 9-run controller, selection and bootstrap
scripts/export_paper_results.py  # export numeric evidence, without case text
scripts/verify_paper_results.py  # consistency checks for the public evidence
inference.sh                    # explicit, greedy adapter inference
paper/results/                  # aggregate metrics, validation trajectories, scalar logs
paper/                          # English report, LaTeX source and figures
tests/                          # synthetic CPU fixtures; no NOVA download required
```

The old dashboard, API-served baselines and historical utilities are preserved in
branch [`codex/legacy-before-public-release`](https://github.com/faxiangcs/final_proj/tree/codex/legacy-before-public-release).
Use `submission` for the final paper. Original local experiment artifacts are
preserved; portable runs default to new `experiments/grpo_ablation` and
`output/grpo_ablation` directories.

## 1. Environment

Reported versions: Python 3.11.18, PyTorch 2.6.0 (CUDA 12.4), Transformers 5.20.2,
ms-swift 4.6.1. Activate the environment before running any command:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check

# Alternative conda-prefix installer:
# ENV_DIR=/path/to/env bash scripts/setup_swift_env.sh
```

Use a CUDA-compatible PyTorch build for your machine. All remaining pins are in
`requirements.txt`. Training used four RTX 3090 GPUs for SFT, then two for GRPO
and two separate GPUs for concurrent evaluation. The code preview, unit tests
and aggregate verification do not load models. Figure generation needs matplotlib.

## 2. Data

Obtain [NOVA](https://huggingface.co/datasets/c-i-ber/Nova) under its
CC BY-NC-SA 4.0 license. [NOVA paper](https://arxiv.org/abs/2505.14064).
This project uses the release's `case_metadata.csv`, `captions.csv`,
`bboxes_gold.csv`, and `images/` (480 × 480 PNG). Data and generated JSONL files
are not redistributed.

```bash
python src/build_dataset.py --nova_root Nova --image_root Nova/images \
  --out_dir data/nova_sft --seed 42
python src/audit_data.py --data_dir data/nova_sft
```

Expected: 164/35/36 cases and 533/135/111 images. Splitting is by case. Images
are stored as absolute paths: rebuild on each machine. The main recipe uses the
default builder; do not enable auxiliary samples or label variants.

## 3. SFT baseline

```bash
MODEL=/path/to/Qwen3.5-9B \
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
DATA_DIR="$PWD/data/nova_sft" OUTPUT_DIR="$PWD/output/qwen_sft" \
bash scripts/run_sft.sh
```

Defaults match the report: three epochs, LR `5e-5`, LoRA r16/α32, frozen vision
tower, trainable aligner, per-device batch 1, accumulation 2, sequence cap 1024,
seed 42, thinking disabled. This gives 201 updates on the reported split.

ms-swift adds a version directory. **Use the actual checkpoint path**, for example:

```bash
export MODEL=/path/to/Qwen3.5-9B
export SFT_ADAPTER="$PWD/output/qwen_sft/v0-YYYYMMDD-HHMMSS/checkpoint-201"
```

Replace the example version with the directory produced by training. Do not omit
it or confuse the base model with the adapter. `DRY_RUN=1` prints shell commands
without requiring data/model files.

## 4. Full paper experiment

```bash
# Preview the exact nine training commands; no data/model/GPU access required.
python scripts/run_refinement.py --model "$MODEL" \
  --sft-adapter "$SFT_ADAPTER" --dry-run

# Full run: two training GPUs, two separate evaluation GPUs.
python -u scripts/run_refinement.py --model "$MODEL" \
  --sft-adapter "$SFT_ADAPTER" \
  --data-dir data/nova_sft --config configs/grpo_ablation.json \
  --train-gpus 0,1 --eval-gpus 2,3 \
  --work-dir experiments/grpo_ablation --output-dir output/grpo_ablation
```

The controller handles SFT validation inference, validation-only box-scale fitting,
all nine GRPO runs, checkpoint selection, configuration selection, test inference
for selected/final policies and SFT, and paired seed-and-case CIs. It never runs
new test inference before writing the locked choices. It uses the active Python
and `swift`; override the latter with `--swift /path/to/swift`.

GRPO defaults: 266 updates, LR `5e-6`, two devices, per-device batch 1,
accumulation 4, `G=4`, temperature .9, top-p .95, β=.001, LoRA r16/α32,
sequence cap 2048, completion cap 512, thinking off. Policy/reference adapters
start equal; the callback freezes the reference, audits its hash and checks that
the policy changes. All formal arms use that callback.

Repeat the same command to resume complete optimizer checkpoints and completed
inference chunks. Source, resolved configuration and data hashes prevent mixing
incompatible runs. Use fresh work/output directories for a new recipe. `--smoke`
and `--resume-smoke` use separate two/three-update smoke directories.

```bash
python scripts/refinement_status.py
# Or launch in the background:
# nohup bash scripts/start_refinement.sh --model "$MODEL" \
#   --sft-adapter "$SFT_ADAPTER" > controller.log 2>&1 &
```

For one audited run, rather than the full controller:

```bash
MODEL="$MODEL" ADAPTERS="$SFT_ADAPTER" \
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
REWARD_FUNCS="shaped_localization_reward diagnosis_reward" REWARD_WEIGHTS="1 1" \
OUTPUT_DIR=output/shaped_dx_seed42 SEED=42 bash scripts/run_grpo.sh
```

## 5. Explicit inference and evaluation

Use an actual saved SFT/GRPO adapter; choose a fresh result file:

```bash
MODEL="$MODEL" ADAPTERS="$SFT_ADAPTER" CUDA_VISIBLE_DEVICES=0 \
TEST_FILE=data/nova_sft/test.jsonl RESULT_PATH=result/sft_test/pred.jsonl \
bash inference.sh
python src/eval_nova.py --pred result/sft_test/pred.jsonl \
  --gold data/nova_sft/test.jsonl --coord auto --out result/sft_test/report.json
```

The wrapper fixes greedy decoding, thinking off and a 512-token cap, and passes an
explicit output path. It refuses an existing output to avoid silently appending
old predictions. The evaluator includes localization, caption and lexical diagnosis
checks. Its AP proxy is not COCO mAP. Diagnosis metrics are not clinical accuracy.

The paper's fixed spatial prior can be checked without a model:

```bash
python scripts/eval_center_prior.py --gold data/nova_sft/test.jsonl
```

It predicts `[120,120,360,360]` on every image and scores F1@.5 = .0766 on the
reported test split. SFT-201 is the main learned baseline.

## 6. Verify evidence, regenerate figures and compile

The public numeric summaries include all configurations/seeds, validation choices,
266 scalar tied-group fractions per run, and aggregate test results. No patient
identifiers, report text, image paths or per-case predictions are included.

```bash
python -m unittest discover -s tests -v
python scripts/verify_paper_results.py
python paper/make_figure.py
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The public files suffice for figures and internal numeric consistency checks.
**Recomputing bootstrap intervals requires predictions regenerated with NOVA and
the trained adapters**, not just aggregate means. The joint bootstrap implementation
is `paired_seed_case_ci` in `scripts/run_refinement.py`; `calibration_ci.py` only
implements a single-policy case bootstrap. Use `--summarize-only` on an existing
portable run to recompute its aggregate statistics without inference.

To export a newly completed run's aggregate evidence:

```bash
python scripts/export_paper_results.py --work-dir experiments/grpo_ablation \
  --training-dir output/grpo_ablation --out-dir paper/results
```

The bundled evidence is from the historical reported run, not a new experiment.
Source hashes in `paper/results/provenance.json` document the original files;
portable launcher changes and their validation are described in
[paper/REPRODUCE.md](paper/REPRODUCE.md).

## 7. Presentation and submission

Prepare a 10-minute presentation followed by answers to the three required
questions: Why use RL? Why this method? What is your contribution?
The assignment requires one ZIP containing code, report and an actual video by
December 5, 2026; students selected for the December 1 live talk are exempt from video.

Commit the final files before packaging. Choose one explicit mode:

```bash
bash scripts/make_submission_zip.sh --video /path/to/presentation.mp4
# Only if selected for the live presentation:
# bash scripts/make_submission_zip.sh --live
# Code/report preview (not a complete deliverable):
# bash scripts/make_submission_zip.sh /tmp/iota-code-only.zip --code-only
```

NOVA data, weights, checkpoints, raw logs and predictions remain local and ignored.
The archival branch preserves removed code; it is not needed in the submission ZIP.

## Citation

```bibtex
@misc{yan2026rewardshaping,
  author = {Yan, Faxiang},
  title = {Reward Shaping for Informative GRPO Updates in Brain MRI Localization},
  year = {2026},
  note = {IOTA 5201 course project, HKUST(GZ)},
  url = {https://github.com/faxiangcs/final_proj}
}
```

Cite and follow the license of the [NOVA benchmark](https://arxiv.org/abs/2505.14064)
when using its data.
