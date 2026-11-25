# Experiment entry points

Run commands from the repository root in the pinned Python environment.
See [the main README](../README.md) for the complete data → SFT → GRPO → evaluation recipe.

| Entry point | Purpose |
|---|---|
| `setup_swift_env.sh` | Install the pinned environment into `ENV_DIR` |
| `run_sft.sh` | Three-epoch Qwen LoRA SFT, paper defaults |
| `run_grpo.sh` | One GRPO run with an explicit SFT adapter and frozen-reference audit |
| `train.sh sft\|grpo` | Convenience wrapper for the preceding two commands |
| `run_refinement.py` / `start_refinement.sh` | Three rewards × three seeds, validation selection, test inference, statistics |
| `refinement_status.py` | Read persisted progress without model/GPU dependencies |
| `run_eval.sh` | Wrapper for `src/eval_nova.py`; pass both `PRED` and `GOLD` |
| `eval_center_prior.py` | Reproduce the fixed central-box heuristic baseline |
| `fit_box_scale.py` | Fit a box scale using validation predictions only |
| `calibration_ci.py` | Single-policy paired **case** bootstrap for calibration deltas |
| `export_paper_results.py` | Export aggregate metrics and scalar training diagnostics |
| `verify_paper_results.py` | Check consistency of the public evidence files |
| `make_submission_zip.sh` | Archive committed code/report and explicitly include a local video or live exemption |

The main paper's **joint seed-and-case bootstrap** is `paired_seed_case_ci` in
`run_refinement.py`. It resamples whole cases, retaining the image weights within
sampled cases, and resamples RL seeds. It does not resample SFT initializations.

The controller takes `--model`, `--sft-adapter`, `--data-dir`, `--work-dir`,
`--output-dir`, `--train-gpus`, `--eval-gpus`, and `--config`. It uses the active
Python environment; `--swift` can select a different executable. Concurrent
training and evaluation require disjoint GPU lists (the reported run used two
training GPUs and two evaluation GPUs).

```bash
# Preview all nine commands without loading data/models or creating experiment files.
python scripts/run_refinement.py --model /path/to/Qwen3.5-9B \
  --sft-adapter /path/to/the/actual/checkpoint-201 --dry-run

# Resume by repeating the same command, with the same work/output directories.
# Source, resolved configuration and data hashes prevent incompatible resumption.
python scripts/refinement_status.py --work-dir experiments/grpo_ablation
```

`--smoke` trains two updates for the first recipe/seed; `--resume-smoke` resumes
that smoke run to update three. They use separate `smoke_*` output directories.
`--summarize-only` recomputes aggregate statistics from the saved baseline and
policy predictions without running inference. Existing historical experiments
are preserved under their original directories; use fresh default directories
for the portable release.

The old dashboard, API-served zero-shot tools, compatibility evaluator, unused
checkpoint/bias utilities, and old figures have been removed from the public
submission. They do not support the final paper's main comparison. Historical
local data, checkpoints and predictions are retained.
