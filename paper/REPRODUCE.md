# Reproducing the final report

Use the [root README](../README.md) for executable setup, data, SFT, GRPO and
inference commands. It is the single command reference. The final report follows
Path 2 and consists of five IEEE double-column pages, including references.

## Public method and evidence

| Paper component | Public source |
|---|---|
| Hard localization and diagnosis | `medical_reward.py` |
| Continuous IoU shaping | `src/refinement_rewards.py` |
| Frozen SFT reference and hash checks | `src/grpo_reference_audit.py`, `refinement_plugin.py` |
| Three reward arms × seeds 42–44 | `configs/grpo_ablation.json` |
| Training, validation selection, inference, calibration | `scripts/run_refinement.py` |
| Paired seed-and-case bootstrap | `paired_seed_case_ci` in that controller |
| Validation candidates and locked choices | `paper/results/selection.json`, `configuration_selection.json` |
| Test mean/SD/CI and 18 seed/checkpoint summaries | `paper/results/aggregate_results.json`, `test_results.json` |
| SFT localization, caption and diagnosis metrics | `paper/results/baseline.json` |
| Figure 1b: 266 scalar diagnostics per run | `paper/results/training_signal.json` |
| Original source/artifact SHA-256 hashes | `paper/results/provenance.json` |
| Figure 1a analytical mechanism and Fig. 1b/c measurements | `paper/make_figure.py` |

No local `experiments/`, `output/`, `result/`, data or model download is needed to
verify the public summaries or redraw Figure 1. To recompute the CIs themselves,
regenerate the predictions: aggregate means alone do not retain paired case data.
The public export deliberately omits case IDs, report text and image/model paths.

The main bootstrap has 4,000 replicates, random seed 42, and resamples RL seeds
and whole cases. Images retain their weight within sampled cases. It does not
resample SFT initializations. The helper `scripts/calibration_ci.py` is a separate,
single-policy case bootstrap, not the joint procedure.

## Execution and selection order

1. Build the 235-case dataset with seed 42. The current builder was independently
   checked against the saved split: 164/35/36 cases, 533/135/111 images, same rows.
2. Train Qwen3.5-9B for three SFT epochs. With the reported four-GPU batch recipe,
   step 201 is the common policy initialization and separate frozen reference.
3. Pass the actual `v*/checkpoint-201` path and base model to the controller.
   SFT validation inference and baseline box-scale fitting run first.
4. Run all three configurations with seeds 42/43/44 to 266 updates. Evaluate step 0
   and GRPO steps 50/100/175/266 on validation. Select by F1@.5, then mean IoU,
   then earlier step. Select the configuration by mean selected validation F1.
5. Write all checkpoint selections and configuration selection before new test
   inference. Then evaluate the selected and final checkpoint for every seed,
   including all three configurations. Compute SFT and policy calibration using
   validation predictions only, on the 1.00–1.50 grid in .05 increments.
6. Aggregate seed means/sample SDs and paired seed-and-case intervals. The
   historical SFT validation-selected scale was 1.20; fresh runs refit it rather
   than silently hardcoding the historical value.

The two training GPUs and concurrent evaluation GPU lists must be disjoint.
A full rerun needs the licensed dataset, downloaded model, storage and GPUs.
The released controller uses the active interpreter, configurable paths and a
new default work/output directory. It never rewrites the original historical run.

## Preservation and release changes

Branch [`codex/legacy-before-public-release`](https://github.com/faxiangcs/final_proj/tree/codex/legacy-before-public-release)
preserves the tracked code before cleanup plus the original local reward,
reference audit, controller, plan and tests. Its base is commit
`659fa8dc6f60d510c607fc29c4739508086050db`. The historical source hashes in the
public provenance file correspond to that experiment, not to the revised portable
controller. Old dashboard/API baseline/checkpoint utilities and unused figures
are archived there rather than included in the final submission branch.

Release changes parameterize paths, move configuration into `configs/`, automate
baseline inference, refit baseline calibration on validation, make tests independent
of local data, and expose aggregate plotting inputs. Reward formulas, parser,
evaluator and bootstrap sampling procedure are preserved. Historical training
outputs and paper results are unchanged. No new GPU training was run for release
verification; tests, command previews, local saved-prediction comparison, figure
regeneration and clean paper compilation provide the validation.

The results remain exploratory: one split, a shared SFT initialization, historical
test access and no compute-matched continued-SFT control. Completing the source
release does not strengthen those statistical claims.
