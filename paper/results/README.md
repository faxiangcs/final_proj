# Public evidence for the final paper

These are numeric exports of the reported experiment, not synthetic or newly
rerun results. `training_signal.json` retains one scalar tied-group fraction per
training step for each of nine runs; `selection.json` retains candidate scores
and chosen steps. `test_results.json` contains 18 aggregate evaluation rows;
`aggregate_results.json` contains seed means, SDs and the reported paired CIs.
`baseline.json` is aggregate SFT evaluation. `provenance.json` records source
hashes without copying clinical data or checkpoint paths.

Run `python scripts/verify_paper_results.py` and `python paper/make_figure.py`
from the repository root. These need no private dataset/model artifacts. The
verifier checks internal consistency; recomputing the paired case/seed intervals
requires predictions from a locally reproduced run.
