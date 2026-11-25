"""ms-swift plugin: legacy + shaped rewards and a frozen-reference audit."""
import sys
from pathlib import Path

root = str(Path(__file__).resolve().parent)
if root not in sys.path:
    sys.path.insert(0, root)

import medical_reward  # registers unchanged legacy rewards
import src.refinement_rewards  # registers shaped_localization_reward
import src.grpo_reference_audit  # registers frozen_sft_reference_audit
