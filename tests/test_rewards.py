"""Unit tests for the redesigned GRPO rewards."""

from __future__ import annotations

import os
import sys
import types
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_swift = types.ModuleType("swift")
_plugin = types.ModuleType("swift.plugin")


class _ORM:
    pass


_plugin.ORM = _ORM
_plugin.orms = {}
sys.modules.setdefault("swift", _swift)
sys.modules.setdefault("swift.plugin", _plugin)

import medical_reward as rewards  # noqa: E402
from src.nova_schema import NovaTarget, render_label  # noqa: E402


GOLD = NovaTarget(
    boxes=[[100, 100, 200, 200]],
    caption="Axial T2 hyperintense left frontal lesion.",
    final_diagnosis="Glioma",
    differential=["Metastasis", "Abscess", "Infarction"],
)
GOLD_TEXT = render_label(GOLD)


def render(**overrides) -> str:
    target = NovaTarget(**{**GOLD.to_dict(), **overrides})
    return render_label(target)


class LocalizationRewardTest(unittest.TestCase):
    def test_perfect_is_one(self):
        self.assertAlmostEqual(
            rewards.localization_reward_value(GOLD_TEXT, GOLD_TEXT), 1.0
        )

    def test_wrong_box_is_zero(self):
        wrong = render(boxes=[[10, 10, 40, 40]])
        self.assertEqual(rewards.localization_reward_value(wrong, GOLD_TEXT), 0.0)

    def test_mean_box_gets_nothing(self):
        mean_box = render(boxes=[[0, 0, 480, 480]])
        self.assertLess(rewards.localization_reward_value(mean_box, GOLD_TEXT), 0.1)

    def test_duplicate_boxes_are_deduped(self):
        dup = render(boxes=GOLD.boxes * 5)
        self.assertAlmostEqual(
            rewards.localization_reward_value(dup, GOLD_TEXT), 1.0
        )

    def test_norm1000_boxes_are_converted(self):
        target = NovaTarget(
            boxes=[[300, 100, 400, 200]],
            caption=GOLD.caption,
            final_diagnosis=GOLD.final_diagnosis,
            differential=list(GOLD.differential),
        )
        gold_text = render_label(target)
        scaled = [[round(v * 1000 / 480, 2) for v in box] for box in target.boxes]
        pred = render_label(
            NovaTarget(
                boxes=scaled,
                caption=GOLD.caption,
                final_diagnosis=GOLD.final_diagnosis,
                differential=list(GOLD.differential),
            )
        )
        self.assertAlmostEqual(
            rewards.localization_reward_value(pred, gold_text), 1.0
        )

    def test_one_of_two_boxes(self):
        pred = render(boxes=[[100, 100, 200, 200], [300, 300, 400, 400]])
        # TP=1, FP=1, FN=0 -> 1/2
        self.assertAlmostEqual(
            rewards.localization_reward_value(pred, GOLD_TEXT), 0.5
        )

    def test_empty_prediction(self):
        self.assertEqual(rewards.localization_reward_value("", GOLD_TEXT), 0.0)


class DiagnosisRewardTest(unittest.TestCase):
    def test_perfect_is_one(self):
        self.assertAlmostEqual(
            rewards.diagnosis_reward_value(GOLD_TEXT, GOLD_TEXT), 1.0
        )

    def test_wrong_everything_only_format(self):
        pred = render(
            final_diagnosis="Stroke",
            differential=["Hemorrhage"],
        )
        score = rewards.diagnosis_reward_value(pred, GOLD_TEXT)
        self.assertAlmostEqual(score, rewards.FORMAT_WEIGHT, places=6)

    def test_wrong_final_keeps_differential_credit(self):
        pred = render(final_diagnosis="Stroke")
        expected = (1 - rewards.FORMAT_WEIGHT) * rewards.DIFF_WEIGHT + rewards.FORMAT_WEIGHT
        self.assertAlmostEqual(
            rewards.diagnosis_reward_value(pred, GOLD_TEXT), expected, places=6
        )

    def test_not_found_is_zero(self):
        text = (
            "【异常定位】：[100, 100, 200, 200]\n"
            "【图像描述】：not found\n"
            "【诊断推理】：考虑鉴别诊断包括 not found。结合特征，最终诊断为：not found。"
        )
        self.assertEqual(rewards.diagnosis_reward_value(text, GOLD_TEXT), 0.0)

    def test_contains_is_not_enough(self):
        """旧实现里 gold 是 pred 的子串就给满分，现在不允许。"""
        self.assertEqual(
            rewards.diagnosis_score("This is definitely not a glioma", "Glioma"),
            0.0,
        )
        self.assertEqual(
            rewards.diagnosis_score("The lesion is probably a metastasis", "Glioma"),
            0.0,
        )

    def test_partial_differential(self):
        self.assertAlmostEqual(
            rewards.differential_score(["Metastasis"], GOLD.differential), 1 / 3, places=6
        )

    def test_bare_diagnosis_without_heading_gets_nothing(self):
        self.assertEqual(rewards.diagnosis_reward_value("Glioma", GOLD_TEXT), 0.0)

    def test_missing_heading_loses_format_bonus(self):
        text = GOLD_TEXT.replace("【图像描述】：", "")
        self.assertEqual(rewards.format_score(text), 0.0)


class ORMTest(unittest.TestCase):
    def test_message_dict_inputs(self):
        orm = rewards.LocalizationReward()
        values = orm(
            [{"content": GOLD_TEXT}],
            solution=[{"content": GOLD_TEXT}],
        )
        self.assertEqual(len(values), 1)
        self.assertAlmostEqual(values[0], 1.0)

    def test_solution_broadcast(self):
        orm = rewards.DiagnosisReward()
        values = orm([GOLD_TEXT, GOLD_TEXT], solution=[GOLD_TEXT])
        self.assertEqual(len(values), 2)
        for value in values:
            self.assertAlmostEqual(value, 1.0)


class CompatibilityTest(unittest.TestCase):
    def test_extract_diagnosis(self):
        diff, final = rewards.extract_diagnosis(GOLD_TEXT)
        self.assertEqual(final, "Glioma")
        self.assertIn("Metastasis", diff)

    def test_compute_rewards_keys(self):
        result = rewards.compute_rewards(GOLD_TEXT, GOLD_TEXT)
        for key in ("localization", "diagnosis", "final", "differential", "format"):
            self.assertIn(key, result)
        self.assertAlmostEqual(result["localization"], 1.0)
        self.assertAlmostEqual(result["diagnosis"], 1.0)


if __name__ == "__main__":
    unittest.main()
