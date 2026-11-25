"""Behavioral checks for dense geometric feedback and reward bounds."""
import unittest
from tests.test_rewards import GOLD_TEXT, render
from src.refinement_rewards import shaped_localization_value


class RefinementRewardTest(unittest.TestCase):
    def test_perfect_and_empty(self):
        self.assertAlmostEqual(shaped_localization_value(GOLD_TEXT, GOLD_TEXT), 1.0)
        self.assertEqual(shaped_localization_value('', GOLD_TEXT), 0.0)

    def test_feedback_below_threshold(self):
        # Both boxes miss IoU 0.5, but a closer box must receive more credit.
        far = render(boxes=[[100, 100, 140, 200]])
        near = render(boxes=[[100, 100, 149, 200]])
        self.assertGreater(shaped_localization_value(near, GOLD_TEXT),
                           shaped_localization_value(far, GOLD_TEXT))
        self.assertGreater(shaped_localization_value(far, GOLD_TEXT), 0.0)

    def test_unmatched_extra_box_loses_credit(self):
        extra = render(boxes=[[100, 100, 200, 200], [300, 300, 400, 400]])
        self.assertLess(shaped_localization_value(extra, GOLD_TEXT), 1.0)

    def test_wrong_hemisphere(self):
        wrong = render(boxes=[[300, 300, 400, 400]])
        self.assertEqual(shaped_localization_value(wrong, GOLD_TEXT), 0.0)

    def test_bounds_under_box_size_changes(self):
        for extent in range(2, 301, 7):
            pred = render(boxes=[[100, 100, 100 + extent, 100 + extent]])
            reward = shaped_localization_value(pred, GOLD_TEXT)
            self.assertGreaterEqual(reward, 0.0)
            self.assertLessEqual(reward, 1.0)


if __name__ == '__main__':
    unittest.main()
