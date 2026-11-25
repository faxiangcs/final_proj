"""Continuous localization shaping for a separately registered GRPO experiment.

Keep legacy rewards unchanged. Share legacy coordinate/NMS preprocessing to
avoid conflating reward shaping with a preprocessing change.
"""
from medical_reward import (
    ORM, orms, align_solutions, _text, canonicalize_boxes, localization_f1,
)
from src.eval_nova import greedy_match
from src.nova_schema import parse_prediction


def shaped_localization_value(pred_text: str, gold_text: str) -> float:
    pred = canonicalize_boxes(parse_prediction(pred_text).boxes)
    gold = canonicalize_boxes(parse_prediction(gold_text).boxes)
    if not gold:
        return float(not pred)
    if not pred:
        return 0.0
    pairs = greedy_match(pred, gold)
    # Every prediction and every target enters the denominator, including
    # unmatched boxes. The quality term therefore cannot improve by appending
    # arbitrary distinct boxes. Legacy NMS still suppresses duplicates.
    quality = 2.0 * sum(overlap for _, _, overlap in pairs) / (len(pred) + len(gold))
    discrete = localization_f1(pred, gold)
    return 0.5 * discrete + 0.5 * quality


class ShapedLocalizationReward(ORM):
    def __call__(self, completions, **kwargs):
        solutions = align_solutions(completions, kwargs)
        if len(solutions) != len(completions):
            raise ValueError('Completion/solution alignment failed')
        return [shaped_localization_value(_text(p), _text(g))
                for p, g in zip(completions, solutions)]


orms['shaped_localization_reward'] = ShapedLocalizationReward
