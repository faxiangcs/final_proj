"""Nova GRPO terminal rewards used by the final paper.

两个 ORM，注册名保持不变：

- ``localization_reward``：一对一多框匹配的 Jaccard@IoU0.5，带 clip/NMS/上限；
- ``diagnosis_reward``：final 0.7 + differential 0.3，外加 0.05 二值格式分。

设计要点：

- 用 ``src/nova_schema.parse_prediction`` 统一解析，不再硬编码旧模板；
- 坐标按整个 completion 的峰值自动换算（``[0,1]`` / ``[0,100]`` / 480 像素 /
  ``[0,1000]``），与 ``src/eval_nova.py`` 口径一致；
- 不再用 ``true_diag in pred_text``；只有归一化全等（1.0）或内容 token
  包含度 ≥0.8（0.8）才计命中；
- ``not found`` 不产生任何奖励；
- 重复框、越界框、空预测都会被 NMS/clip 处理，常量均值框拿不到分。

注意：480 与 1000 的坐标边界在单条 completion 上无法完全消除歧义；如果某条
输出的所有坐标都 ≤480 但实际是 0-1000 归一化，会被当作像素。训练数据本身是
480 像素标签，SFT 模型通常输出像素坐标，风险可接受。
"""

from __future__ import annotations

import os
import sys
from typing import List, Sequence

try:  # ms-swift <= 3.x
    from swift.plugin import ORM, orms
except ImportError:  # ms-swift >= 4.x
    from swift.rewards import ORM, orms

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.eval_nova import diagnosis_match, normalize_diagnosis  # noqa: E402
from src.nova_schema import (  # noqa: E402
    IMG_SIZE,
    MAX_BOXES,
    SECTION_BOX,
    SECTION_CAPTION,
    SECTION_DIAG,
    is_bad_text,
    iou,
    parse_prediction,
)

# 各分量权重；奖励最终落在 [0, 1]。
FINAL_WEIGHT = 0.7
DIFF_WEIGHT = 0.3
FORMAT_WEIGHT = 0.05
LOC_IOU_THRESHOLD = 0.5
DIFF_MATCH_THRESHOLD = 0.8


# --------------------------------------------------------------------------- #
# 解析 / 几何
# --------------------------------------------------------------------------- #


def extract_bboxes(text: str) -> List[List[float]]:
    """兼容旧接口：只取坐标，不做清洗。"""
    try:
        return [list(map(float, box)) for box in parse_prediction(text).boxes]
    except Exception:
        return []


def extract_diagnosis(text: str) -> tuple[str, str]:
    """兼容旧接口：返回 (differential 文本, final)。"""
    try:
        target = parse_prediction(text)
    except Exception:
        return "", ""
    return "; ".join(target.differential), target.final_diagnosis


def calculate_iou(boxA: Sequence[float], boxB: Sequence[float]) -> float:
    return iou(list(boxA), list(boxB))


def calculate_diagnosis_overlap(pred_text: str, true_diag: str) -> float:
    """兼容旧接口：归一化后的 0/0.8/1.0 三档命中分。"""
    return diagnosis_score(pred_text, true_diag)


def _scale_and_clip(
    boxes: Sequence[Sequence[float]],
    img_size: float = IMG_SIZE,
) -> List[List[float]]:
    values = [float(v) for box in boxes for v in list(box)[:4]]
    peak = max(values, default=0.0)
    if peak <= 1.0 + 1e-6:
        scale = float(img_size)
    elif peak <= 100.0 + 1e-6:
        scale = float(img_size) / 100.0
    elif peak <= img_size + 1e-6:
        scale = 1.0
    elif peak <= 1000.0 + 1e-6:
        scale = float(img_size) / 1000.0
    else:
        scale = 1.0

    out: List[List[float]] = []
    for box in boxes:
        if len(box) < 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) * scale for v in list(box)[:4])
        except (TypeError, ValueError):
            continue
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        x1, y1 = min(max(x1, 0.0), img_size), min(max(y1, 0.0), img_size)
        x2, y2 = min(max(x2, 0.0), img_size), min(max(y2, 0.0), img_size)
        if x2 - x1 <= 1.0 or y2 - y1 <= 1.0:
            continue
        out.append([x1, y1, x2, y2])
    return out


def _nms(
    boxes: Sequence[Sequence[float]],
    threshold: float = 0.5,
    max_boxes: int = MAX_BOXES,
) -> List[List[float]]:
    ordered = sorted(
        boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True
    )
    keep: List[List[float]] = []
    for box in ordered:
        if all(iou(list(box), list(other)) <= threshold for other in keep):
            keep.append(list(box))
    return keep[:max_boxes]


def canonicalize_boxes(boxes: Sequence[Sequence[float]]) -> List[List[float]]:
    return _nms(_scale_and_clip(boxes))


def localization_f1(
    pred_boxes: Sequence[Sequence[float]],
    gold_boxes: Sequence[Sequence[float]],
    threshold: float = LOC_IOU_THRESHOLD,
) -> float:
    """Box-set Jaccard = TP / (TP + FP + FN)，一对一贪心匹配；保留历史函数名。"""
    if not gold_boxes:
        return 1.0 if not pred_boxes else 0.0
    if not pred_boxes:
        return 0.0

    pairs = sorted(
        (
            (iou(list(pred), list(gold)), i, j)
            for i, pred in enumerate(pred_boxes)
            for j, gold in enumerate(gold_boxes)
        ),
        reverse=True,
    )
    used_pred: set[int] = set()
    used_gold: set[int] = set()
    true_positive = 0
    for value, i, j in pairs:
        if value < threshold:
            break
        if i in used_pred or j in used_gold:
            continue
        true_positive += 1
        used_pred.add(i)
        used_gold.add(j)

    false_positive = len(pred_boxes) - true_positive
    false_negative = len(gold_boxes) - true_positive
    denominator = true_positive + false_positive + false_negative
    return true_positive / denominator if denominator else 0.0


# --------------------------------------------------------------------------- #
# 诊断
# --------------------------------------------------------------------------- #


def diagnosis_score(pred: str, gold: str) -> float:
    """0 / 0.8 / 1.0；``not found`` 或空串直接 0。"""
    if not pred or not gold or is_bad_text(pred) or is_bad_text(gold):
        return 0.0
    p_norm, g_norm = normalize_diagnosis(pred), normalize_diagnosis(gold)
    if not p_norm or not g_norm:
        return 0.0
    if p_norm == g_norm:
        return 1.0
    return 0.8 if diagnosis_match(pred, gold, lenient=True) else 0.0


def differential_score(pred: Sequence[str], gold: Sequence[str], k: int = 5) -> float:
    """预测前 k 个 differential 与 gold 一对一匹配的命中比例。"""
    golds = [g for g in gold if g and not is_bad_text(g)]
    if not golds:
        return 0.0
    used_gold: set[int] = set()
    hit = 0
    for item in list(pred)[:k]:
        if not item or is_bad_text(item):
            continue
        best_score, best_index = 0.0, -1
        for index, gold in enumerate(golds):
            if index in used_gold:
                continue
            score = diagnosis_score(item, gold)
            if score > best_score:
                best_score, best_index = score, index
        if best_score >= DIFF_MATCH_THRESHOLD and best_index >= 0:
            hit += 1
            used_gold.add(best_index)
    return hit / len(golds)


def format_score(text: str) -> float:
    """三段标题齐全且 caption/final 非空 = 1，否则 0。"""
    if not all(section in (text or "") for section in (SECTION_BOX, SECTION_CAPTION, SECTION_DIAG)):
        return 0.0
    try:
        target = parse_prediction(text)
    except Exception:
        return 0.0
    if not target.final_diagnosis or is_bad_text(target.final_diagnosis):
        return 0.0
    if not target.caption.strip():
        return 0.0
    return 1.0


def diagnosis_reward_value(pred_text: str, gold_text: str) -> float:
    # 没有诊断段标题时不允许拿诊断分，否则只输出一个病名就能刷分。
    if SECTION_DIAG not in (pred_text or ""):
        return 0.0
    try:
        pred = parse_prediction(pred_text)
        gold = parse_prediction(gold_text)
    except Exception:
        return 0.0

    final = diagnosis_score(pred.final_diagnosis, gold.final_diagnosis)
    diff = differential_score(pred.differential, gold.differential)
    accuracy = FINAL_WEIGHT * final + DIFF_WEIGHT * diff
    score = (1.0 - FORMAT_WEIGHT) * accuracy + FORMAT_WEIGHT * format_score(pred_text)
    return min(max(score, 0.0), 1.0)


def localization_reward_value(pred_text: str, gold_text: str) -> float:
    try:
        pred = parse_prediction(pred_text)
        gold = parse_prediction(gold_text)
    except Exception:
        return 0.0
    pred_boxes = canonicalize_boxes(pred.boxes)
    gold_boxes = canonicalize_boxes(gold.boxes)
    return localization_f1(pred_boxes, gold_boxes)


def compute_rewards(pred_text: str, gold_text: str) -> dict[str, float]:
    """离线评估用：返回各分量。"""
    pred = parse_prediction(pred_text)
    gold = parse_prediction(gold_text)
    return {
        "localization": localization_reward_value(pred_text, gold_text),
        "diagnosis": diagnosis_reward_value(pred_text, gold_text),
        "final": diagnosis_score(pred.final_diagnosis, gold.final_diagnosis),
        "differential": differential_score(pred.differential, gold.differential),
        "format": format_score(pred_text),
    }


# --------------------------------------------------------------------------- #
# ORM 适配
# --------------------------------------------------------------------------- #


def align_solutions(completions, kwargs):
    solutions = kwargs.get("solution")
    if not solutions:
        return [""] * len(completions)
    if len(solutions) == len(completions):
        return solutions
    num_gens = max(1, len(completions) // len(solutions))
    aligned = []
    for solution in solutions:
        aligned.extend([solution] * num_gens)
    return aligned[: len(completions)]


def _text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _text(value.get("content", ""))
    if isinstance(value, (list, tuple)):
        return _text(value[0]) if value else ""
    return str(value)


class LocalizationReward(ORM):
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        solutions = align_solutions(completions, kwargs)
        return [
            localization_reward_value(_text(comp), _text(label))
            for comp, label in zip(completions, solutions)
        ]


class DiagnosisReward(ORM):
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        solutions = align_solutions(completions, kwargs)
        return [
            diagnosis_reward_value(_text(comp), _text(label))
            for comp, label in zip(completions, solutions)
        ]


orms["localization_reward"] = LocalizationReward
orms["diagnosis_reward"] = DiagnosisReward
