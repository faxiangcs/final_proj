"""Nova evaluation: canonical parsing, coordinate conversion, honest metrics.

The evaluator:

- matches predicted and gold boxes one-to-one with greedy IoU matching, instead
  of scoring only the first predicted box;
- reports strict and lenient diagnosis Top-1/Top-5, keyword-alias matching, and
  caption token F1 / BLEU-4; metric names match their definitions;
- converts predicted boxes from ``auto / pixel / norm1 / norm100 / norm1000``
  into the 480-pixel space before computing IoU, so LLaVA-NeXT ``[0,1]`` and
  Qwen ``[0,1000]`` coordinates do not collapse to zero IoU.

Usage::

    python3 src/eval_nova.py --pred result/xxx/infer_result/answer.jsonl
    python3 src/eval_nova.py --pred pred.jsonl --gold data/nova_sft/test.jsonl \\
        --coord auto --out eval_report.json

预测文件需包含以下任一字段：``response`` / ``predict`` / ``prediction`` /
``messages``（最后一条 assistant）。gold 缺省时从同一文件的 ``labels`` 字段读取。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

try:  # `python src/eval_nova.py` 与 `python -m src.eval_nova` 两种入口
    from src.nova_schema import (
        IMG_SIZE,
        iou,
        parse_label,
        parse_prediction,
    )
except ImportError:  # pragma: no cover - script mode
    from nova_schema import (  # type: ignore
        IMG_SIZE,
        iou,
        parse_label,
        parse_prediction,
    )


# --------------------------------------------------------------------------- #
# 坐标
# --------------------------------------------------------------------------- #

COORD_MODES = ("auto", "pixel", "norm1", "norm100", "norm1000")


def coord_scale(mode: str, img_size: float = IMG_SIZE) -> float:
    if mode == "pixel":
        return 1.0
    if mode == "norm1":
        return float(img_size)
    if mode == "norm100":
        return float(img_size) / 100.0
    if mode == "norm1000":
        return float(img_size) / 1000.0
    raise ValueError(f"unknown coord mode: {mode}")


def detect_coord_mode(
    box_lists: Iterable[Sequence[Sequence[float]]],
    img_size: float = IMG_SIZE,
) -> str:
    """Dataset-level heuristic.

    只看整个文件里所有框的最大坐标，避免单样本的歧义被放大：
    <=1 -> [0,1]；<=100 -> [0,100]；<=480 -> 像素；<=1000 -> [0,1000]。
    480 与 1000 的边界是已知歧义点，需要时用 ``--coord`` 显式指定。
    """
    values: list[float] = []
    for boxes in box_lists:
        for box in boxes:
            values.extend(float(v) for v in box[:4])
    if not values:
        return "pixel"
    peak = max(values)
    if peak <= 1.0 + 1e-6:
        return "norm1"
    if peak <= 100.0 + 1e-6:
        return "norm100"
    if peak <= img_size + 1e-6:
        return "pixel"
    if peak <= 1000.0 + 1e-6:
        return "norm1000"
    return "pixel"  # 超过 1000：当作原图像素，配合 --source_image_size


def convert_boxes(
    boxes: Sequence[Sequence[float]],
    mode: str,
    img_size: float = IMG_SIZE,
    source_image_size: float | None = None,
) -> tuple[list[list[float]], dict[str, int]]:
    """把一组框转换到 [0, img_size] 像素空间，并返回清洗统计。"""
    stats = {"swapped": 0, "clipped": 0, "dropped": 0, "in_range": 0}
    if mode == "auto":
        mode = detect_coord_mode([boxes], img_size=img_size)

    scale = coord_scale(mode, img_size)
    if mode == "pixel" and source_image_size and abs(source_image_size - img_size) > 1e-6:
        scale = float(img_size) / float(source_image_size)

    out: list[list[float]] = []
    for box in boxes:
        if len(box) < 4:
            stats["dropped"] += 1
            continue
        try:
            x1, y1, x2, y2 = (float(v) * scale for v in box[:4])
        except (TypeError, ValueError):
            stats["dropped"] += 1
            continue
        if x2 < x1:
            x1, x2 = x2, x1
            stats["swapped"] += 1
        if y2 < y1:
            y1, y2 = y2, y1
            stats["swapped"] += 1

        cx1, cy1 = min(max(x1, 0.0), img_size), min(max(y1, 0.0), img_size)
        cx2, cy2 = min(max(x2, 0.0), img_size), min(max(y2, 0.0), img_size)
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            stats["clipped"] += 1
        if cx2 - cx1 <= 1.0 or cy2 - cy1 <= 1.0:
            stats["dropped"] += 1
            continue
        out.append([cx1, cy1, cx2, cy2])
        stats["in_range"] += 1
    return out, stats


# --------------------------------------------------------------------------- #
# 多框一对一匹配
# --------------------------------------------------------------------------- #


def scale_boxes(
    boxes: Sequence[Sequence[float]],
    scale: float,
    img_size: float = IMG_SIZE,
) -> list[list[float]]:
    """绕框中心缩放，再 clip 到图像范围；``scale=1`` 时原样返回。"""
    if abs(scale - 1.0) < 1e-9:
        return [list(map(float, box[:4])) for box in boxes]
    out: list[list[float]] = []
    for box in boxes:
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        half_w, half_h = (x2 - x1) * scale / 2.0, (y2 - y1) * scale / 2.0
        out.append([
            min(max(cx - half_w, 0.0), img_size),
            min(max(cy - half_h, 0.0), img_size),
            min(max(cx + half_w, 0.0), img_size),
            min(max(cy + half_h, 0.0), img_size),
        ])
    return out


def greedy_match(
    pred: Sequence[Sequence[float]],
    gold: Sequence[Sequence[float]],
    min_iou: float = 0.0,
) -> list[tuple[int, int, float]]:
    """按 IoU 从高到低做一对一贪心匹配，返回 (pred_idx, gold_idx, iou)。"""
    pairs: list[tuple[float, int, int]] = []
    for pi, pbox in enumerate(pred):
        for gi, gbox in enumerate(gold):
            value = iou(list(pbox), list(gbox))
            if value > min_iou:
                pairs.append((value, pi, gi))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_pred: set[int] = set()
    used_gold: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for value, pi, gi in pairs:
        if pi in used_pred or gi in used_gold:
            continue
        used_pred.add(pi)
        used_gold.add(gi)
        matched.append((pi, gi, value))
    matched.sort(key=lambda item: item[0])
    return matched


def localization_metrics(
    pred_boxes: Sequence[Sequence[float]],
    gold_boxes: Sequence[Sequence[float]],
    thresholds: Sequence[float] = (0.3, 0.5, 0.75),
) -> dict[str, float]:
    """单样本定位指标。

    precision/recall/F1 均基于一对一匹配；``map50_95`` 是以预测顺序为置信度的
    rank-based AP（COCO 式阈值平均）。没有 gold 的样本不应调用本函数。
    """
    n_pred, n_gold = len(pred_boxes), len(gold_boxes)
    matched = greedy_match(pred_boxes, gold_boxes)

    result: dict[str, float] = {
        "n_pred": float(n_pred),
        "n_gold": float(n_gold),
        "mean_iou": (sum(m[2] for m in matched) / len(matched)) if matched else 0.0,
    }
    for t in thresholds:
        hit = sum(1 for _, _, value in matched if value >= t)
        precision = hit / n_pred if n_pred else 0.0
        recall = hit / n_gold if n_gold else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        result[f"precision@{t:g}"] = precision
        result[f"recall@{t:g}"] = recall
        result[f"f1@{t:g}"] = f1
        result[f"hit@{t:g}"] = 1.0 if hit else 0.0

    coco_thresholds = [0.5 + 0.05 * i for i in range(10)]
    aps: list[float] = []
    for t in coco_thresholds:
        tp = 0
        ap = 0.0
        seen_gold: set[int] = set()
        for rank, pbox in enumerate(pred_boxes, start=1):
            best_iou, best_gold = 0.0, -1
            for gi, gbox in enumerate(gold_boxes):
                if gi in seen_gold:
                    continue
                value = iou(list(pbox), list(gbox))
                if value > best_iou:
                    best_iou, best_gold = value, gi
            if best_iou >= t and best_gold >= 0:
                seen_gold.add(best_gold)
                tp += 1
                ap += tp / rank
        aps.append(ap / n_gold if n_gold else 0.0)
    result["map50_95"] = sum(aps) / len(aps) if aps else 0.0
    return result


# --------------------------------------------------------------------------- #
# 诊断匹配
# --------------------------------------------------------------------------- #

_SYNONYMS = {
    "tumour": "tumor",
    "tumours": "tumor",
    "tumors": "tumor",
    "haemorrhage": "hemorrhage",
    "haemorrhages": "hemorrhage",
    "hemorrhages": "hemorrhage",
    "oedema": "edema",
    "metastases": "metastasis",
    "mets": "metastasis",
    "meningiomas": "meningioma",
    "gliomas": "glioma",
    "lymphomas": "lymphoma",
    "infarcts": "infarct",
    "infarctions": "infarction",
}

_STOPWORDS = {
    "a", "an", "the", "of", "with", "and", "in", "on", "at", "to", "for",
    "type", "pattern", "nos", "disease", "syndrome", "lesion", "lesions",
    "multiple", "bilateral", "left", "right", "possible", "likely",
}

_TOKEN_RE = re.compile(r"[0-9a-zA-Z]+|[一-鿿]")


def _singular(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if (
        len(token) > 4
        and token.endswith("s")
        and not token.endswith("ss")
        and not token.endswith("sis")
    ):
        return token[:-1]
    return token


def _canon_token(raw: str) -> str:
    token = _SYNONYMS.get(raw, raw)
    token = _singular(token)
    return _SYNONYMS.get(token, token)


def normalize_diagnosis(text: str) -> str:
    """小写 + 去标点 + 单复数/拼写同义归一，用于 strict 全等比较。"""
    return " ".join(_canon_token(raw) for raw in _TOKEN_RE.findall((text or "").lower()))


def diagnosis_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in _TOKEN_RE.findall((text or "").lower()):
        token = _canon_token(raw)
        if token in _STOPWORDS:
            continue
        tokens.add(token)
    # 中英混排（如 ``毛细胞型星形细胞瘤（Pilocytic Astrocytoma）``）只保留英文
    # token 再和英文 gold 比较，避免中文把 Jaccard 稀释成 0；纯中文输出没有英文
    # token，仍按原 token 参与 strict/lenient，不会凭空命中。
    latin = {token for token in tokens if token.isascii()}
    return latin or tokens


def diagnosis_match(pred: str, gold: str, lenient: bool = True) -> bool:
    """exact（归一化后全等）或 lenient（token 集合近似）。"""
    if not pred or not gold:
        return False
    p_norm, g_norm = normalize_diagnosis(pred), normalize_diagnosis(gold)
    if not p_norm or not g_norm:
        return False
    if p_norm == g_norm:
        return True
    if not lenient:
        return False
    p_tok, g_tok = diagnosis_tokens(pred), diagnosis_tokens(gold)
    if not p_tok or not g_tok:
        return False
    if p_tok == g_tok:
        return True
    inter = len(p_tok & g_tok)
    union = len(p_tok | g_tok)
    if union and inter / union >= 0.6:
        return True
    # 允许 2 个以内修饰词（如 intra-cranial / multiple），但禁止长句包含
    # gold 单词的情况，否则 “not a glioma” 也会命中 “glioma”。
    return (
        inter / min(len(p_tok), len(g_tok)) >= 0.8
        and len(p_tok ^ g_tok) <= 2
    )


# --------------------------------------------------------------------------- #
# 诊断关键词别名匹配（辅助指标）
# --------------------------------------------------------------------------- #

DEFAULT_ALIAS_PATH = Path(__file__).with_name("diagnosis_aliases.json")

# 只去掉纯语法词；保留 multiple / bilateral / disease / syndrome 等修饰词，
# 否则 "Multiple intra-cranial meningiomas" 会和 "Recurrent meningioma" 撞车。
_FUNCTION_WORDS = {
    "a", "an", "the", "of", "with", "and", "or", "in", "on", "at",
    "to", "for", "from", "by", "is", "are",
}

_ALIAS_CACHE: dict[str, list[list[str]]] | None = None


def keyword_tokens(text: str) -> set[str]:
    """保留修饰词、去掉语法词的规范化 token 集合，供别名命中使用。"""
    return {
        _canon_token(raw)
        for raw in _TOKEN_RE.findall((text or "").lower())
        if raw not in _FUNCTION_WORDS
    }


def load_diagnosis_aliases(
    path: str | Path = DEFAULT_ALIAS_PATH,
) -> dict[str, list[list[str]]]:
    """读取 ``{gold label: [[alias token, ...], ...]}``，按归一化 gold 建索引。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    aliases: dict[str, list[list[str]]] = {}
    for label, groups in raw.items():
        key = normalize_diagnosis(label)
        cleaned = [
            [_canon_token(token.lower()) for token in group]
            for group in groups
            if group
        ]
        if key and cleaned:
            aliases[key] = [group for group in cleaned if group]
    return aliases


def _default_aliases() -> dict[str, list[list[str]]]:
    global _ALIAS_CACHE
    if _ALIAS_CACHE is None:
        try:
            _ALIAS_CACHE = load_diagnosis_aliases()
        except (OSError, ValueError, json.JSONDecodeError):
            _ALIAS_CACHE = {}
    return _ALIAS_CACHE


def diagnosis_keyword_match(
    pred: str,
    gold: str,
    aliases: dict[str, list[list[str]]] | None = None,
) -> bool:
    """预测文本是否包含 gold 诊断的任一关键实体词/词组。

    比 strict/lenient 宽松：只要求医学实体（如 ``pilocytic``、``percheron``）
    出现，不要求复述 gold 整句。没有别名的 gold 回落到 lenient 匹配。
    """
    if not pred or not gold:
        return False
    table = _default_aliases() if aliases is None else aliases
    groups = table.get(normalize_diagnosis(gold))
    if not groups:
        return diagnosis_match(pred, gold, lenient=True)
    tokens = keyword_tokens(pred)
    return any(all(token in tokens for token in group) for group in groups)


# --------------------------------------------------------------------------- #
# Caption
# --------------------------------------------------------------------------- #

_CAPTION_TOKEN_RE = re.compile(r"[a-z0-9]+")


def caption_tokens(text: str) -> list[str]:
    return _CAPTION_TOKEN_RE.findall((text or "").lower())


def token_f1(pred: str, gold: str) -> tuple[float, float, float]:
    p_tokens, g_tokens = caption_tokens(pred), caption_tokens(gold)
    if not p_tokens or not g_tokens:
        return 0.0, 0.0, 0.0
    common = Counter(p_tokens) & Counter(g_tokens)
    same = sum(common.values())
    if not same:
        return 0.0, 0.0, 0.0
    precision = same / len(p_tokens)
    recall = same / len(g_tokens)
    return precision, recall, 2 * precision * recall / (precision + recall)


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(preds: Sequence[str], golds: Sequence[str], max_n: int = 4) -> float:
    """带 add-1 平滑的语料级 BLEU-4，避免依赖 nltk。"""
    matches = [0] * max_n
    totals = [0] * max_n
    pred_len = gold_len = 0
    for pred, gold in zip(preds, golds):
        p_tokens, g_tokens = caption_tokens(pred), caption_tokens(gold)
        pred_len += len(p_tokens)
        gold_len += len(g_tokens)
        for n in range(1, max_n + 1):
            p_ng, g_ng = _ngrams(p_tokens, n), _ngrams(g_tokens, n)
            totals[n - 1] += max(0, len(p_tokens) - n + 1)
            matches[n - 1] += sum((p_ng & g_ng).values())
    if not pred_len or not gold_len:
        return 0.0
    precisions = [
        (matches[i] + 1.0) / (totals[i] + 1.0) if totals[i] else 0.0
        for i in range(max_n)
    ]
    if min(precisions) <= 0:
        return 0.0
    brevity = min(1.0, math.exp(1.0 - gold_len / pred_len))
    return brevity * math.exp(sum(math.log(p) for p in precisions) / max_n)


# --------------------------------------------------------------------------- #
# 读取 / 报告
# --------------------------------------------------------------------------- #


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        head = value[0]
        if isinstance(head, dict):
            return str(head.get("content", ""))
        return _as_text(head)
    if isinstance(value, dict):
        return str(value.get("content", ""))
    return str(value)


def _messages_text(messages: Any, role: str = "assistant") -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == role:
            return _as_text(message.get("content", ""))
    return ""


def _images(messages: Any) -> list[str]:
    if not isinstance(messages, list):
        return []
    for message in messages:
        if isinstance(message, dict) and message.get("image"):
            value = message["image"]
            if isinstance(value, str):
                return [value]
            if isinstance(value, (list, tuple)):
                return [str(v) for v in value]
    return []


def _row_images(row: dict[str, Any]) -> list[str]:
    """Images live at row level in ms-swift 4.x; keep the old key as fallback."""
    value = row.get("images")
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    images = _images(row.get("messages"))
    if images:
        return images
    legacy = row.get("image")
    if isinstance(legacy, str):
        return [legacy]
    if isinstance(legacy, (list, tuple)):
        return [str(v) for v in legacy]
    return []


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL, skipping non-JSON preamble lines.

    ``swift infer`` prints ``run sh: ...`` to stdout before the JSON records,
    so a redirected prediction file starts with that line.
    """
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def extract_prediction(row: dict[str, Any]) -> str:
    for key in ("response", "predict", "prediction", "output", "completion"):
        if key in row and row[key] is not None:
            return _as_text(row[key])
    return _messages_text(row.get("messages"), role="assistant")


def extract_label(row: dict[str, Any]) -> str:
    for key in ("labels", "label", "solution", "gold", "target"):
        if key in row and row[key] is not None:
            return _as_text(row[key])
    return _messages_text(row.get("messages"), role="assistant")


def case_id_of(row: dict[str, Any], index: int) -> str:
    images = _row_images(row)
    candidate = images[0] if images else ""
    match = re.search(r"(case\d+)", Path(candidate).name.lower())
    if match:
        return match.group(1)
    return f"row{index:05d}"


@dataclass
class SampleResult:
    index: int
    case_id: str
    pred: str
    gold: str
    pred_parsed: bool = False
    gold_parsed: bool = False
    pred_final: str = ""
    pred_diffs: list[str] = field(default_factory=list)
    gold_final: str = ""
    pred_caption: str = ""
    gold_caption: str = ""
    final_strict: bool = False
    final_lenient: bool = False
    top5_candidates: list[str] = field(default_factory=list)
    top5_strict: bool = False
    top5_lenient: bool = False
    final_keyword: bool = False
    top5_keyword: bool = False
    loc: dict[str, float] = field(default_factory=dict)
    caption: dict[str, float] = field(default_factory=dict)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_rows(
    pred_rows: Sequence[dict[str, Any]],
    gold_rows: Sequence[dict[str, Any]] | None = None,
    img_size: float = IMG_SIZE,
    coord: str = "auto",
    source_image_size: float | None = None,
    lenient: bool = True,
    verbose: bool = False,
    aliases: dict[str, list[list[str]]] | None = None,
    box_scale: float = 1.0,
    collect: bool = False,
) -> dict[str, Any]:
    if gold_rows is not None and len(gold_rows) != len(pred_rows):
        raise ValueError(
            f"pred/gold size mismatch: {len(pred_rows)} vs {len(gold_rows)}"
        )
    if coord not in COORD_MODES:
        raise ValueError(f"coord must be one of {COORD_MODES}")
    aliases = _default_aliases() if aliases is None else aliases

    gold_texts = [
        extract_label(gold_rows[i] if gold_rows is not None else row)
        for i, row in enumerate(pred_rows)
    ]
    pred_texts = [extract_prediction(row) for row in pred_rows]

    gold_parsed = [parse_label(text, strict=False) for text in gold_texts]
    pred_parsed = [parse_prediction(text) for text in pred_texts]

    if coord == "auto":
        mode = detect_coord_mode([t.boxes for t in pred_parsed], img_size=img_size)
    else:
        mode = coord
    peak = max(
        (
            float(value)
            for target in pred_parsed
            for box in target.boxes
            for value in box[:4]
        ),
        default=0.0,
    )
    coord_ambiguous = (
        coord == "auto" and img_size < peak <= 1000.0
    )

    samples: list[SampleResult] = []
    parse_stats = {
        "empty_pred": 0,
        "no_box_pred": 0,
        "no_final_pred": 0,
        "empty_gold": 0,
        "no_gold_box": 0,
        "coord_clipped": 0,
        "coord_dropped": 0,
        "coord_swapped": 0,
    }

    for index, (row, pred_text, gold_text) in enumerate(
        zip(pred_rows, pred_texts, gold_texts)
    ):
        gold_target = gold_parsed[index]
        pred_target = pred_parsed[index]
        # `swift infer`/baseline outputs do not carry image paths; use the gold
        # row for case grouping when it is available.
        case_source = row
        if not _row_images(row) and gold_rows is not None:
            case_source = gold_rows[index]
        sample = SampleResult(
            index=index,
            case_id=case_id_of(case_source, index),
            pred=pred_text,
            gold=gold_text,
            pred_final=pred_target.final_diagnosis,
            pred_diffs=pred_target.differential,
            gold_final=gold_target.final_diagnosis,
            pred_caption=pred_target.caption,
            gold_caption=gold_target.caption,
        )
        sample.pred_parsed = bool(pred_text.strip()) and (
            bool(pred_target.final_diagnosis) or bool(pred_target.boxes)
        )
        sample.gold_parsed = bool(gold_target.final_diagnosis) or bool(gold_target.boxes)

        if not pred_text.strip():
            parse_stats["empty_pred"] += 1
        if not pred_target.boxes:
            parse_stats["no_box_pred"] += 1
        if not pred_target.final_diagnosis:
            parse_stats["no_final_pred"] += 1
        if not gold_text.strip():
            parse_stats["empty_gold"] += 1
        if not gold_target.boxes:
            parse_stats["no_gold_box"] += 1

        # --- 诊断 ---
        if gold_target.final_diagnosis:
            sample.final_strict = diagnosis_match(
                pred_target.final_diagnosis, gold_target.final_diagnosis, lenient=False
            )
            sample.final_lenient = diagnosis_match(
                pred_target.final_diagnosis, gold_target.final_diagnosis, lenient=lenient
            )
            candidates = [
                item
                for item in [pred_target.final_diagnosis] + list(pred_target.differential)
                if item
            ]
            sample.top5_candidates = candidates[:5]
            sample.top5_strict = any(
                diagnosis_match(item, gold_target.final_diagnosis, lenient=False)
                for item in sample.top5_candidates
            )
            sample.top5_lenient = any(
                diagnosis_match(item, gold_target.final_diagnosis, lenient=lenient)
                for item in sample.top5_candidates
            )
            sample.final_keyword = diagnosis_keyword_match(
                pred_target.final_diagnosis, gold_target.final_diagnosis, aliases
            )
            sample.top5_keyword = any(
                diagnosis_keyword_match(item, gold_target.final_diagnosis, aliases)
                for item in sample.top5_candidates
            )

        # --- 定位 ---
        if gold_target.boxes:
            pred_boxes, p_stats = convert_boxes(
                pred_target.boxes, mode, img_size, source_image_size
            )
            if abs(box_scale - 1.0) > 1e-9:
                pred_boxes = scale_boxes(pred_boxes, box_scale, img_size)
            gold_boxes, g_stats = convert_boxes(
                gold_target.boxes, "pixel", img_size, source_image_size
            )
            for key in ("clipped", "dropped", "swapped"):
                parse_stats[f"coord_{key}"] += p_stats[key] + g_stats[key]
            sample.loc = localization_metrics(pred_boxes, gold_boxes)

        # --- 描述 ---
        if gold_target.caption:
            precision, recall, f1 = token_f1(pred_target.caption, gold_target.caption)
            sample.caption = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "empty": 0.0 if pred_target.caption else 1.0,
                "char_len": float(len(pred_target.caption)),
            }

        samples.append(sample)
        if verbose:
            print(_format_sample(sample))

    # --- 汇总 ---
    loc_samples = [s.loc for s in samples if s.loc]
    loc_agg: dict[str, float] = {}
    if loc_samples:
        for key in loc_samples[0]:
            loc_agg[key] = _mean([item[key] for item in loc_samples])
        loc_agg["samples"] = float(len(loc_samples))
        loc_agg["exact_box_count_rate"] = _mean(
            [1.0 if item["n_pred"] == item["n_gold"] else 0.0 for item in loc_samples]
        )

    cap_samples = [s.caption for s in samples if s.caption]
    cap_agg = {
        "precision": _mean([c["precision"] for c in cap_samples]),
        "recall": _mean([c["recall"] for c in cap_samples]),
        "f1": _mean([c["f1"] for c in cap_samples]),
        "bleu4": corpus_bleu(
            [s.pred_caption for s in samples if s.gold_caption],
            [s.gold_caption for s in samples if s.gold_caption],
        ),
        "empty_rate": _mean([c["empty"] for c in cap_samples]),
        "samples": float(len(cap_samples)),
    }

    diag_samples = [s for s in samples if s.gold_final]
    diag_agg = {
        "samples": float(len(diag_samples)),
        "top1_strict": _mean([1.0 if s.final_strict else 0.0 for s in diag_samples]),
        "top1_lenient": _mean([1.0 if s.final_lenient else 0.0 for s in diag_samples]),
        "top5_strict": _mean([1.0 if s.top5_strict else 0.0 for s in diag_samples]),
        "top5_lenient": _mean([1.0 if s.top5_lenient else 0.0 for s in diag_samples]),
        "top1_keyword": _mean([1.0 if s.final_keyword else 0.0 for s in diag_samples]),
        "top5_keyword": _mean([1.0 if s.top5_keyword else 0.0 for s in diag_samples]),
        "empty_final_rate": _mean(
            [1.0 if not s.pred_final else 0.0 for s in diag_samples]
        ),
    }

    # --- case 级诊断（多切片投票） ---
    by_case: dict[str, list[SampleResult]] = defaultdict(list)
    for sample in samples:
        by_case[sample.case_id].append(sample)
    case_hits: list[float] = []
    case_hits_lenient: list[float] = []
    case_hits_keyword: list[float] = []
    case_hits_top5_keyword: list[float] = []
    inconsistent_gold = 0
    for _, group in by_case.items():
        golds = {normalize_diagnosis(s.gold_final) for s in group if s.gold_final}
        if not golds:
            continue
        if len(golds) > 1:
            inconsistent_gold += 1
        gold_final = group[0].gold_final
        votes = Counter(
            normalize_diagnosis(s.pred_final) for s in group if s.pred_final
        )
        if votes:
            case_pred = votes.most_common(1)[0][0]
        else:
            case_pred = ""
        case_hits.append(1.0 if diagnosis_match(case_pred, gold_final, lenient=False) else 0.0)
        case_hits_lenient.append(
            1.0 if diagnosis_match(case_pred, gold_final, lenient=True) else 0.0
        )
        # 关键词指标按切片多数票，不依赖措辞一致
        graded = [s for s in group if s.gold_final]
        kw_hits = sum(1.0 for s in graded if s.final_keyword)
        kw_hits5 = sum(1.0 for s in graded if s.top5_keyword)
        case_hits_keyword.append(
            1.0 if graded and kw_hits * 2 > len(graded) else 0.0
        )
        case_hits_top5_keyword.append(
            1.0 if graded and kw_hits5 * 2 > len(graded) else 0.0
        )
    case_agg = {
        "cases": float(len(case_hits)),
        "top1_strict": _mean(case_hits),
        "top1_lenient": _mean(case_hits_lenient),
        "top1_keyword": _mean(case_hits_keyword),
        "top5_keyword": _mean(case_hits_top5_keyword),
        "inconsistent_gold": float(inconsistent_gold),
    }

    return {
        "meta": {
            "pred_rows": len(pred_rows),
            "gold_rows": len(gold_rows) if gold_rows is not None else len(pred_rows),
            "images": len(samples),
            "img_size": img_size,
            "coord_mode": mode,
            "coord_peak": peak,
            "coord_ambiguous": coord_ambiguous,
            "match": "lenient" if lenient else "strict",
            "alias_labels": len(aliases),
            "box_scale": box_scale,
        },
        "parsing": parse_stats,
        "localization": loc_agg,
        "caption": cap_agg,
        "diagnosis": diag_agg,
        "cases": case_agg,
        "per_sample": [s.__dict__ for s in samples] if (verbose or collect) else [],
    }


def _format_sample(sample: SampleResult) -> str:
    loc = sample.loc
    iou_text = f"{loc.get('mean_iou', 0):.3f}" if loc else "-"
    diag = "✅" if sample.final_lenient else ("🔶" if sample.top5_lenient else "❌")
    return (
        f"[{sample.index:03d}] {sample.case_id} {diag} "
        f"iou={iou_text} gold={sample.gold_final[:40]!r} pred={sample.pred_final[:40]!r}"
    )


def format_report(report: dict[str, Any]) -> str:
    meta = report["meta"]
    loc = report["localization"]
    cap = report["caption"]
    diag = report["diagnosis"]
    cases = report["cases"]
    parsing = report["parsing"]

    lines = [
        "=" * 72,
        f"Nova evaluation | images={meta['images']} coord={meta['coord_mode']} "
        f"match={meta['match']}",
        "=" * 72,
    ]
    if meta.get("coord_ambiguous"):
        lines += [
            f"WARNING: max coordinate={meta.get('coord_peak'):.1f} lies between 480 and 1000;",
            "         auto picked norm1000. If the model emitted original pixels, rerun with",
            "         --coord pixel --source_image_size <W>.",
            "",
        ]
    lines.append("")
    lines.append("📍 异常定位（一对一匹配，480 像素空间）")
    if loc.get("samples"):
        for key_t in ("0.3", "0.5", "0.75"):
            lines.append(
                f"  IoU>={key_t}: P={loc.get(f'precision@{key_t}', 0):.4f} "
                f"R={loc.get(f'recall@{key_t}', 0):.4f} "
                f"F1={loc.get(f'f1@{key_t}', 0):.4f}"
            )
        lines.append(f"  mean IoU (matched): {loc.get('mean_iou', 0):.4f}")
        lines.append(f"  mAP@[.50:.95]:      {loc.get('map50_95', 0):.4f}")
        lines.append(f"  box-count exact:    {loc.get('exact_box_count_rate', 0):.4f}")
    else:
        lines.append("  (no gold boxes)")

    lines += ["", "📝 图像描述"]
    lines.append(f"  Token P/R/F1: {cap['precision']:.4f} / {cap['recall']:.4f} / {cap['f1']:.4f}")
    lines.append(f"  BLEU-4:       {cap['bleu4']:.4f}")
    lines.append(f"  empty rate:   {cap['empty_rate']:.4f}")

    lines += ["", "🩺 诊断推理（gold final 作为唯一正确答案）"]
    lines.append(
        f"  Top-1 strict/lenient: {diag['top1_strict']:.4f} / {diag['top1_lenient']:.4f}"
    )
    lines.append(
        f"  Top-5 strict/lenient: {diag['top5_strict']:.4f} / {diag['top5_lenient']:.4f}"
    )
    lines.append(
        f"  Top-1/Top-5 keyword:  {diag.get('top1_keyword', 0):.4f} / "
        f"{diag.get('top5_keyword', 0):.4f}  (aliases={meta.get('alias_labels', 0)})"
    )
    lines.append(f"  empty final rate:     {diag['empty_final_rate']:.4f}")
    lines.append(
        f"  case-level Top-1:     {cases['top1_strict']:.4f} strict / "
        f"{cases['top1_lenient']:.4f} lenient / "
        f"{cases.get('top1_keyword', 0):.4f} keyword  (cases={int(cases['cases'])})"
    )

    lines += ["", "🔍 解析质量"]
    lines.append(
        "  " + " ".join(f"{k}={v}" for k, v in parsing.items() if v)
        if any(parsing.values())
        else "  clean"
    )
    lines.append("=" * 72)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Nova evaluation")
    parser.add_argument("--pred", required=True, help="prediction jsonl")
    parser.add_argument("--gold", default=None, help="optional gold jsonl")
    parser.add_argument("--image_size", type=float, default=IMG_SIZE)
    parser.add_argument("--coord", choices=COORD_MODES, default="auto")
    parser.add_argument(
        "--source_image_size",
        type=float,
        default=None,
        help="coordinate space of the predictions when it is raw pixels (e.g. 512)",
    )
    parser.add_argument(
        "--strict_match",
        action="store_true",
        help="diagnosis match requires exact normalized equality",
    )
    parser.add_argument("--out", default=None, help="write json report here")
    parser.add_argument(
        "--aliases",
        default=None,
        help=f"diagnosis alias json (default: {DEFAULT_ALIAS_PATH.name})",
    )
    parser.add_argument(
        "--box_scale",
        type=float,
        default=1.0,
        help="scale predicted boxes about their center before matching (e.g. 1.3)",
    )
    parser.add_argument("--verbose", action="store_true", help="per-sample output")
    args = parser.parse_args(argv)

    pred_rows = read_rows(args.pred)
    gold_rows = read_rows(args.gold) if args.gold else None
    aliases = (
        load_diagnosis_aliases(args.aliases) if args.aliases else None
    )
    report = evaluate_rows(
        pred_rows,
        gold_rows,
        img_size=args.image_size,
        coord=args.coord,
        source_image_size=args.source_image_size,
        lenient=not args.strict_match,
        verbose=args.verbose,
        aliases=aliases,
        box_scale=args.box_scale,
    )
    print(format_report(report))
    if args.out:
        Path(args.out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
