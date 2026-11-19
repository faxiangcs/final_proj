"""Canonical Nova prompt/label schema, renderer, parser and validator.

The schema is the single source of truth for data construction, rewards, and
evaluation. It fixes the three-section prompt/label format, the 480-pixel
coordinate space, and the box/differential limits.

Canonical prompt
----------------
作为神经影像AI助手，请结合患者病史：【...】，分析该脑部MRI影像。
请严格按照以下三个部分输出；固定标题保持中文，正文内容（图像描述、鉴别诊断、
最终诊断）一律使用英文：
1. 异常定位：给出病灶的坐标边界框 [x_min, y_min, x_max, y_max]，坐标为 0–480 像素。
2. 图像描述：描述影像中的异常特征。
3. 诊断推理：给出最多 4 个英文鉴别诊断（用分号分隔，不要编号），
   并以“结合特征，最终诊断为：<英文诊断>。”结尾。

Canonical label
---------------
【异常定位】：[x1, y1, x2, y2], ...
【图像描述】：...
【诊断推理】：考虑鉴别诊断包括 d1; d2; ...。结合特征，最终诊断为：final。

所有数据构建、奖励、评测均从这里导入，避免旧/新格式漂移。当前
`src/build_dataset.py`、`src/audit_data.py`、`medical_reward.py` 和
`src/eval_nova.py` 都使用本模块。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

IMG_SIZE = 480
MAX_BOXES = 3
MAX_DIFF = 4
NMS_IOU = 0.5

PROMPT_VERSION = "old_three_sections_v2"
PROMPT_BOX_ONLY_VERSION = "box_only_v1"
LABEL_VERSION = "old_three_sections_v1"
LABEL_VERSION_INT = "old_three_sections_v1_int"

SECTION_BOX = "【异常定位】："
SECTION_CAPTION = "【图像描述】："
SECTION_DIAG = "【诊断推理】："

PANEL_RE = re.compile(r"\([A-H]\)|\b[A-H]\s*[,;]")

BOX_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)

FINAL_RE = re.compile(
    r"(?:最终(?:的)?(?:临床)?诊断\*{0,2}(?:是|为)?"
    r"|final\s+diagnosis\s*(?:is)?"
    r"|(?:the\s+)?most\s+likely\s+diagnosis\s*(?:is|:))"
    r"\s*[：:]?\s*\*{0,2}\s*(.{0,400}?)(?:\n\s*\n|\Z)",
    re.S | re.I,
)

DIFF_RE = re.compile(
    r"(?:(?:考虑)?鉴别诊断(?:包括|为)?"
    r"|differential\s+diagnosis(?:\s+includes)?"
    r"|diagnostic\s+reasoning)"
    r"\s*[：:]?\s*(.*?)(?:。|\Z)",
    re.S | re.I,
)

_SENTENCE_TAIL_RE = re.compile(
    r"\.\s+(?=(?:given|the|however|therefore|considering|in|this|these|"
    r"based|combining|overall|thus|patient|imaging)\b)",
    re.I,
)

# 模型常用 "1. X；2. Y" / "(1) X" / "- X" 列举；统一剥掉前缀再比较。
_ITEM_PREFIX_RE = re.compile(
    r"^\s*(?:\d{1,2}\s*[.、)]|\(\s*\d{1,2}\s*\)"
    r"|[A-Za-z]\s*[.)](?=\s*[A-Za-z(])|[-*•·])\s*"
)

HEADING_RE = {
    "boxes": re.compile(
        r"(?:^|\n)\s*(?:#{1,6}\s*)?(?:\d+[\.、]\s*)?(?:\*\*)?(?:【)?"
        r"(?:异常定位|abnormal\s+location|locali[sz]ation|lesion\s+locali[sz]ation)"
        r"(?:】)?(?:\*\*)?\s*[：:]?\s*",
        re.I,
    ),
    "caption": re.compile(
        r"(?:^|\n)\s*(?:#{1,6}\s*)?(?:\d+[\.、]\s*)?(?:\*\*)?(?:【)?"
        r"(?:图像描述|image\s+description|caption|imaging\s+findings)"
        r"(?:】)?(?:\*\*)?\s*[：:]?\s*",
        re.I,
    ),
    "diagnosis": re.compile(
        r"(?:^|\n)\s*(?:#{1,6}\s*)?(?:\d+[\.、]\s*)?(?:\*\*)?(?:【)?"
        r"(?:诊断推理|diagnosis\s+reasoning|diagnostic\s+reasoning|diagnosis"
        r"|diagnostic\s+impression)"
        r"(?:】)?(?:\*\*)?\s*[：:]?\s*",
        re.I,
    ),
}


@dataclass
class NovaTarget:
    """Structured target for one Nova sample."""

    boxes: list[list[float]] = field(default_factory=list)
    caption: str = ""
    final_diagnosis: str = ""
    differential: list[str] = field(default_factory=list)
    legacy_format: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "boxes": self.boxes,
            "caption": self.caption,
            "final_diagnosis": self.final_diagnosis,
            "differential": self.differential,
            "legacy_format": self.legacy_format,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NovaTarget":
        return cls(
            boxes=[list(map(float, b)) for b in data.get("boxes", [])],
            caption=str(data.get("caption", "")),
            final_diagnosis=str(data.get("final_diagnosis", "")),
            differential=[str(x) for x in data.get("differential", [])],
            legacy_format=bool(data.get("legacy_format", False)),
        )

    def to_label(self) -> str:
        return render_label(self)


def normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split())


def is_bad_text(text: str) -> bool:
    s = normalize_whitespace(text).lower()
    return (
        not s
        or "not found" in s
        or s in {"final diagnosis not found.", "differential diagnosis not found."}
    )


_PREFIX_CLEANERS = (
    # "Regarding the whole file, we believe it is falx EMH"
    re.compile(
        r"^regarding\b[^,;:]*[,;:]\s*(?:we|i)\s+(?:believe|think|conclude|suspect)"
        r"\s+(?:that\s+)?(?:it|this|the (?:lesion|finding))\s+(?:is|represents)?\s*",
        re.I,
    ),
    # "This patient was diagnosed as having isolated agenesis ..."
    re.compile(
        r"^this\s+(?:patient|case)\s+(?:was|is|had been)\s+diagnosed"
        r"\s+(?:as\s+having|with)\s*",
        re.I,
    ),
    # "Laboratory tests and MRI confirmed ...", "Surgically and histologically
    # confirmed ...", "MRI findings were suggestive of ..."
    re.compile(
        r"^(?:laboratory|lab|imaging|mri|ct|histology|histological|biopsy|"
        r"surgical|surgery|pathology)[^.;:]*?\b"
        r"(?:confirmed|showed|revealed|suggested|suggestive of|consistent with)\s*",
        re.I,
    ),
    # "Final diagnosis: X", "Most likely diagnosis is X"
    re.compile(
        r"^(?:the\s+)?(?:final|clinical|most likely)\s+diagnosis"
        r"\s*(?:is|:|was)\s*",
        re.I,
    ),
    re.compile(r"^diagnosis\s*[:\-]\s*", re.I),
    re.compile(r"^case of\s+", re.I),
)

_TRAILING_PATIENT_CLAUSE = re.compile(
    r"\s+in (?:a|an) (?:patient|adult|child|man|woman|boy|girl|infant)\b.*$",
    re.I,
)


def clean_final_diagnosis(text: str) -> str:
    """Strip descriptive boilerplate from a final-diagnosis field.

    Nova's metadata occasionally stores a sentence rather than a disease name,
    e.g. ``Regarding the whole file, we believe it is falx EMH``.  Those
    prefixes make exact/lenient matching impossible and add noise to SFT.
    """
    original = normalize_whitespace(text).strip(" .。")
    s = original
    case_of = re.match(r"^case of\b", s, re.I) is not None
    changed = True
    while changed and s:
        changed = False
        for pattern in _PREFIX_CLEANERS:
            candidate = pattern.sub("", s, count=1).strip(" .。:;-")
            if candidate != s:
                s = candidate
                changed = True
    # Only "Case of <disease> in a patient with <symptom>" is safe to trim.
    # "Adult onset seizures in a patient with neurocysticercosis" keeps the
    # disease after "with", so trimming there would delete the answer.
    if case_of:
        s = _TRAILING_PATIENT_CLAUSE.sub("", s).strip(" .。")
    return s or original


def iou(a: list[float], b: list[float]) -> float:
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def clip_nms_boxes(
    rows: list[dict],
    max_boxes: int = MAX_BOXES,
    nms_iou: float = NMS_IOU,
    img_size: int = IMG_SIZE,
) -> list[list[float]]:
    """gold x,y,width,height -> clipped [x1,y1,x2,y2] + NMS + cap."""
    boxes: list[list[float]] = []
    for row in rows:
        try:
            x1 = float(row["x"])
            y1 = float(row["y"])
            x2 = x1 + float(row["width"])
            y2 = y1 + float(row["height"])
        except (KeyError, TypeError, ValueError):
            continue

        x1 = min(max(x1, 0.0), float(img_size))
        x2 = min(max(x2, 0.0), float(img_size))
        y1 = min(max(y1, 0.0), float(img_size))
        y2 = min(max(y2, 0.0), float(img_size))

        if x2 - x1 <= 1.0 or y2 - y1 <= 1.0:
            continue
        boxes.append([x1, y1, x2, y2])

    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    keep: list[list[float]] = []
    for box in boxes:
        if all(iou(box, other) <= nms_iou for other in keep):
            keep.append(box)
    return keep[:max_boxes]


def _clean_item(part: str) -> str:
    """Normalize one differential item: drop colon, list numbering and bullets."""
    s = normalize_whitespace(part).strip()
    s = s.lstrip("：:;；、 ").strip()
    s = _ITEM_PREFIX_RE.sub("", s).strip()
    # Drop trailing reasoning glued to the last English differential:
    # "Hemorrhagic contusion. Given the history, ..."
    s = _SENTENCE_TAIL_RE.split(s, maxsplit=1)[0].strip()
    if s.rstrip(" .。*").endswith((":", "：")):
        return ""  # category header such as "Cortically based neoplasms:"
    return s.strip(" .。*：:")


def clean_differential_list(
    raw: str,
    final: str,
    max_diff: int = MAX_DIFF,
) -> list[str]:
    """Split / clean / dedupe differential diagnosis list."""
    final_key = normalize_whitespace(final).strip(" .。").lower()
    out: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[;；、\n]", raw or ""):
        s = _clean_item(part)
        if is_bad_text(s):
            continue
        key = s.lower()
        if key in seen or key == final_key:
            continue
        seen.add(key)
        out.append(s)
    return out[:max_diff]


def render_prompt(history: str) -> str:
    return (
        "作为神经影像AI助手，请结合患者病史："
        f"【{normalize_whitespace(history)}】，分析该脑部MRI影像。\n"
        "请严格按照以下三个部分输出；固定标题保持中文，正文内容"
        "（图像描述、鉴别诊断、最终诊断）一律使用英文：\n"
        "1. 异常定位：给出病灶的坐标边界框 [x_min, y_min, x_max, y_max]，"
        "坐标为 0–480 像素。\n"
        "2. 图像描述：描述影像中的异常特征。\n"
        "3. 诊断推理：给出最多 4 个英文鉴别诊断（用分号分隔，不要编号），"
        "并以“结合特征，最终诊断为：<英文诊断>。”结尾。"
    )


def render_box_prompt(history: str) -> str:
    """Localization-only prompt used for auxiliary SFT rows.

    The first sentence stays identical to :func:`render_prompt`, so the model
    sees the same image/history conditioning; the trailing instruction tells it
    to emit only the localization section.  At inference the full prompt is
    still used, so this auxiliary task does not change the output contract.
    """
    return (
        "作为神经影像AI助手，请结合患者病史："
        f"【{normalize_whitespace(history)}】，定位该脑部MRI影像中的异常病灶。\n"
        "只输出异常定位这一个部分，固定标题保持中文：\n"
        "【异常定位】：给出病灶的坐标边界框 "
        "[x_min, y_min, x_max, y_max]，坐标为 0–480 像素；"
        "多个病灶的框用逗号分隔，不要输出图像描述或诊断。"
    )


def _format_box(box: list[float], integer: bool = False) -> str:
    if integer:
        x1, y1, x2, y2 = (int(round(float(v))) for v in box[:4])
        return f"[{x1}, {y1}, {x2}, {y2}]"
    return (
        f"[{float(box[0]):.2f}, {float(box[1]):.2f}, "
        f"{float(box[2]):.2f}, {float(box[3]):.2f}]"
    )


def render_box_label(
    boxes: list[list[float]], integer_boxes: bool = False
) -> str:
    return SECTION_BOX + ", ".join(
        _format_box(box, integer=integer_boxes) for box in boxes
    )


def render_label(target: NovaTarget, integer_boxes: bool = False) -> str:
    box_str = ", ".join(
        _format_box(b, integer=integer_boxes) for b in target.boxes
    )
    diffs = "; ".join(normalize_whitespace(d) for d in target.differential)
    final = normalize_whitespace(target.final_diagnosis).strip(" .。")
    return (
        f"{SECTION_BOX}{box_str}\n"
        f"{SECTION_CAPTION}{normalize_whitespace(target.caption)}\n"
        f"{SECTION_DIAG}考虑鉴别诊断包括 {diffs}。"
        f"结合特征，最终诊断为：{final}。"
    )


def _split_sections(text: str) -> dict[str, str]:
    hits: list[tuple[int, int, str]] = []
    for key, pattern in HEADING_RE.items():
        match = pattern.search(text or "")
        if match:
            hits.append((match.start(), match.end(), key))
    hits.sort()

    sections: dict[str, str] = {}
    for i, (_, end, key) in enumerate(hits):
        next_start = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        sections[key] = text[end:next_start].strip()
    return sections


def parse_boxes(text: str) -> list[list[float]]:
    return [
        [float(x) for x in match.groups()]
        for match in BOX_RE.finditer(text or "")
    ]


def _split_items(raw: str) -> list[str]:
    items: list[str] = []
    for part in re.split(r"[;；、\n]", raw or ""):
        s = _clean_item(part)
        if s:
            items.append(s)
    return items


def _extract_final_phrase(block: str) -> str:
    """Pull the diagnosis phrase out of a free-form final-diagnosis block.

    Qwen-style baselines often write ``**最终临床诊断：**\\n...最可能的诊断是
    **X**。``; canonical labels write ``最终诊断为：X。``.
    """
    text = normalize_whitespace((block or "").replace("**", "")).strip(" *：:")
    if not text:
        return ""
    for pattern in (
        r"最可能的(?:临床)?诊断(?:是|为)\s*[：:]?\s*([^\n。；;]{2,200})",
        r"(?:最终)?诊断(?:是|为)\s*[：:]?\s*([^\n。；;]{2,200})",
        r"考虑(?:为|是)\s*([^\n。；;]{2,200})",
        r"most likely diagnosis(?: is|:)\s*(?:a |an )?([^\n.]{2,200})",
        r"(?:final|primary) diagnosis(?: is|:)\s*(?:a |an )?([^\n.]{2,200})",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip(" *：:")
    generic = {
        "病史分析", "影像分析", "综合判断", "诊断依据", "鉴别诊断", "分析",
        "结论", "影像表现", "影像学表现", "病史", "依据", "临床关联",
    }
    for bold in re.finditer(r"\*\*([^*\n]{2,200})\*\*", block or ""):
        phrase = bold.group(1).strip(" *：:")
        if phrase and phrase not in generic:
            return phrase
    first = re.split(r"[。\n]|\.\s", text, maxsplit=1)[0]
    return first.strip(" .。*：:")[:200]


def parse_diagnosis(diag_text: str) -> tuple[str, list[str], bool]:
    """Return (final, differential, legacy_new_format)."""
    raw = (diag_text or "").strip()
    if not raw:
        return "", [], False

    # 取最后一个“最终诊断”块：baseline 常在中间先提一次、结尾给出真正结论。
    matches = list(FINAL_RE.finditer(raw))
    final_match = matches[-1] if matches else None
    if final_match:
        final = _extract_final_phrase(final_match.group(1))
        diff_match = DIFF_RE.search(raw)
        if diff_match:
            diffs = _split_items(diff_match.group(1))
        else:
            # English free-form often omits the "differential diagnosis" label:
            # the section starts directly with "A; B; C. Given the history ...".
            prefix = raw[: final_match.start()]
            prefix = re.split(r"\.\s+(?=[A-Z])", prefix, maxsplit=1)[0]
            diffs = _split_items(prefix)
        legacy = False
    else:
        # 兼容旧的“5 个诊断分号分隔”格式：第一个视为 final。
        # 但整段英文推理没有分号时，第一句往往不是诊断，宁可留空也不误判。
        items = _split_items(raw)
        if len(items) >= 2:
            final, diffs = items[0], items[1:]
        elif items and len(items[0]) <= 120:
            final, diffs = items[0], []
        else:
            final, diffs = "", []
        legacy = True

    cleaned: list[str] = []
    seen: set[str] = set()
    final_key = final.lower()
    for diff in diffs:
        key = diff.lower()
        if not diff or is_bad_text(diff) or key == final_key or key in seen:
            continue
        seen.add(key)
        cleaned.append(diff)
    return final, cleaned, legacy


def parse_label(text: str, strict: bool = True) -> NovaTarget:
    """Parse the canonical label (or a model output when strict=False)."""
    sections = _split_sections(text)
    missing = [key for key in ("boxes", "caption", "diagnosis") if key not in sections]
    if strict and missing:
        raise ValueError(f"missing sections: {missing}")

    if sections:
        box_text = sections.get("boxes", "")
        caption = normalize_whitespace(sections.get("caption", ""))
        diag_text = sections.get("diagnosis", "")
    else:
        # 没有任何标题时，尽量从全文提取框；诊断部分用全文兜底。
        box_text = text
        caption = ""
        diag_text = text

    final, diffs, legacy = parse_diagnosis(diag_text)
    return NovaTarget(
        boxes=parse_boxes(box_text),
        caption=caption,
        final_diagnosis=final,
        differential=diffs,
        legacy_format=legacy,
    )


def _try_json_object(text: str) -> dict[str, Any] | None:
    s = (text or "").strip()
    candidates: list[str] = []
    if s.startswith("{"):
        candidates.append(s)
    match = re.search(r"\{.*\}", s, re.S)
    if match and match.group(0) != s:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_json_prediction(text: str) -> NovaTarget | None:
    obj = _try_json_object(text)
    if obj is None:
        return None

    boxes = obj.get("boxes") or []
    parsed_boxes: list[list[float]] = []
    for box in boxes:
        if isinstance(box, dict):
            try:
                parsed_boxes.append(
                    [
                        float(box["x1"]),
                        float(box["y1"]),
                        float(box["x2"]),
                        float(box["y2"]),
                    ]
                )
            except (KeyError, TypeError, ValueError):
                continue
        elif isinstance(box, (list, tuple)) and len(box) == 4:
            try:
                parsed_boxes.append([float(x) for x in box])
            except (TypeError, ValueError):
                continue

    final = (
        obj.get("final_diagnosis")
        or obj.get("final")
        or obj.get("diagnosis")
        or ""
    )
    diffs = obj.get("differential") or obj.get("differentials") or []
    if isinstance(diffs, str):
        diffs = _split_items(diffs)
    caption = obj.get("caption") or obj.get("image_description") or ""

    return NovaTarget(
        boxes=parsed_boxes,
        caption=normalize_whitespace(str(caption)),
        final_diagnosis=normalize_whitespace(str(final)),
        differential=[normalize_whitespace(str(x)) for x in diffs if str(x).strip()],
        legacy_format=False,
    )


def parse_prediction(text: str) -> NovaTarget:
    """Lenient parser for model outputs: JSON first, then three-section text."""
    parsed = parse_json_prediction(text)
    if parsed is not None:
        return parsed
    return parse_label(text, strict=False)


def validate_target(
    target: NovaTarget,
    img_size: int = IMG_SIZE,
    max_boxes: int = MAX_BOXES,
    max_diff: int = MAX_DIFF,
    require_boxes: bool = True,
    require_differential: bool = True,
) -> dict[str, list[str]]:
    """Return {field: [errors]}; empty dict means valid."""
    errors: dict[str, list[str]] = {
        "final": [],
        "differential": [],
        "boxes": [],
        "caption": [],
    }

    final = normalize_whitespace(target.final_diagnosis)
    if not final:
        errors["final"].append("empty final_diagnosis")
    elif "not found" in final.lower():
        errors["final"].append("final_diagnosis contains 'not found'")

    diffs = target.differential or []
    if require_differential and not diffs:
        errors["differential"].append("empty differential")
    if len(diffs) > max_diff:
        errors["differential"].append(
            f"too many differential: {len(diffs)} > {max_diff}"
        )
    seen_diff: set[str] = set()
    for diff in diffs:
        d = normalize_whitespace(diff)
        if not d:
            errors["differential"].append("empty differential item")
            continue
        if "not found" in d.lower():
            errors["differential"].append(f"contains 'not found': {d}")
        if d.lower() == final.lower():
            errors["differential"].append(f"duplicates final: {d}")
        if d.lower() in seen_diff:
            errors["differential"].append(f"duplicate: {d}")
        seen_diff.add(d.lower())

    boxes = target.boxes or []
    if require_boxes and not boxes:
        errors["boxes"].append("empty boxes")
    if len(boxes) > max_boxes:
        errors["boxes"].append(f"too many boxes: {len(boxes)} > {max_boxes}")
    seen_box: set[tuple[float, float, float, float]] = set()
    for box in boxes:
        if len(box) != 4:
            errors["boxes"].append(f"box must have 4 coords: {box}")
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in box)
        except (TypeError, ValueError):
            errors["boxes"].append(f"non-numeric box: {box}")
            continue
        if x1 < 0 or y1 < 0 or x2 > img_size or y2 > img_size:
            errors["boxes"].append(f"out of range: {box}")
        if x2 <= x1 or y2 <= y1:
            errors["boxes"].append(f"invalid extent: {box}")
        key = (round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2))
        if key in seen_box:
            errors["boxes"].append(f"duplicate box: {box}")
        seen_box.add(key)

    if not normalize_whitespace(target.caption):
        errors["caption"].append("empty caption")

    return {field: items for field, items in errors.items() if items}


def assert_valid_target(target: NovaTarget, **kwargs) -> None:
    errors = validate_target(target, **kwargs)
    if errors:
        parts = [f"{field}: {'; '.join(items)}" for field, items in errors.items()]
        raise ValueError(" | ".join(parts))


def validate_label_text(text: str, strict: bool = True, **kwargs) -> dict[str, list[str]]:
    try:
        target = parse_label(text, strict=strict)
    except ValueError as exc:
        return {"parse": [str(exc)]}
    return validate_target(target, **kwargs)


def estimate_text_tokens(text: str) -> int:
    """Rough token estimate when no tokenizer is available.

    CJK ~1.2 token/char, English ~1.3 token/word, other ~0.3 token/char.
    Deliberately conservative; use a real tokenizer for final decisions.
    """
    cjk = len(re.findall(r"[一-鿿]", text or ""))
    words = re.findall(r"[A-Za-z]+", text or "")
    other = max(0, len(text or "") - cjk - sum(len(w) for w in words))
    return int(cjk * 1.2 + len(words) * 1.3 + other * 0.3)
