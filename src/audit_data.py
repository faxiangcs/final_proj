#!/usr/bin/env python3
"""Audit generated Nova SFT/GRPO splits against the canonical schema.

Checks (A01-A16)
----------------
A01  case-level split 零重叠
A02  图片存在
A02b 多模态字段符合 ms-swift 4.x（top-level ``images``，非 message 内 ``image``）
A03  同一张图片只出现在一个 split
A04  标签中无 ``not found``
A05  final diagnosis 非空
A05b 渲染后的 label 能 round-trip 回结构化 target（final/differential/caption/boxes）
A06  differential 非空、去重、<= max_diff、不等于 final
A07  bbox 合法（坐标范围 / extent / 数量）
A08  bbox 无重复
A09  caption 非空；统计 panel caption
A10  prompt 版本一致（三段式标题）
A11  token 预算：prompt + vision + completion <= max_length
A12  标签可被 ``nova_schema.parse_label`` 解析
A13  标签分布统计
A14  token 长度分布
A15  文件 sha256
A16  标签仍满足当前 DiagnosisReward 的旧格式门槛

用法
----
    python src/audit_data.py --data_dir data/nova_sft
    python src/audit_data.py --data_dir data/nova_sft \
        --tokenizer /path/to/llava-1.5-7b-hf --max_length 2048

没有 tokenizer 时用 ``nova_schema.estimate_text_tokens`` 估算，A11 标记为估计值。
审计失败时退出码为 1。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.nova_schema import (  # noqa: E402
    IMG_SIZE,
    LABEL_VERSION,
    MAX_BOXES,
    MAX_DIFF,
    PANEL_RE,
    PROMPT_VERSION,
    NovaTarget,
    estimate_text_tokens,
    parse_label,
    validate_target,
)

SPLITS = ("train", "val", "test")

# 现有 medical_reward.py:DiagnosisReward 的硬编码格式门槛。
# 本轮数据必须继续满足它，否则旧 reward 会把所有样本判成格式错误。
REWARD_FORMAT_RE = re.compile(
    r"^\s*考虑鉴别诊断包括.+。结合特征，最终诊断为：.+", re.DOTALL
)


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def add(self, check_id: str, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append(
            {"id": check_id, "name": name, "ok": bool(ok), "detail": detail}
        )
        if not ok:
            self.errors.append(f"{check_id} {name}: {detail}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def case_id_from_image(image: str) -> str:
    match = re.search(r"(case\d+)", os.path.basename(image or ""))
    return match.group(1) if match else ""


def resolve_image(image: str, image_root: str | None) -> str | None:
    if image and os.path.exists(image):
        return image
    if image_root:
        candidate = os.path.join(image_root, os.path.basename(image or ""))
        if os.path.exists(candidate):
            return candidate
    return None


def load_tokenizer(path: str | None, audit: Audit):
    if not path:
        return None
    try:
        from transformers import AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        audit.warn(f"A11 uses tokenizer: {path}")
        return tokenizer
    except Exception as exc:  # noqa: BLE001
        audit.warn(f"failed to load tokenizer {path!r}: {exc}; falling back to heuristic")
        return None


def count_tokens(tokenizer, text: str) -> int:
    if tokenizer is None:
        return estimate_text_tokens(text)
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return ordered[idx]


def audit_split(
    audit: Audit,
    data_dir: Path,
    split: str,
    args: argparse.Namespace,
    tokenizer,
    expected_targets: dict[str, dict],
) -> dict:
    sft_path = data_dir / f"{split}.jsonl"
    grpo_path = data_dir / f"grpo_{split}.jsonl"

    stats: dict = {
        "split": split,
        "sft_file": str(sft_path),
        "grpo_file": str(grpo_path),
        "sft_exists": sft_path.exists(),
        "grpo_exists": grpo_path.exists(),
    }
    if not sft_path.exists() or not grpo_path.exists():
        audit.add(
            "A00",
            "split files exist",
            False,
            f"missing {sft_path if not sft_path.exists() else grpo_path}",
        )
        return stats

    sft_rows = read_jsonl(sft_path)
    grpo_rows = read_jsonl(grpo_path)
    stats["sft_count"] = len(sft_rows)
    stats["grpo_count"] = len(grpo_rows)

    structure_errors: list[str] = []
    image_missing: list[str] = []
    not_found: list[str] = []
    prompt_bad: list[str] = []
    parse_errors: list[str] = []
    reward_format_bad: list[str] = []
    roundtrip_errors: list[str] = []
    field_errors: dict[str, list[str]] = {
        "final": [],
        "differential": [],
        "boxes": [],
        "caption": [],
    }
    duplicate_box_errors: list[str] = []
    token_over: list[str] = []
    cases: set[str] = set()
    images: set[str] = set()
    box_hist: Counter = Counter()
    diff_hist: Counter = Counter()
    final_hist: Counter = Counter()
    prompt_tokens: list[int] = []
    label_tokens: list[int] = []
    sft_total_tokens: list[int] = []
    grpo_total_tokens: list[int] = []
    panel_count = 0
    targets: dict[str, NovaTarget] = {}
    sft_labels_by_image: dict[str, str] = {}

    for idx, row in enumerate(sft_rows):
        where = f"{split}[{idx}]"
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            structure_errors.append(f"{where}: messages must have 2 turns")
            continue
        user, assistant = messages
        if user.get("role") != "user":
            structure_errors.append(f"{where}: first role must be user")
        if assistant.get("role") != "assistant":
            structure_errors.append(f"{where}: second role must be assistant")

        prompt = user.get("content") or ""
        row_images = row.get("images") or []
        image = row_images[0] if isinstance(row_images, list) and row_images else ""
        if "image" in user:
            structure_errors.append(
                f"{where}: message-level image is ignored by ms-swift 4.x; "
                "use top-level images"
            )
        if not isinstance(row_images, list) or len(row_images) != 1:
            structure_errors.append(
                f"{where}: expected exactly 1 top-level image, got {row_images!r}"
            )
        label = assistant.get("content") or ""

        if not image:
            structure_errors.append(f"{where}: missing image")
        if not label:
            structure_errors.append(f"{where}: empty assistant content")
            continue

        cases.add(case_id_from_image(image))
        images.add(image)
        sft_labels_by_image[image] = label

        if not resolve_image(image, args.image_root):
            image_missing.append(image)

        if "not found" in label.lower():
            not_found.append(where)

        prompt_ok = (
            prompt.startswith("作为神经影像AI助手")
            and "1. 异常定位" in prompt
            and "2. 图像描述" in prompt
            and "3. 诊断推理" in prompt
        )
        if not prompt_ok:
            prompt_bad.append(where)

        try:
            target = parse_label(label, strict=True)
        except ValueError as exc:
            parse_errors.append(f"{where}: {exc}")
            continue
        targets[image] = target

        expected = expected_targets.get(image)
        if expected:
            exp_target = expected.get("target") or {
                "boxes": expected.get("boxes", []),
                "caption": expected.get("caption", ""),
                "final_diagnosis": expected.get("final_diagnosis", ""),
                "differential": expected.get("differential", []),
            }
            if target.final_diagnosis != exp_target.get("final_diagnosis", ""):
                roundtrip_errors.append(
                    f"{where}: final {target.final_diagnosis!r} != "
                    f"{exp_target.get('final_diagnosis')!r}"
                )
            if target.differential != exp_target.get("differential", []):
                roundtrip_errors.append(
                    f"{where}: differential {target.differential!r} != "
                    f"{exp_target.get('differential')!r}"
                )
            if target.caption != exp_target.get("caption", ""):
                roundtrip_errors.append(f"{where}: caption mismatch")
            exp_boxes = exp_target.get("boxes", [])
            if len(target.boxes) != len(exp_boxes) or any(
                len(got) != 4
                or len(want) != 4
                # Integer rendering rounds each coordinate by <= 0.5 px.
                or any(abs(float(g) - float(w)) > 1.0 for g, w in zip(got, want))
                for got, want in zip(target.boxes, exp_boxes)
            ):
                roundtrip_errors.append(f"{where}: boxes mismatch")

        diag_text = label.split("【诊断推理】：")[-1]
        if not REWARD_FORMAT_RE.search(diag_text):
            reward_format_bad.append(where)

        errors = validate_target(
            target,
            img_size=IMG_SIZE,
            max_boxes=MAX_BOXES,
            max_diff=MAX_DIFF,
            require_boxes=True,
            require_differential=True,
        )
        for field, items in errors.items():
            field_errors[field].extend(f"{where}: {item}" for item in items)
        duplicate_box_errors.extend(
            f"{where}: {item}" for item in errors.get("boxes", []) if "duplicate box" in item
        )

        box_hist[len(target.boxes)] += 1
        diff_hist[len(target.differential)] += 1
        final_hist[target.final_diagnosis] += 1
        if PANEL_RE.search(target.caption):
            panel_count += 1

        p_tokens = count_tokens(tokenizer, prompt)
        l_tokens = count_tokens(tokenizer, label)
        prompt_tokens.append(p_tokens)
        label_tokens.append(l_tokens)
        sft_total = p_tokens + args.vision_tokens + l_tokens
        grpo_total = p_tokens + args.vision_tokens + args.max_completion_tokens
        sft_total_tokens.append(sft_total)
        grpo_total_tokens.append(grpo_total)
        if sft_total > args.max_length or grpo_total > args.max_length:
            token_over.append(
                f"{where}: sft={sft_total} grpo={grpo_total} > {args.max_length}"
            )

    # GRPO structure / consistency
    grpo_structure: list[str] = []
    grpo_parse: list[str] = []
    grpo_mismatch: list[str] = []
    grpo_images: set[str] = set()
    for idx, row in enumerate(grpo_rows):
        where = f"grpo_{split}[{idx}]"
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 1:
            grpo_structure.append(f"{where}: messages must have 1 user turn")
            continue
        if messages[0].get("role") != "user":
            grpo_structure.append(f"{where}: first role must be user")
        row_images = row.get("images") or []
        image = row_images[0] if isinstance(row_images, list) and row_images else ""
        if "image" in messages[0]:
            grpo_structure.append(
                f"{where}: message-level image is ignored by ms-swift 4.x; "
                "use top-level images"
            )
        if not isinstance(row_images, list) or len(row_images) != 1:
            grpo_structure.append(
                f"{where}: expected exactly 1 top-level image, got {row_images!r}"
            )
        grpo_images.add(image)
        solution = row.get("solution") or ""
        if not solution:
            grpo_structure.append(f"{where}: missing solution")
            continue
        try:
            target = parse_label(solution, strict=True)
        except ValueError as exc:
            grpo_parse.append(f"{where}: {exc}")
            continue
        errors = validate_target(target)
        if errors:
            grpo_parse.append(f"{where}: {errors}")
        if image in sft_labels_by_image and solution != sft_labels_by_image[image]:
            grpo_mismatch.append(f"{where}: solution != SFT label for {image}")

    stats.update(
        {
            "cases": len(cases),
            "images": len(images),
            "sft_image_set": sorted(images),
            "box_hist": dict(sorted(box_hist.items())),
            "diff_hist": dict(sorted(diff_hist.items())),
            "panel_caption_images": panel_count,
            "prompt_token_p50": percentile(prompt_tokens, 0.50),
            "prompt_token_p90": percentile(prompt_tokens, 0.90),
            "prompt_token_max": max(prompt_tokens, default=0),
            "label_token_p50": percentile(label_tokens, 0.50),
            "label_token_p90": percentile(label_tokens, 0.90),
            "label_token_max": max(label_tokens, default=0),
            "sft_total_token_max": max(sft_total_tokens, default=0),
            "grpo_total_token_max": max(grpo_total_tokens, default=0),
            "top_final": final_hist.most_common(5),
            "structure_errors": structure_errors[:20],
            "image_missing": image_missing[:20],
            "not_found": not_found[:20],
            "prompt_bad": prompt_bad[:20],
            "parse_errors": parse_errors[:20],
            "reward_format_bad": reward_format_bad[:20],
            "roundtrip_errors": roundtrip_errors[:20],
            "grpo_structure_errors": grpo_structure[:20],
            "grpo_parse_errors": grpo_parse[:20],
            "grpo_label_mismatch": grpo_mismatch[:20],
        }
    )

    audit.add(
        "A02",
        f"{split}: images exist",
        not image_missing,
        f"{len(image_missing)} missing" + (f", e.g. {image_missing[:3]}" if image_missing else ""),
    )
    audit.add(
        "A02b",
        f"{split}: ms-swift images field",
        not structure_errors,
        f"{len(structure_errors)} structure errors"
        + (f", e.g. {structure_errors[:2]}" if structure_errors else ""),
    )
    audit.add(
        "A04",
        f"{split}: no 'not found'",
        not not_found,
        f"{len(not_found)} labels contain 'not found'",
    )
    audit.add(
        "A05",
        f"{split}: final diagnosis",
        not field_errors["final"],
        f"{len(field_errors['final'])} errors" + (f", e.g. {field_errors['final'][:2]}" if field_errors["final"] else ""),
    )
    audit.add(
        "A05b",
        f"{split}: label round-trip",
        not roundtrip_errors,
        f"{len(roundtrip_errors)} render/parse mismatches"
        + (f", e.g. {roundtrip_errors[:2]}" if roundtrip_errors else ""),
    )
    audit.add(
        "A06",
        f"{split}: differential",
        not field_errors["differential"],
        f"{len(field_errors['differential'])} errors" + (f", e.g. {field_errors['differential'][:2]}" if field_errors["differential"] else ""),
    )
    audit.add(
        "A07",
        f"{split}: boxes",
        not field_errors["boxes"],
        f"{len(field_errors['boxes'])} errors" + (f", e.g. {field_errors['boxes'][:2]}" if field_errors["boxes"] else ""),
    )
    audit.add(
        "A08",
        f"{split}: duplicate boxes",
        not duplicate_box_errors,
        f"{len(duplicate_box_errors)} duplicates",
    )
    audit.add(
        "A09",
        f"{split}: caption",
        not field_errors["caption"],
        f"{len(field_errors['caption'])} empty; panel_caption={panel_count}",
    )
    audit.add(
        "A10",
        f"{split}: prompt version",
        not prompt_bad,
        f"{len(prompt_bad)} prompts not matching {PROMPT_VERSION}",
    )
    audit.add(
        "A11",
        f"{split}: token budget",
        not token_over,
        f"{len(token_over)} over max_length={args.max_length}"
        + (f", e.g. {token_over[:2]}" if token_over else ""),
    )
    audit.add(
        "A12",
        f"{split}: label parseable",
        not parse_errors and not grpo_parse,
        f"sft={len(parse_errors)} grpo={len(grpo_parse)} parse errors",
    )
    audit.add(
        "A16",
        f"{split}: legacy reward format",
        not reward_format_bad,
        f"{len(reward_format_bad)} labels fail current DiagnosisReward format gate",
    )
    audit.add(
        "A12b",
        f"{split}: GRPO structure",
        not grpo_structure and not grpo_mismatch,
        f"structure={len(grpo_structure)} mismatch={len(grpo_mismatch)}",
    )
    audit.add(
        "A13",
        f"{split}: label distribution",
        True,
        f"cases={len(cases)} images={len(images)} boxes={dict(sorted(box_hist.items()))} "
        f"diffs={dict(sorted(diff_hist.items()))} top_final={final_hist.most_common(3)}",
    )
    audit.add(
        "A14",
        f"{split}: token distribution",
        True,
        f"prompt p50/p90/max={percentile(prompt_tokens,0.5)}/{percentile(prompt_tokens,0.9)}/{max(prompt_tokens, default=0)}; "
        f"label p50/p90/max={percentile(label_tokens,0.5)}/{percentile(label_tokens,0.9)}/{max(label_tokens, default=0)}",
    )
    audit.add(
        "A15",
        f"{split}: file hash",
        True,
        f"sft={sha256_file(sft_path)[:16]}... grpo={sha256_file(grpo_path)[:16]}...",
    )

    # cross-check SFT/GRPO counts and image sets
    if len(sft_rows) != len(grpo_rows):
        audit.add(
            "A12c",
            f"{split}: SFT/GRPO count",
            False,
            f"sft={len(sft_rows)} grpo={len(grpo_rows)}",
        )
    if images != grpo_images:
        audit.add(
            "A12d",
            f"{split}: SFT/GRPO image set",
            False,
            f"only_sft={len(images - grpo_images)} only_grpo={len(grpo_images - images)}",
        )

    stats["targets"] = {image: target.to_dict() for image, target in targets.items()}
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/nova_sft")
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_completion_tokens", type=int, default=512)
    parser.add_argument("--vision_tokens", type=int, default=576)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="optional HF tokenizer path; without it A11 uses a heuristic",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        print(f"data_dir not found: {data_dir}", file=sys.stderr)
        return 2

    audit = Audit()
    expected_targets: dict[str, dict] = {}
    audit_jsonl = data_dir / "audit.jsonl"
    if audit_jsonl.exists():
        for row in read_jsonl(audit_jsonl):
            if row.get("image"):
                expected_targets[row["image"]] = row
    else:
        audit.warn("audit.jsonl missing; A05b round-trip check is skipped")

    tokenizer = load_tokenizer(args.tokenizer, audit)
    if tokenizer is None:
        audit.warn(
            "A11 uses heuristic token counts; pass --tokenizer for exact counts. "
            "Heuristic: CJK*1.2 + English_word*1.3 + other*0.3."
        )

    split_stats = {
        split: audit_split(audit, data_dir, split, args, tokenizer, expected_targets)
        for split in SPLITS
    }

    # A01 / A03: cross-split leakage
    image_sets = {
        split: {image for image in stats.get("sft_image_set", []) if image}
        for split, stats in split_stats.items()
    }
    case_sets = {
        split: {
            case_id
            for image in images
            if (case_id := case_id_from_image(image))
        }
        for split, images in image_sets.items()
    }
    case_overlap = (
        (case_sets["train"] & case_sets["val"])
        | (case_sets["train"] & case_sets["test"])
        | (case_sets["val"] & case_sets["test"])
    )
    audit.add(
        "A01",
        "case-level split disjoint",
        not case_overlap,
        f"overlap={sorted(case_overlap)[:10]}",
    )

    image_overlap = (
        (image_sets["train"] & image_sets["val"])
        | (image_sets["train"] & image_sets["test"])
        | (image_sets["val"] & image_sets["test"])
    )
    audit.add(
        "A03",
        "image-level split disjoint",
        not image_overlap,
        f"overlap={len(image_overlap)}",
    )

    # Manifest consistency
    manifest_path = data_dir / "manifest.json"
    manifest = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            audit.warn(f"cannot read manifest: {exc}")
    if manifest:
        counts = manifest.get("counts", {})
        expected = {
            "train": counts.get("train_images"),
            "val": counts.get("val_images"),
            "test": counts.get("test_images"),
        }
        actual = {split: split_stats[split].get("sft_count") for split in SPLITS}
        mismatch = {k: (expected[k], actual[k]) for k in SPLITS if expected[k] != actual[k]}
        audit.add(
            "A13b",
            "manifest counts",
            not mismatch,
            f"mismatch={mismatch}" if mismatch else "ok",
        )
        if manifest.get("prompt_version") != PROMPT_VERSION:
            audit.add(
                "A10b",
                "manifest prompt version",
                False,
                f"manifest={manifest.get('prompt_version')} code={PROMPT_VERSION}",
            )

    # A17: optional localization-only auxiliary SFT rows.  They intentionally
    # contain no caption/diagnosis, so they are validated separately from the
    # full three-section checks above.
    aux_path = data_dir / "train_aux_boxes.jsonl"
    aux_expected = (manifest or {}).get("counts", {}).get("aux_box_rows", 0)
    if aux_path.exists() or aux_expected:
        aux_errors: list[str] = []
        aux_images: set[str] = set()
        aux_rows = read_jsonl(aux_path) if aux_path.exists() else []
        if not aux_path.exists():
            aux_errors.append("train_aux_boxes.jsonl missing")
        if len(aux_rows) != aux_expected:
            aux_errors.append(f"rows={len(aux_rows)} manifest={aux_expected}")
        for row_no, row in enumerate(aux_rows, 1):
            image = (row.get("images") or [""])[0]
            if image:
                aux_images.add(image)
            messages = row.get("messages") or []
            label = messages[-1].get("content", "") if messages else ""
            prompt = messages[0].get("content", "") if messages else ""
            if "只输出异常定位" not in prompt:
                aux_errors.append(f"row {row_no}: prompt is not box-only")
            if not label.startswith("【异常定位】："):
                aux_errors.append(f"row {row_no}: label lacks box heading")
                continue
            target = parse_label(label, strict=False)
            if not target.boxes:
                aux_errors.append(f"row {row_no}: empty boxes")
                continue
            box_errors = validate_target(
                target,
                img_size=IMG_SIZE,
                max_boxes=MAX_BOXES,
                max_diff=MAX_DIFF,
                require_boxes=True,
                require_differential=False,
            ).get("boxes", [])
            if box_errors:
                aux_errors.append(f"row {row_no}: {'; '.join(box_errors)}")
            if target.caption or target.final_diagnosis or target.differential:
                aux_errors.append(f"row {row_no}: auxiliary label has extra sections")
        leaked = aux_images - image_sets["train"]
        if leaked:
            aux_errors.append(f"images outside train split: {len(leaked)}")
        audit.add(
            "A17",
            "aux box-only rows",
            not aux_errors,
            (
                f"rows={len(aux_rows)} images={len(aux_images)}"
                if not aux_errors
                else "; ".join(aux_errors[:5])
            ),
        )

    hashes = {}
    for name in ("manifest.json", "audit.jsonl"):
        path = data_dir / name
        if path.exists():
            hashes[name] = sha256_file(path)

    report = {
        "ok": not audit.errors,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "schema": {
            "prompt_version": PROMPT_VERSION,
            "label_version": LABEL_VERSION,
            "img_size": IMG_SIZE,
            "max_boxes": MAX_BOXES,
            "max_diff": MAX_DIFF,
        },
        "settings": {
            "max_length": args.max_length,
            "max_completion_tokens": args.max_completion_tokens,
            "vision_tokens": args.vision_tokens,
            "tokenizer": args.tokenizer or "heuristic",
            "image_root": args.image_root,
        },
        "checks": audit.checks,
        "errors": audit.errors,
        "warnings": audit.warnings,
        "splits": {
            split: {k: v for k, v in stats.items() if k != "targets"}
            for split, stats in split_stats.items()
        },
        "hashes": hashes,
    }

    out_path = Path(args.out) if args.out else data_dir / "audit_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for check in audit.checks:
        mark = "OK " if check["ok"] else "FAIL"
        print(f"[{mark}] {check['id']:5s} {check['name']}: {check['detail']}")
    for warning in audit.warnings:
        print(f"[WARN] {warning}")
    print(f"\nreport: {out_path}")
    print("AUDIT PASSED" if report["ok"] else "AUDIT FAILED")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
