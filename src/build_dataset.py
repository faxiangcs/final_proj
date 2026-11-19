#!/usr/bin/env python3
"""Build Nova SFT/GRPO train/val/test splits from local CSV annotations.

设计目标
--------
1. 沿用之前的 ms-swift 数据格式（4.x：图片必须放在 row 级 ``images``，
   message 内写 ``image`` 会被静默丢弃）：
   - SFT:  {"messages": [user, assistant], "images": [path]}
   - GRPO: {"messages": [user], "solution": assistant_label, "images": [path]}
2. 沿用旧的三段式 prompt / label，保证现有 medical_reward.py 和
   eval_all_metric.py 可以直接解析。
3. 按病例切分 train/val/test，避免同一病人的切片跨集合。
4. 清洗标签：去掉 not found、空 caption、无效框；gold 框 clip + NMS + 上限。
5. 输出 manifest / audit，便于复查。

用法
----
    python src/build_dataset.py \
        --nova_root Nova \
        --image_root Nova/images \
        --out_dir data/nova_sft \
        --seed 42

如果训练机上的图片路径不同，重新运行并指定 --image_root 即可。

默认输出
--------
    data/nova_sft/train.jsonl
    data/nova_sft/train_aux_boxes.jsonl   # 仅 --aux_boxes_ratio > 0
    data/nova_sft/val.jsonl
    data/nova_sft/test.jsonl
    data/nova_sft/grpo_train.jsonl
    data/nova_sft/grpo_val.jsonl
    data/nova_sft/grpo_test.jsonl
    data/nova_sft/audit.jsonl
    data/nova_sft/manifest.json
    data/nova_sft/README.md
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

# 支持 `python src/build_dataset.py` 和 `python -m src.build_dataset` 两种方式。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.nova_schema import (  # noqa: E402
    IMG_SIZE,
    LABEL_VERSION,
    LABEL_VERSION_INT,
    MAX_BOXES,
    MAX_DIFF,
    PANEL_RE,
    PROMPT_BOX_ONLY_VERSION,
    PROMPT_VERSION,
    NovaTarget,
    clean_differential_list,
    clean_final_diagnosis,
    clip_nms_boxes,
    is_bad_text,
    normalize_whitespace,
    render_box_label,
    render_box_prompt,
    render_label,
    render_prompt,
)


def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_case_ids(
    case_ids: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    max_cases: int,
) -> dict[str, list[str]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    rng = random.Random(seed)
    ids = list(case_ids)
    rng.shuffle(ids)

    if max_cases and max_cases > 0:
        ids = ids[:max_cases]

    n = len(ids)
    if n < 3:
        raise ValueError(f"need at least 3 cases, got {n}")

    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    n_test = n - n_train - n_val

    return {
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train : n_train + n_val]),
        "test": sorted(ids[n_train + n_val : n_train + n_val + n_test]),
    }


def write_jsonl(path: str, records: list[dict], with_solution: bool) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            item = {
                "messages": [
                    {
                        "role": "user",
                        "content": r["prompt"],
                    }
                ],
                # ms-swift 4.x reads multimodal inputs from a row-level `images`
                # column. A `image` key inside the message is silently dropped by
                # RowPreprocessor, which makes training text-only.
                "images": [r["image"]],
            }
            if with_solution:
                item["solution"] = r["label"]
            else:
                item["messages"].append(
                    {"role": "assistant", "content": r["label"]}
                )
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nova_root", default="Nova")
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--out_dir", default="data/nova_sft")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument(
        "--max_cases",
        type=int,
        default=0,
        help="0 = use all valid cases; otherwise randomly use N cases",
    )
    parser.add_argument("--max_boxes", type=int, default=MAX_BOXES)
    parser.add_argument("--max_diff", type=int, default=MAX_DIFF)
    parser.add_argument(
        "--drop_panel_captions",
        action="store_true",
        help="drop captions that reference multi-panel figures (all splits)",
    )
    parser.add_argument(
        "--drop_train_panel_captions",
        action="store_true",
        help="drop panel captions from the train split only, after splitting; "
        "keeps val/test identical to the un-dropped run",
    )
    parser.add_argument(
        "--clean_final",
        action="store_true",
        help="strip descriptive boilerplate from final_diagnosis "
        "('Regarding the whole file, we believe it is ...', 'Case of ...')",
    )
    parser.add_argument(
        "--integer_boxes",
        action="store_true",
        help="render box coordinates as integers (0-480) instead of "
        "two-decimal values; cuts about 39 percent of box tokens and "
        "removes the '.00' pattern",
    )
    parser.add_argument(
        "--aux_boxes_ratio",
        type=float,
        default=0.0,
        help="fraction (0-1) of train images duplicated as localization-only "
        "auxiliary SFT rows; written to train_aux_boxes.jsonl",
    )
    args = parser.parse_args()
    if not 0.0 <= args.aux_boxes_ratio <= 1.0:
        parser.error("--aux_boxes_ratio must be in [0, 1]")
    label_version = LABEL_VERSION_INT if args.integer_boxes else LABEL_VERSION

    nova_root = os.path.abspath(args.nova_root)
    image_root = os.path.abspath(
        args.image_root or os.path.join(nova_root, "images")
    )
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    meta = {r["case_id"]: r for r in read_csv(os.path.join(nova_root, "case_metadata.csv"))}
    captions = read_csv(os.path.join(nova_root, "captions.csv"))
    gold: dict[str, list[dict]] = defaultdict(list)
    for row in read_csv(os.path.join(nova_root, "bboxes_gold.csv")):
        gold[row["filename"]].append(row)

    dropped = Counter()
    records_by_case: dict[str, list[dict]] = defaultdict(list)
    all_box_counts: list[int] = []
    panel_count = 0

    for cap in sorted(captions, key=lambda r: r["filename"]):
        filename = cap["filename"]
        case_id = cap["case_id"]
        image_path = os.path.join(image_root, filename)

        if not os.path.exists(image_path):
            dropped["missing_image"] += 1
            continue

        case_meta = meta.get(case_id)
        if not case_meta:
            dropped["missing_meta"] += 1
            continue

        history = normalize_whitespace(case_meta.get("clinical_history", ""))
        if not history:
            dropped["empty_history"] += 1
            continue

        final = normalize_whitespace(case_meta.get("final_diagnosis", "")).strip(" .。")
        if args.clean_final:
            final = clean_final_diagnosis(final)
        if is_bad_text(final):
            dropped["bad_final"] += 1
            continue

        diffs = clean_differential_list(
            case_meta.get("differential_diagnosis", ""),
            final,
            max_diff=args.max_diff,
        )
        if not diffs:
            dropped["empty_differential"] += 1
            continue

        caption = normalize_whitespace(cap.get("caption", ""))
        if not caption:
            dropped["empty_caption"] += 1
            continue

        is_panel = bool(PANEL_RE.search(caption))
        if is_panel:
            panel_count += 1
        if args.drop_panel_captions and is_panel:
            dropped["panel_caption"] += 1
            continue

        boxes = clip_nms_boxes(
            gold.get(filename, []),
            max_boxes=args.max_boxes,
            img_size=IMG_SIZE,
        )
        if not boxes:
            dropped["no_valid_boxes"] += 1
            continue

        target = NovaTarget(
            boxes=boxes,
            caption=caption,
            final_diagnosis=final,
            differential=diffs,
        )
        record = {
            "id": os.path.splitext(filename)[0],
            "case_id": case_id,
            "filename": filename,
            "image": image_path,
            "history": history,
            "prompt": render_prompt(history),
            "label": render_label(target, integer_boxes=args.integer_boxes),
            "target": target.to_dict(),
            "boxes": boxes,
            "caption": caption,
            "final": final,
            "diffs": diffs,
            "panel_caption": is_panel,
        }
        records_by_case[case_id].append(record)
        all_box_counts.append(len(boxes))

    valid_cases = sorted(records_by_case)
    if not valid_cases:
        raise SystemExit("no valid cases after filtering; check --image_root and CSVs")

    split_cases = split_case_ids(
        valid_cases,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
        args.max_cases,
    )

    split_records: dict[str, list[dict]] = {}
    for split, cases in split_cases.items():
        recs = [r for c in cases for r in records_by_case[c]]
        recs.sort(key=lambda r: r["filename"])
        split_records[split] = recs

    # Safety: no case leakage.
    train_cases = set(split_cases["train"])
    val_cases = set(split_cases["val"])
    test_cases = set(split_cases["test"])
    assert not (train_cases & val_cases)
    assert not (train_cases & test_cases)
    assert not (val_cases & test_cases)

    # Optional train-only panel-caption removal, after the split has been
    # frozen, so val/test stay byte-comparable with runs that keep them.
    # A case whose every image is a panel caption keeps its rows: each case is
    # a distinct disease, and deleting all of them would drop the disease from
    # training entirely.
    train_panel_dropped = 0
    if args.drop_train_panel_captions:
        by_case: dict[str, list[dict]] = defaultdict(list)
        for rec in split_records["train"]:
            by_case[rec["case_id"]].append(rec)
        kept: list[dict] = []
        for recs in by_case.values():
            non_panel = [rec for rec in recs if not rec["panel_caption"]]
            if non_panel:
                kept.extend(non_panel)
                train_panel_dropped += len(recs) - len(non_panel)
            else:
                kept.extend(recs)
        kept.sort(key=lambda rec: rec["filename"])
        split_records["train"] = kept
        train_cases = {rec["case_id"] for rec in kept}
        if train_panel_dropped:
            dropped["train_panel_caption"] += train_panel_dropped

    train_recs = split_records["train"]
    val_recs = split_records["val"]
    test_recs = split_records["test"]

    # Localization-only auxiliary rows.  They live in a separate file so the
    # SFT/GRPO row-for-row audit and the GRPO ``solution`` contract stay intact;
    # run_sft.sh passes both files to ms-swift when the aux file exists.
    aux_recs: list[dict] = []
    if args.aux_boxes_ratio > 0:
        n_aux = max(1, int(round(len(train_recs) * args.aux_boxes_ratio)))
        rng = random.Random(args.seed + 17)
        for rec in rng.sample(train_recs, min(n_aux, len(train_recs))):
            aux_recs.append(
                {
                    "id": f"{rec['id']}_loc",
                    "image": rec["image"],
                    "prompt": render_box_prompt(rec["history"]),
                    "label": render_box_label(
                        rec["boxes"], integer_boxes=args.integer_boxes
                    ),
                }
            )
    aux_path = os.path.join(out_dir, "train_aux_boxes.jsonl")
    if aux_recs:
        write_jsonl(aux_path, aux_recs, with_solution=False)
    elif os.path.exists(aux_path):
        os.remove(aux_path)

    # SFT / inference format
    write_jsonl(os.path.join(out_dir, "train.jsonl"), train_recs, with_solution=False)
    write_jsonl(os.path.join(out_dir, "val.jsonl"), val_recs, with_solution=False)
    write_jsonl(os.path.join(out_dir, "test.jsonl"), test_recs, with_solution=False)

    # GRPO format
    write_jsonl(os.path.join(out_dir, "grpo_train.jsonl"), train_recs, with_solution=True)
    write_jsonl(os.path.join(out_dir, "grpo_val.jsonl"), val_recs, with_solution=True)
    write_jsonl(os.path.join(out_dir, "grpo_test.jsonl"), test_recs, with_solution=True)

    # Audit file for inspection.
    with open(os.path.join(out_dir, "audit.jsonl"), "w", encoding="utf-8") as f:
        for split, recs in split_records.items():
            for r in recs:
                item = {
                    "id": r["id"],
                    "case_id": r["case_id"],
                    "split": split,
                    "image": r["image"],
                    "n_boxes": len(r["boxes"]),
                    "boxes": r["boxes"],
                    "target": r["target"],
                    "final_diagnosis": r["final"],
                    "differential": r["diffs"],
                    "caption": r["caption"],
                    "panel_caption": r["panel_caption"],
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    box_counter = Counter(all_box_counts)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "nova_root": nova_root,
        "image_root": image_root,
        "out_dir": out_dir,
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "max_cases": args.max_cases,
        "max_boxes": args.max_boxes,
        "max_diff": args.max_diff,
        "drop_panel_captions": args.drop_panel_captions,
        "drop_train_panel_captions": args.drop_train_panel_captions,
        "train_panel_dropped": train_panel_dropped,
        "clean_final": args.clean_final,
        "integer_boxes": args.integer_boxes,
        "aux_boxes_ratio": args.aux_boxes_ratio,
        "prompt_version": PROMPT_VERSION,
        "prompt_box_only_version": (
            PROMPT_BOX_ONLY_VERSION if aux_recs else None
        ),
        "label_version": label_version,
        "label_format": (
            "【异常定位】：...\\n【图像描述】：...\\n"
            "【诊断推理】：考虑鉴别诊断包括 ...。结合特征，最终诊断为：...。"
        ),
        "counts": {
            "valid_cases": len(valid_cases),
            "valid_images": sum(len(v) for v in split_records.values()),
            "train_cases": len(train_cases),
            "train_images": len(train_recs),
            "val_cases": len(val_cases),
            "val_images": len(val_recs),
            "test_cases": len(test_cases),
            "test_images": len(test_recs),
            "panel_caption_images": panel_count,
            "aux_box_rows": len(aux_recs),
        },
        "splits": {
            "train": {"case_ids": sorted(train_cases)},
            "val": {"case_ids": split_cases["val"]},
            "test": {"case_ids": split_cases["test"]},
        },
        "dropped": dict(dropped),
        "box_stats": {
            "distribution": {str(k): v for k, v in sorted(box_counter.items())},
            "mean": round(sum(all_box_counts) / len(all_box_counts), 4),
            "max": max(all_box_counts),
        },
        "notes": [
            "case-level split; no case appears in more than one split",
            "gold boxes only; clipped to 480x480; NMS IoU 0.5; capped",
            "not found / empty final / empty differential / empty caption dropped",
        ],
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    readme = f"""# Nova SFT / GRPO 数据分片

由 `src/build_dataset.py` 生成。

## 统计

| Split | Cases | Images |
|---|---:|---:|
| train | {len(train_cases)} | {len(train_recs)} |
| val | {len(val_cases)} | {len(val_recs)} |
| test | {len(test_cases)} | {len(test_recs)} |
| **total** | **{len(valid_cases)}** | **{sum(len(v) for v in split_records.values())}** |

- 切分方式：case-level，seed={args.seed}，比例 {args.train_ratio}/{args.val_ratio}/{args.test_ratio}
- 图像路径根目录：`{image_root}`
- Prompt 版本：`{PROMPT_VERSION}`
- 标签版本：`{label_version}`
- 数据契约：`src/nova_schema.py`（prompt/label 渲染、解析、校验的唯一来源）
- 标签：旧的三段式格式，兼容现有 `medical_reward.py` / `eval_all_metric.py`
- 框：gold only；clip 到 480×480；NMS IoU 0.5；每图最多 {args.max_boxes} 个
- 坐标格式：{"整数 0–480（省 token）" if args.integer_boxes else "两位小数"}
- 辅助定位样本：{len(aux_recs)} 条（`train_aux_boxes.jsonl`，仅 SFT 使用）
- 过滤：`not found`、空 final、空 differential、空 caption、无有效框
- 被过滤统计：`{dict(dropped)}`

## 文件

| 文件 | 用途 |
|---|---|
| `train.jsonl` / `val.jsonl` / `test.jsonl` | SFT / `swift infer` 格式 |
| `train_aux_boxes.jsonl` | 仅定位的辅助 SFT 样本（可选，`--aux_boxes_ratio`） |
| `grpo_train.jsonl` / `grpo_val.jsonl` / `grpo_test.jsonl` | GRPO 格式（含 `solution`） |
| `audit.jsonl` | 逐样本审计信息（含结构化 `target`） |
| `manifest.json` | 统计、split case list、过滤原因 |

## 数据审计

生成后必须跑一次：

```bash
python src/audit_data.py --data_dir data/nova_sft
```

它会检查 A01–A16：case 零重叠、图片存在、无 `not found`、final/differential/box 合法、
prompt 版本一致、标签可被 `nova_schema.parse_label` 解析、旧 reward 格式门槛、token 预算、
文件 hash 等。审计不通过时退出码为 1。

## 训练提示

- SFT 建议 `--max_length 2048`。
- GRPO 建议 `--max_length 2048`（至少 1536），`--max_completion_length 512`；不要再用 768/1024。
- 旧三段式格式可以直接被现有 `medical_reward.py` 的 `LocalizationReward` / `DiagnosisReward` 解析。
- 想快速跑一个小实验时，加 `--max_cases 60`，会先随机取 60 个病例再按比例切分。
- 根目录旧的 `demo_*.jsonl`、`grpo_nova_*.jsonl` 已弃用；不要和新数据混用。

## 重新生成

```bash
python src/build_dataset.py \\
  --nova_root Nova \\
  --image_root Nova/images \\
  --out_dir data/nova_sft \\
  --seed 42
```

训练机图片路径不同时，只改 `--image_root`，例如：

```bash
python src/build_dataset.py \\
  --nova_root /mnt/workspace/final_proj/Nova \\
  --image_root /mnt/workspace/final_proj/Nova/images \\
  --out_dir /mnt/workspace/final_proj/data/nova_sft
```

> 这些 JSONL 默认被 `.gitignore` 忽略，不会上传 GitHub；如需跨机器使用，请用 `scp` / `rsync`，或临时 `git add -f`。
"""
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    print("=" * 60)
    print("Nova split built")
    print(f"  image_root : {image_root}")
    print(f"  out_dir    : {out_dir}")
    for split in ("train", "val", "test"):
        split_case_count = len({rec["case_id"] for rec in split_records[split]})
        print(
            f"  {split:5s}: {split_case_count:3d} cases, "
            f"{len(split_records[split]):3d} images"
        )
    print(f"  dropped    : {dict(dropped)}")
    print(f"  boxes/img  : mean={manifest['box_stats']['mean']} max={manifest['box_stats']['max']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
