"""Qwen3.5-9B baseline inference against an OpenAI-compatible endpoint.

只依赖标准库，避免在训练环境里再装 openai。输出格式与 ``swift infer`` 对齐，
可直接交给 ``src/eval_nova.py``：

    python3 src/infer_qwen_baseline.py \
        --data data/nova_sft/test.jsonl \
        --out result/qwen35_baseline/pred.jsonl

纯文本对照（不传图，prompt 保持一致）：

    python3 src/infer_qwen_baseline.py --text_only \
        --data data/nova_sft/test.jsonl \
        --out result/qwen35_baseline_text/pred.jsonl

注意：服务端必须以多模态方式启动（不要带 ``--language-model-only``），
否则传图请求会失败或图片被忽略。
"""

from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


FORMAT_HINT = (
    "你是神经影像AI助手。请严格按以下格式输出；固定标题保持中文，正文内容"
    "（图像描述、鉴别诊断、最终诊断）一律使用英文；不要使用 markdown 标题、"
    "不要编号、不要输出额外解释：\n"
    "【异常定位】：[x_min, y_min, x_max, y_max]\n"
    "【图像描述】：<English description>\n"
    "【诊断推理】：考虑鉴别诊断包括 a; b; c。结合特征，最终诊断为："
    "<English diagnosis>。\n"
    "坐标必须是 0–480 像素；鉴别诊断最多 4 个，用分号分隔，不要编号。"
)


def encode_image(path: str) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("ascii")


def build_content(prompt: str, images: list[str], text_only: bool) -> Any:
    if text_only or not images:
        return prompt
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_path in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encode_image(image_path)}"},
            }
        )
    return content


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def read_rows(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3.5-9B Nova baseline")
    parser.add_argument("--data", default="data/nova_sft/test.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--text_only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--no_thinking",
        action="store_true",
        help="send chat_template_kwargs.enable_thinking=false",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip indices already present in --out",
    )
    parser.add_argument(
        "--format_hint",
        action="store_true",
        help="prepend a system prompt requiring the canonical three sections "
        "and an English final diagnosis (for a comparable zero-shot baseline)",
    )
    args = parser.parse_args()

    rows = read_rows(args.data)
    if args.limit:
        rows = rows[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[int] = set()
    if args.resume and out_path.exists():
        for existing in read_rows(str(out_path)):
            done.add(int(existing.get("index", -1)))

    mode = "w" if not args.resume else "a"
    url = args.base_url.rstrip("/") + "/chat/completions"
    ok = failed = 0

    with open(out_path, mode, encoding="utf-8") as out:
        for index, row in enumerate(rows):
            if index in done:
                continue
            messages = row.get("messages") or []
            user = next((m for m in messages if m.get("role") == "user"), {})
            prompt = user.get("content", "")
            # ms-swift 4.x puts images at row level; fall back to the old
            # message-level key for datasets built before the schema fix.
            images = row.get("images") or user.get("image") or []
            if isinstance(images, str):
                images = [images]
            label = ""
            for message in reversed(messages):
                if message.get("role") == "assistant":
                    label = message.get("content", "")
                    break

            chat_messages: list[dict[str, Any]] = []
            if args.format_hint:
                chat_messages.append({"role": "system", "content": FORMAT_HINT})
            chat_messages.append(
                {
                    "role": "user",
                    "content": build_content(prompt, images, args.text_only),
                }
            )
            payload: dict[str, Any] = {
                "model": args.model,
                "messages": chat_messages,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            }
            if args.no_thinking:
                payload["chat_template_kwargs"] = {"enable_thinking": False}

            started = time.time()
            try:
                response = post_json(url, payload, args.timeout)
                content = response["choices"][0]["message"]["content"]
                usage = response.get("usage", {})
                ok += 1
                record: dict[str, Any] = {
                    "index": index,
                    "response": content,
                    "labels": label,
                    "messages": messages,
                    "model": args.model,
                    "text_only": bool(args.text_only),
                    "latency": round(time.time() - started, 3),
                    "usage": usage,
                }
            except (urllib.error.URLError, TimeoutError, KeyError) as exc:
                failed += 1
                record = {
                    "index": index,
                    "response": "",
                    "labels": label,
                    "messages": messages,
                    "model": args.model,
                    "text_only": bool(args.text_only),
                    "error": str(exc),
                }
                print(f"[{index}] ERROR: {exc}")
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            if (index + 1) % 10 == 0:
                print(f"progress {index + 1}/{len(rows)} ok={ok} failed={failed}")

    print(f"done: ok={ok} failed={failed} -> {out_path}")
    if args.text_only:
        print("warning: text-only baseline, do not compare its localization IoU with image models")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
