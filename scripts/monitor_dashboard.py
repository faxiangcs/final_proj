#!/usr/bin/env python3
"""Nova experiment dashboard: Qwen baseline progress, GPUs, training runs.

Stdlib only. Binds to 127.0.0.1 by default.

    python3 scripts/monitor_dashboard.py --port 8777

Open http://127.0.0.1:8777/ (remote machine: ssh -L 8777:127.0.0.1:8777 ...).
`--once` prints the JSON snapshot and exits, useful for scripts/tests.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]

BASELINES = {
    "image": {
        "label": "图像 baseline",
        "path": ROOT / "result/qwen35_baseline/pred.jsonl",
        "log": ROOT / "output/qwen35_baseline_image_v2.log",
    },
    "text": {
        "label": "纯文本 baseline",
        "path": ROOT / "result/qwen35_baseline_text/pred.jsonl",
        "log": ROOT / "output/qwen35_baseline_text_v2.log",
    },
    "image_fmt": {
        "label": "图像 baseline · 格式引导",
        "path": ROOT / "result/qwen35_baseline_fmt/pred.jsonl",
        "log": ROOT / "output/qwen35_baseline_fmt_image_v2.log",
    },
    "text_fmt": {
        "label": "纯文本 baseline · 格式引导",
        "path": ROOT / "result/qwen35_baseline_text_fmt/pred.jsonl",
        "log": ROOT / "output/qwen35_baseline_text_fmt_v2.log",
    },
}

FINAL_RE = re.compile(
    r"最终(?:的)?(?:临床)?诊断[^\n：:]{0,12}[：:]\s*\**\s*([^\n。]{1,80})"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        return []
    return rows


def _tail_lines(path: Path, limit: int = 6) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    cleaned: list[str] = []
    for line in lines[-limit:]:
        line = ANSI_RE.sub("", line).replace("\r", " ")
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            cleaned.append(line[:220])
    return cleaned


def _final_snippet(row: dict[str, Any]) -> str:
    text = (row.get("response") or "").strip()
    if not text:
        return ""
    match = FINAL_RE.search(text)
    if match:
        return match.group(1).strip().strip("*").strip()
    for line in reversed(text.splitlines()):
        line = line.strip().strip("#* ").strip()
        if line:
            return line[:80]
    return ""


def _baseline_stats(mode: str, total: int) -> dict[str, Any]:
    spec = BASELINES[mode]
    rows = _read_jsonl(spec["path"])
    latencies: list[float] = []
    tokens = 0
    failed = empty = 0
    for row in rows:
        try:
            latencies.append(float(row.get("latency") or 0.0))
        except (TypeError, ValueError):
            pass
        usage = row.get("usage") or {}
        try:
            tokens += int(usage.get("completion_tokens") or 0)
        except (TypeError, ValueError):
            pass
        if row.get("error"):
            failed += 1
        elif not (row.get("response") or "").strip():
            empty += 1

    updated = ""
    if spec["path"].exists():
        updated = datetime.fromtimestamp(spec["path"].stat().st_mtime).strftime(
            "%H:%M:%S"
        )

    if rows:
        if len(rows) >= total:
            state = "done"
        elif time.time() - spec["path"].stat().st_mtime < 300:
            state = "running"
        else:
            state = "stalled"
    else:
        state = "pending"

    recent = []
    for row in rows[-6:]:
        usage = row.get("usage") or {}
        status = "ok"
        if row.get("error"):
            status = "error"
        elif not (row.get("response") or "").strip():
            status = "empty"
        recent.append(
            {
                "mode": spec["label"],
                "index": row.get("index"),
                "latency": row.get("latency"),
                "tokens": usage.get("completion_tokens"),
                "final": _final_snippet(row),
                "status": status,
            }
        )

    elapsed = sum(latencies) or 0.0
    return {
        "mode": mode,
        "label": spec["label"],
        "done": len(rows),
        "total": total,
        "failed": failed,
        "empty": empty,
        "avg_latency": round(sum(latencies) / len(latencies), 1) if latencies else 0,
        "tok_per_s": round(tokens / elapsed, 1) if elapsed else 0,
        "updated": updated,
        "state": state,
        "recent": recent,
        "log": _tail_lines(spec["log"]),
    }


def _gpu_stats() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=4, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "mem_used": int(float(parts[2])),
                    "mem_total": int(float(parts[3])),
                    "util": int(float(parts[4])),
                }
            )
        except ValueError:
            continue
    return gpus


def _vllm_status() -> dict[str, Any]:
    result: dict[str, Any] = {"up": False, "model": ""}
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:8000/v1/models", timeout=1.5
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        result["up"] = True
        data = payload.get("data") or []
        if data:
            result["model"] = data[0].get("id", "")
    except Exception:
        pass
    return result


def _training_runs(limit: int = 4) -> list[dict[str, Any]]:
    logs = sorted(
        ROOT.glob("output/**/logging.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:limit]
    runs = []
    for path in logs:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-120:]
        except OSError:
            continue
        loss = None
        step = ""
        eval_loss = None
        eval_step = ""
        for line in lines:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "loss" in obj:
                loss = obj.get("loss")
                step = obj.get("global_step/max_steps") or obj.get("global_step") or step
            if "eval_loss" in obj:
                eval_loss = obj.get("eval_loss")
                eval_step = obj.get("global_step/max_steps") or obj.get("step") or ""
                step = obj.get("global_step/max_steps") or step
        args_path = path.parent / "args.json"
        model = ""
        dataset = ""
        if args_path.exists():
            try:
                args = json.loads(args_path.read_text(encoding="utf-8"))
                model = Path(str(args.get("model", ""))).name
                datasets = args.get("dataset") or []
                dataset = Path(str(datasets[0])).name if datasets else ""
            except (OSError, json.JSONDecodeError):
                pass
        runs.append(
            {
                "run": str(path.parent.relative_to(ROOT)),
                "step": step,
                "loss": round(loss, 4) if isinstance(loss, (int, float)) else None,
                "eval_loss": round(eval_loss, 4)
                if isinstance(eval_loss, (int, float))
                else None,
                "eval_step": eval_step,
                "model": model,
                "dataset": dataset,
                "updated": datetime.fromtimestamp(path.stat().st_mtime).strftime(
                    "%m-%d %H:%M"
                ),
            }
        )
    return runs


def _test_total() -> int:
    path = ROOT / "data/nova_sft/test.jsonl"
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def collect() -> dict[str, Any]:
    total = _test_total()
    baselines = {mode: _baseline_stats(mode, total) for mode in BASELINES}
    recent = []
    for mode in BASELINES:
        recent.extend(baselines[mode]["recent"])
    recent.sort(key=lambda item: (item["index"] if item["index"] is not None else -1))
    return {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epoch": int(time.time()),
        "test_total": total,
        "vllm": _vllm_status(),
        "gpus": _gpu_stats(),
        "baselines": baselines,
        "recent": recent[-12:],
        "trainings": _training_runs(),
    }


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nova 实验监控</title>
<style>
:root{
  --bg:#0e1116; --surface:#161b23; --surface2:#1c222c; --border:#2a323e;
  --ink:#e7ebf2; --muted:#8b96a7; --accent:#5aa9ff; --accent-soft:#1d3a5c;
  --good:#3ddc97; --warn:#f2b544; --bad:#ff6b6b; --grid:#232b36;
}
@media (prefers-color-scheme: light){
  :root{
    --bg:#f3f5f8; --surface:#ffffff; --surface2:#f7f9fc; --border:#d9e0ea;
    --ink:#1a2230; --muted:#5c687a; --accent:#1f6fd0; --accent-soft:#e3eefb;
    --good:#12855b; --warn:#9a6200; --bad:#c62828; --grid:#e4e9f1;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",sans-serif}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:baseline;
  padding:18px 24px 6px}
h1{font-size:18px;margin:0;font-weight:650;letter-spacing:.2px}
.sub{color:var(--muted);font-size:12.5px}
main{padding:10px 24px 32px;max-width:1280px;margin:0 auto}
.grid{display:grid;gap:14px}
.kpis{grid-template-columns:repeat(auto-fit,minmax(180px,1fr));margin:12px 0 16px}
@media(max-width:900px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 16px}
.kpi .label{color:var(--muted);font-size:12px;letter-spacing:.3px;text-transform:uppercase}
.kpi .value{font-size:26px;font-weight:650;margin-top:4px;
  font-variant-numeric:tabular-nums}
.kpi .value small{font-size:13px;color:var(--muted);font-weight:500}
.kpi .foot{color:var(--muted);font-size:12px;margin-top:4px}
.pill{display:inline-flex;gap:6px;align-items:center;border-radius:999px;
  padding:2px 9px;font-size:12px;font-weight:600;border:1px solid var(--border)}
.pill.ok{color:var(--good);background:color-mix(in srgb,var(--good) 12%,transparent)}
.pill.warn{color:var(--warn);background:color-mix(in srgb,var(--warn) 12%,transparent)}
.pill.bad{color:var(--bad);background:color-mix(in srgb,var(--bad) 12%,transparent)}
.pill.idle{color:var(--muted);background:var(--surface2)}
.bar{height:8px;background:var(--grid);border-radius:4px;overflow:hidden;margin:8px 0 4px}
.bar > span{display:block;height:100%;background:var(--accent);border-radius:4px;
  transition:width .4s ease}
.bar.warn > span{background:var(--warn)}
.bar.bad > span{background:var(--bad)}
.bar.good > span{background:var(--good)}
.row{display:flex;align-items:center;justify-content:space-between;gap:10px}
h2{font-size:13px;letter-spacing:.4px;text-transform:uppercase;color:var(--muted);
  margin:18px 0 8px;font-weight:650}
.gpus{grid-template-columns:repeat(4,minmax(0,1fr))}
@media(max-width:900px){.gpus{grid-template-columns:repeat(2,minmax(0,1fr))}}
.gpu .name{font-size:12px;color:var(--muted);white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis}
.gpu .num{font-variant-numeric:tabular-nums}
table{width:100%;border-collapse:collapse;font-size:13px}
th{color:var(--muted);font-weight:600;text-align:left;padding:8px 10px;
  border-bottom:1px solid var(--border);font-size:12px}
td{padding:8px 10px;border-bottom:1px solid var(--grid);vertical-align:top}
tr:last-child td{border-bottom:0}
td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
.snippet{max-width:520px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
pre{background:var(--surface2);border:1px solid var(--border);border-radius:8px;
  padding:10px 12px;margin:6px 0;max-height:190px;overflow:auto;font-size:12px;
  color:var(--muted);white-space:pre-wrap;word-break:break-word}
.two{grid-template-columns:minmax(0,1.15fr) minmax(0,1fr)}
@media(max-width:900px){.two{grid-template-columns:1fr}}
.err{display:none;background:color-mix(in srgb,var(--bad) 15%,transparent);
  color:var(--bad);border:1px solid var(--bad);padding:8px 12px;border-radius:8px;
  margin:10px 0;font-size:13px}
.muted{color:var(--muted)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.footer{color:var(--muted);font-size:12px;margin-top:18px}
.empty{padding:14px;color:var(--muted);text-align:center}
</style>
</head>
<body>
<header>
  <h1>Nova 实验监控</h1>
  <span class="sub" id="clock">--</span>
  <span class="sub" id="refresh">每 5s 刷新</span>
</header>
<main>
  <div class="err" id="err">无法连接监控服务，正在重试…</div>
  <section class="grid kpis" id="kpis"></section>

  <h2>Baseline 进度</h2>
  <section class="grid two">
    <div class="card" id="baseline-card"></div>
    <div class="card"><div class="row"><h2 style="margin:0">GPU</h2>
      <span class="muted" id="gpu-time"></span></div>
      <section class="grid gpus" id="gpus"></section></div>
  </section>

  <h2>最近输出</h2>
  <section class="card"><table>
    <thead><tr><th>阶段</th><th>#</th><th>延迟</th><th>tokens</th>
      <th>最终诊断</th><th>状态</th></tr></thead>
    <tbody id="recent"></tbody>
  </table></section>

  <h2>训练任务</h2>
  <section class="card" id="trainings"></section>

  <h2>日志尾部</h2>
  <section class="grid two" id="logs"></section>

  <p class="footer" id="footer"></p>
</main>
<script>
const $ = (id) => document.getElementById(id);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g,
  (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}[c]));
const fmt = (v, d=1) => (v === null || v === undefined || v === "")
  ? "—" : Number(v).toFixed(d);
const statePill = (state) => {
  const map = {running:["ok","运行中"], done:["good","已完成"],
    stalled:["warn","停滞"], pending:["idle","等待中"],
    error:["bad","错误"], empty:["warn","空输出"], ok:["good","正常"]};
  const [cls, text] = map[state] || map.pending;
  return `<span class="pill ${cls}">${text}</span>`;
};
const pct = (a, b) => b > 0 ? Math.min(100, a / b * 100) : 0;

function render(data){
  $("clock").textContent = `更新 ${data.time}`;
  const runs = Object.values(data.baselines);
  const v = data.vllm;
  const bars = {done:"good", running:"", stalled:"warn", pending:""};
  $("kpis").innerHTML = runs.map(b =>
    `<div class="card kpi"><div class="label">${esc(b.label)}</div>
      <div class="value">${b.done}<small> / ${b.total}</small></div>
      <div class="bar ${bars[b.state] ?? ""}"><span style="width:${pct(b.done, b.total)}%"></span></div>
      <div class="foot">${statePill(b.state)} 失败 ${b.failed} · 空 ${b.empty}</div></div>`
  ).join("") + [
    `<div class="card kpi"><div class="label">vLLM 服务</div>
      <div class="value" style="font-size:20px">${v.up
        ? `<span class="pill ok">在线</span>` : `<span class="pill bad">离线</span>`}</div>
      <div class="foot mono">${esc(v.model || "127.0.0.1:8000")}</div></div>`,
    `<div class="card kpi"><div class="label">GPU 0</div>
      <div class="value">${data.gpus[0] ? data.gpus[0].util + "<small>%</small>" : "—"}</div>
      <div class="foot">${data.gpus[0]
        ? `${(data.gpus[0].mem_used/1024).toFixed(1)} / ${(data.gpus[0].mem_total/1024).toFixed(0)} GB`
        : "nvidia-smi 不可用"}</div></div>`
  ].join("");

  $("baseline-card").innerHTML = runs.map(b => `
    <div class="row"><h2 style="margin:0">${esc(b.label)}</h2>
      <span class="muted">${b.updated ? "更新 " + b.updated : "尚未开始"}</span></div>
    <div class="bar ${bars[b.state] ?? ""}"><span style="width:${pct(b.done, b.total)}%"></span></div>
    <div class="row"><span class="muted">${b.done}/${b.total} · ${statePill(b.state)}</span>
      <span class="mono">${fmt(b.avg_latency)} s/条 · ${fmt(b.tok_per_s)} tok/s</span></div>`
  ).join("");

  $("gpus").innerHTML = data.gpus.map(g => {
    const memPct = pct(g.mem_used, g.mem_total);
    const cls = g.util >= 90 ? "warn" : "";
    return `<div class="card gpu"><div class="name" title="${esc(g.name)}">GPU ${g.index} · ${esc(g.name)}</div>
      <div class="row"><span class="num">${(g.mem_used/1024).toFixed(1)} / ${(g.mem_total/1024).toFixed(0)} GB</span>
        <span class="muted num">${memPct.toFixed(0)}%</span></div>
      <div class="bar"><span style="width:${memPct}%"></span></div>
      <div class="row"><span class="muted">利用率</span>
        <span class="num">${g.util}%</span></div>
      <div class="bar ${cls}"><span style="width:${Math.min(100,g.util)}%"></span></div></div>`;
  }).join("") || `<div class="empty">无 GPU 数据</div>`;

  $("recent").innerHTML = data.recent.length ? data.recent.map(r =>
    `<tr><td>${esc(r.mode)}</td><td class="num">${r.index ?? "—"}</td>
     <td class="num">${fmt(r.latency)}s</td><td class="num">${r.tokens ?? "—"}</td>
     <td class="snippet" title="${esc(r.final)}">${esc(r.final) || "<span class='muted'>未解析</span>"}</td>
     <td>${statePill(r.status)}</td></tr>`).join("")
    : `<tr><td colspan="6" class="empty">暂无输出</td></tr>`;

  $("trainings").innerHTML = data.trainings.length ? data.trainings.map(t =>
    `<div class="row" style="padding:8px 0;border-bottom:1px solid var(--grid)">
      <div><div class="mono">${esc(t.run)}</div>
        <div class="muted">${esc(t.model)} · ${esc(t.dataset)}</div></div>
      <div style="text-align:right">
        <div class="mono">step ${esc(t.step)} · loss ${t.loss ?? "—"}</div>
        <div class="muted">eval ${t.eval_loss ?? "—"} · ${esc(t.updated)}</div></div>
     </div>`).join("")
    : `<div class="empty">还没有训练日志</div>`;

  $("logs").innerHTML = Object.values(data.baselines).map(b =>
    `<div class="card"><div class="row"><strong>${b.label}</strong>
      ${statePill(b.state)}</div>
      <pre>${b.log.length ? esc(b.log.join("\\n")) : "暂无日志"}</pre></div>`).join("");

  $("footer").textContent =
    `测试集 ${data.test_total} 条 · 数据：result/*/pred.jsonl、output/*.log、nvidia-smi`;
  $("err").style.display = "none";
}

async function refresh(){
  try{
    const res = await fetch("/api/status", {cache:"no-store"});
    if(!res.ok) throw new Error(res.status);
    render(await res.json());
  }catch(e){
    $("err").style.display = "block";
  }finally{
    setTimeout(refresh, 5000);
  }
}
refresh();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "NovaMonitor/1.0"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/status":
            payload = json.dumps(collect(), ensure_ascii=False).encode("utf-8")
            self._send(200, payload, "application/json; charset=utf-8")
        elif path == "/healthz":
            self._send(200, b"ok", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")


def main() -> int:
    parser = argparse.ArgumentParser(description="Nova monitoring dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--once", action="store_true", help="print JSON and exit")
    args = parser.parse_args()

    if args.once:
        print(json.dumps(collect(), ensure_ascii=False, indent=2))
        return 0

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Nova dashboard: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
