#!/usr/bin/env bash
# Detached start/restart; repeat the same command after a process/server failure.
set -euo pipefail
cd "$(dirname "$0")/.."
TASK_ROOT="$PWD/experiments/grpo_refinement_20261119"
mkdir -p "$TASK_ROOT"
if [[ -f "$TASK_ROOT/launcher.pid" ]]; then
    old_pid="$(cat "$TASK_ROOT/launcher.pid")"
    if kill -0 "$old_pid" 2>/dev/null && [[ "$(ps -p "$old_pid" -o args=)" == *scripts/run_refinement.py* ]]; then
        echo "Already running: PID $old_pid"
        exit 0
    fi
fi
nohup setsid /data/yanfaxiang/venvs/ms-swift/bin/python -u scripts/run_refinement.py "$@" \
    >> "$TASK_ROOT/controller.log" 2>&1 < /dev/null &
runner_pid=$!
echo "$runner_pid" > "$TASK_ROOT/launcher.pid"
echo "Started detached controller: PID $runner_pid"
echo "Log: $TASK_ROOT/controller.log"
