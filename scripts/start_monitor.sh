#!/usr/bin/env bash
# Start/stop/status helper for the Nova monitoring dashboard.
#
#   bash scripts/start_monitor.sh start
#   bash scripts/start_monitor.sh status
#   bash scripts/start_monitor.sh restart
#   bash scripts/start_monitor.sh stop
#
# Environment:
#   MONITOR_HOST  bind address (default 127.0.0.1; use 0.0.0.0 for LAN access)
#   MONITOR_PORT  port (default 8777)
#   PID_FILE      default <repo>/output/monitor_dashboard.pid
#   LOG_FILE      default <repo>/output/monitor_dashboard.log
#   PYTHON        default python3
#
# The dashboard is read-only but has no authentication and may show clinical
# text from logs/predictions. Prefer the SSH tunnel printed by start/status
# over exposing it directly to a shared or public network.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="${PID_FILE:-$ROOT_DIR/output/monitor_dashboard.pid}"
LOG_FILE="${LOG_FILE:-$ROOT_DIR/output/monitor_dashboard.log}"
HOST_SET=0
PORT_SET=0
[[ -n "${MONITOR_HOST:-}" ]] && HOST_SET=1
[[ -n "${MONITOR_PORT:-}" ]] && PORT_SET=1
HOST="${MONITOR_HOST:-127.0.0.1}"
PORT="${MONITOR_PORT:-8777}"
PYTHON="${PYTHON:-python3}"

usage() {
    cat <<'EOF'
usage: bash scripts/start_monitor.sh {start|stop|restart|status|health}

environment overrides:
  MONITOR_HOST  bind address (default 127.0.0.1; 0.0.0.0 for LAN access)
  MONITOR_PORT  port (default 8777)
  PID_FILE, LOG_FILE, PYTHON

examples:
  bash scripts/start_monitor.sh start
  MONITOR_HOST=0.0.0.0 bash scripts/start_monitor.sh restart
  bash scripts/start_monitor.sh status
EOF
}

is_running() {
    [[ -f "$PID_FILE" ]] || return 1
    local pid cmdline
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    [[ -n "$pid" ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
    [[ "$cmdline" == *monitor_dashboard.py* ]]
}

load_running_meta() {
    # Read the actual --host/--port from the running process so status/start
    # print the right URLs even when the caller did not repeat the env vars.
    local pid args i
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    [[ -n "$pid" ]] || return 0
    mapfile -t args < <(tr '\0' '\n' <"/proc/$pid/cmdline" 2>/dev/null || true)
    for ((i = 0; i < ${#args[@]}; i++)); do
        case "${args[$i]}" in
            --host) HOST="${args[$((i + 1))]:-$HOST}" ;;
            --port) PORT="${args[$((i + 1))]:-$PORT}" ;;
        esac
    done
}

port_in_use() {
    ss -ltn 2>/dev/null | awk -v p=":$PORT" '$4 ~ p"$" {found=1} END {exit !found}'
}

health() {
    local url="http://127.0.0.1:$PORT/healthz"
    if [[ "$HOST" != "127.0.0.1" && "$HOST" != "0.0.0.0" && "$HOST" != "localhost" ]]; then
        url="http://$HOST:$PORT/healthz"
    fi
    curl --noproxy '*' -fsS --max-time 3 "$url" >/dev/null 2>&1
}

server_ip() {
    hostname -I 2>/dev/null | awk '{print $1}'
}

print_urls() {
    echo "local:  http://127.0.0.1:$PORT/"
    if [[ "$HOST" == "0.0.0.0" ]]; then
        echo "LAN:    http://$(server_ip):$PORT/  (firewall/VPN must allow port $PORT)"
    elif [[ "$HOST" != "127.0.0.1" && "$HOST" != "localhost" ]]; then
        echo "LAN:    http://$HOST:$PORT/"
    fi
    echo "tunnel: ssh -N -L $PORT:127.0.0.1:$PORT ${USER:-user}@$(server_ip)"
}

start() {
    if is_running; then
        load_running_meta
        echo "already running (pid $(cat "$PID_FILE"))"
        print_urls
        return 0
    fi
    if port_in_use; then
        echo "[ERROR] port $PORT is already in use by another process." >&2
        ss -ltnp 2>/dev/null | grep ":$PORT " >&2 || true
        return 1
    fi
    mkdir -p "$(dirname "$PID_FILE")" "$(dirname "$LOG_FILE")"
    rm -f "$PID_FILE"
    echo "--- $(date '+%F %T') start host=$HOST port=$PORT ---" >>"$LOG_FILE"
    nohup "$PYTHON" "$ROOT_DIR/scripts/monitor_dashboard.py" --host "$HOST" --port "$PORT" >>"$LOG_FILE" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_FILE"
    for _ in $(seq 1 40); do
        if health; then
            echo "started (pid $pid)"
            print_urls
            return 0
        fi
        sleep 0.25
    done
    echo "[ERROR] dashboard did not become healthy; see $LOG_FILE" >&2
    tail -20 "$LOG_FILE" >&2 || true
    return 1
}

stop() {
    if ! is_running; then
        echo "not running"
        rm -f "$PID_FILE"
        return 0
    fi
    local pid
    pid="$(cat "$PID_FILE")"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.25
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "force killing $pid" >&2
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
    echo "stopped (pid $pid)"
}

restart() {
    # Preserve the running bind/port unless the caller explicitly overrides it.
    local req_host="$HOST" req_port="$PORT"
    if is_running; then
        load_running_meta
        [[ "$HOST_SET" == "1" ]] && HOST="$req_host"
        [[ "$PORT_SET" == "1" ]] && PORT="$req_port"
        stop
    fi
    start
}

status() {
    if is_running; then
        load_running_meta
        echo "running (pid $(cat "$PID_FILE"), host=$HOST, port=$PORT)"
        if health; then echo "health: ok"; else echo "health: FAILED"; fi
    else
        echo "not running"
    fi
    if port_in_use; then
        ss -ltnp 2>/dev/null | grep ":$PORT " || true
    fi
    print_urls
}

cmd="${1:-status}"
shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="$2"; HOST_SET=1; shift 2 ;;
        --port) PORT="$2"; PORT_SET=1; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$cmd" in
    start) start ;;
    stop) stop ;;
    restart) restart ;;
    status) status ;;
    health) if is_running; then load_running_meta; fi; health && echo ok || { echo failed; exit 1; } ;;
    -h|--help|help) usage ;;
    *) echo "unknown command: $cmd" >&2; usage >&2; exit 2 ;;
esac
