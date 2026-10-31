#!/bin/bash
set -eo pipefail
cd "${PLAIN_MEMORY_HOME:-$HOME/plain-memory}"
service_dir="$PWD"
runtime="$service_dir/.venv/bin/python"
if [ -L current ]; then
    runtime="$service_dir/current/.venv/bin/python"
fi
test -x "$runtime"

if [ -f run.pid ]; then
    read -r old_pid < run.pid
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
        # A stale PID file must never stop an unrelated process.
        if [ "$(readlink -f "/proc/$old_pid/cwd")" != "$service_dir" ] ||
           ! tr '\0' ' ' < "/proc/$old_pid/cmdline" | grep -Eq 'plain_memory|plain-memory'; then
            echo "PID file does not identify this memory service; refusing to stop it." >&2
            exit 1
        fi
        kill "$old_pid"
        for attempt in {1..60}; do
            kill -0 "$old_pid" 2>/dev/null || break
            sleep 0.5
        done
        if kill -0 "$old_pid" 2>/dev/null; then
            echo "Previous memory service has not stopped." >&2
            exit 1
        fi
    fi
fi

set -a
. "$HOME/server.env"
set +a
setsid nohup "$runtime" -m plain_memory --config "$service_dir/config.local.yaml" \
    serve --host 0.0.0.0 --port 8600 >> "$service_dir/serve.log" 2>&1 < /dev/null &
echo $! > "$service_dir/run.pid"
