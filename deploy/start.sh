#!/bin/bash
cd ~/plain-memory
if [ -f run.pid ] && kill -0 $(cat run.pid) 2>/dev/null; then kill $(cat run.pid); sleep 2; fi
set -a; . ~/server.env; set +a
setsid nohup .venv/bin/plain-memory --config config.local.yaml serve --host 0.0.0.0 --port 8600 > serve.log 2>&1 < /dev/null &
echo $! > run.pid
