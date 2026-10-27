#!/bin/bash
# usage: eval/runl.sh NAME PORT PER_TYPE [extra args]   (LongMemEval-S harness)
cd "$(dirname "$0")/.."
NAME=$1; PORT=$2; PER=$3; shift 3
set -a; . <(grep -E '^[A-Z_]+_API_KEY=' .env); set +a
TOKEN=$(.venv/Scripts/python -c "import yaml;print(yaml.safe_load(open('config.eval$NAME.local.yaml'))['api_token'])")
.venv/Scripts/plain-memory --config config.eval$NAME.local.yaml serve --port $PORT > eval/server-$NAME.log 2>&1 &
PID=$!
sleep 6
.venv/Scripts/python eval/lme.py --base-url http://127.0.0.1:$PORT --token $TOKEN --per-type $PER --run ${RUNID:-$NAME} --out eval/res_$NAME.json "$@" > eval/out_$NAME.log 2>&1
kill $PID 2>/dev/null; taskkill //F //PID $PID >/dev/null 2>&1
