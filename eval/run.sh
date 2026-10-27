#!/bin/bash
# usage: eval/run.sh NAME PORT CONVS PER_CONV [extra harness args]
# Starts a server for config.evalNAME.local.yaml, runs the LoCoMo harness, stops the server.
cd "$(dirname "$0")/.."
NAME=$1; PORT=$2; CONVS=$3; PER=$4; shift 4
set -a; . <(grep -E '^[A-Z_]+_API_KEY=' .env); set +a
TOKEN=$(.venv/Scripts/python -c "import yaml;print(yaml.safe_load(open('config.eval$NAME.local.yaml'))['api_token'])")
.venv/Scripts/plain-memory --config config.eval$NAME.local.yaml serve --port $PORT > eval/server-$NAME.log 2>&1 &
PID=$!
sleep 6
.venv/Scripts/python eval/locomo.py --base-url http://127.0.0.1:$PORT --token $TOKEN --convs $CONVS --per-conv $PER --run ${RUNID:-$NAME} --workers 12 --out eval/res_$NAME.json "$@" > eval/out_$NAME.log 2>&1
kill $PID 2>/dev/null; taskkill //F //PID $PID >/dev/null 2>&1
