#!/bin/bash
# Overnight ablation: reference (deployed v4) vs the deterministic features, judged by the free Qwen.
# Run from the project root:   bash eval/overnight.sh
# Finished experiments are skipped, so it is safe to re-run after an interruption.
# Progress bars appear on screen; the final tables are printed at the end and saved to
# eval/results/summary_qwen-local.md and summary_qwen-local-holdout.md.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
JUDGE=qwen-local
DEV="--locomo-per-conv 40 --lme-per-type 6"

echo "=== 1/3 development set (LoCoMo 40/conv x5, LongMemEval 6/type x6) ==="
$PY eval/runner.py v4-window-rerank v6-rule-notes v7-date-boost v8-current-order v9-profile \
    v10-overview v11-all-features --judge $JUDGE $DEV

echo "=== 2/3 held-out confirmation (other conversations, other LongMemEval histories) ==="
$PY eval/runner.py v4-window-rerank v11-all-features --judge $JUDGE $DEV --tag holdout \
    --convs conv-30,conv-44,conv-47,conv-49,conv-50 --seed 7 --lme-seed 7

echo "=== 3/3 results ==="
$PY eval/compare.py v4-window-rerank v6-rule-notes v7-date-boost v8-current-order v9-profile \
    v10-overview v11-all-features --judge $JUDGE
$PY eval/compare.py v4-window-rerank v11-all-features --judge $JUDGE --tag holdout
