judge: qwen-local-holdout   reference: v4-window-rerank

| experiment | LoCoMo | LongMemEval | all | delta | win/loss | z | avg tokens |
|---|---|---|---|---|---|---|---|
| v4-window-rerank | 0.765 (200) | 0.667 (36) | 0.750 | +0.000 | 0/0 | +0.0 | 16011 |
| v11-all-features | - | - | - | - | - | - | - |

Per category: accuracy (delta vs reference, wins/losses)

| experiment | locomo inference | locomo open | locomo single-hop | locomo temporal | lme knowledge-update | lme multi-session | lme single-session-assistant | lme single-session-preference | lme single-session-user | lme temporal-reasoning |
|---|---|---|---|---|---|---|---|---|---|---|
| v4-window-rerank | 0.75 (+0.00, +0/-0) | 0.88 (+0.00, +0/-0) | 0.53 (+0.00, +0/-0) | 0.62 (+0.00, +0/-0) | 1.00 (+0.00, +0/-0) | 0.67 (+0.00, +0/-0) | 1.00 (+0.00, +0/-0) | 0.00 (+0.00, +0/-0) | 0.83 (+0.00, +0/-0) | 0.50 (+0.00, +0/-0) |
| v11-all-features | - | - | - | - | - | - | - | - | - | - |
