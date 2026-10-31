judge: qwen-local   reference: v6-rule-notes

| experiment | LoCoMo | LongMemEval | all | delta | win/loss | z | avg tokens |
|---|---|---|---|---|---|---|---|
| v6-rule-notes | 0.755 (200) | 0.639 (36) | 0.737 | +0.000 | 0/0 | +0.0 | 16399 |
| v12-wide-time | 0.770 (200) | 0.639 (36) | 0.750 | +0.013 | 7/4 | +0.9 | 16354 |
| v13-gated | 0.775 (200) | 0.639 (36) | 0.754 | +0.017 | 7/3 | +1.3 | 16373 |

Per category: accuracy (delta vs reference, wins/losses)

| experiment | locomo inference | locomo open | locomo single-hop | locomo temporal | lme knowledge-update | lme multi-session | lme single-session-assistant | lme single-session-preference | lme single-session-user | lme temporal-reasoning |
|---|---|---|---|---|---|---|---|---|---|---|
| v6-rule-notes | 0.70 (+0.00, +0/-0) | 0.82 (+0.00, +0/-0) | 0.48 (+0.00, +0/-0) | 0.78 (+0.00, +0/-0) | 1.00 (+0.00, +0/-0) | 0.67 (+0.00, +0/-0) | 1.00 (+0.00, +0/-0) | 0.17 (+0.00, +0/-0) | 0.67 (+0.00, +0/-0) | 0.33 (+0.00, +0/-0) |
| v12-wide-time | 0.70 (+0.00, +0/-0) | 0.82 (+0.00, +2/-2) | 0.45 (-0.03, +0/-1) | 0.86 (+0.08, +4/-0) | 0.83 (-0.17, +0/-1) | 0.67 (+0.00, +0/-0) | 1.00 (+0.00, +0/-0) | 0.17 (+0.00, +0/-0) | 0.67 (+0.00, +0/-0) | 0.50 (+0.17, +1/-0) |
| v13-gated | 0.70 (+0.00, +0/-0) | 0.83 (+0.01, +2/-1) | 0.45 (-0.03, +0/-1) | 0.86 (+0.08, +4/-0) | 0.83 (-0.17, +0/-1) | 0.67 (+0.00, +0/-0) | 1.00 (+0.00, +0/-0) | 0.17 (+0.00, +0/-0) | 0.67 (+0.00, +0/-0) | 0.50 (+0.17, +1/-0) |
