# eval/configs

消融实验用的服务配置。每个文件对应一次实验：同一套 embedding（text-embedding-v4）+ 不同的 `retrieval` 开关，各自使用独立的 SQLite 库（`data/eval-<NAME>.sqlite3`）。

- 文件由 `eval/mkcfg.py NAME key=value ...` 生成（从 `config.local.yaml` 和 `.env` 复制密钥），因此都含明文密钥，已被 `*.local.yaml` 忽略，不要提交。
- `database` 是相对**配置文件所在目录**解析的，所以写成 `../../data/...`。
- 用 `eval/run.sh NAME PORT CONVS PER_CONV`（LoCoMo）或 `eval/runl.sh NAME PORT PER_TYPE`（LongMemEval-S）跑：起服务 → 跑评测 → 关服务。结果在 `eval/res_NAME.json` / `eval/out_NAME.log`。

## 实验一览

| 配置 | 评测集 | 与基线的差异 | 日志里的 ACC |
|---|---|---|---|
| `evalbase` | LoCoMo 3 会话 / 180 题 | 无 `llm`、`rerank` 段，纯混合检索（早期基线） | 0.761 |
| `evalA` | 同上 | 配了 llm/rerank 但开关全关（`retrieval: {}`） | 0.794 |
| `evalB` | 同上 | `add_notes: true` | 0.789 |
| `evalEbase` | LoCoMo 5 会话 / 350 题 | 基线，`retrieval: {}` | 0.823 |
| `evalEfh` | 同上 | `add_notes: true` + `show_notes: false` | 0.840 |
| `evalEfs` | 同上 | `add_notes: true` + `show_notes: true` | 0.840 |
| `evalEqx` | 同上 | `query_expansion: true`（延迟约 1.7s，无增益） | 0.823 |
| `evalErr` | 同上 | `rerank: true` | 0.837 |
| `evalEwide` | 同上 | `candidate_units: 150`（上下文约 12k token，反而下降） | 0.814 |
| `evalF1` | 同上 | `add_notes: true`（与 B、Efs 配置等价，仅库不同） | 0.831 |
| `evalF2` | 同上 | `add_notes` + `rerank`，**复用 F1 的库**（不重新入库） | 0.860 |
| `evalL0` | LongMemEval-S 30 题 | `rerank: true` | 0.567 |
| `evalL1` | LongMemEval-S | `add_notes` + `rerank`；运行时连接被重置，没有产出结果 | — |

ACC 来自各次 `eval/out_*.log`，样本量小、由 LLM 评判，差异在 ±0.02 内基本是噪声。
