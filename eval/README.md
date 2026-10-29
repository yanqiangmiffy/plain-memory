# 消融实验（eval/）

所有实验都在本机前台运行，进度用 tqdm 显示。回答和评分用 `judges/` 里选的模型，和平台的官方回答、评分模型不同，所以**只看版本之间的相对差异**，不要把绝对分数当成线上分数。

## 目录

```
eval/
├── experiments/            每个版本一个文件夹，一个文件夹 = 一组配置
│   ├── v1-msg-only/        exp.yaml（配置）+ run.py（启动入口）
│   ├── v2-ctx-embed/
│   ├── v3-window/
│   ├── v4-window-rerank/   ← 线上 0.2.yq 的配置
│   ├── v4a-chrono/ … v4g-rr100/   在 v4 基础上每次只改一个检索开关（已测：都在噪声内）
│   ├── v6-rule-notes/      v4 + 规则版相对时间标注（不用 LLM）
│   ├── v7 … v10            v6 + 日期加权 / 当前状态优先 / 个人画像摘要 / 全局概览，各加一项
│   ├── v11-all-features/   v6 + 以上全部
│   └── v5-notes-gpt4omini/ 写入阶段用 gpt-4o-mini（要 OpenRouter 额度）
├── judges/                 回答与评分模型：deepseek / qwen-local / gpt-4o-mini
├── results/<实验>/<judge>/ locomo.json、lme.json、*.log、server.log
├── runner.py               通用执行器（run.py 只是它的薄封装）
├── compare.py              多个实验的逐题配对对比
├── lib/                    评测脚本：locomo.py、lme.py、llm_client.py
├── _archive/               之前探索阶段的日志和结果（已忽略，不入库）
└── .cache/ .run/           LLM 应答缓存、运行时生成的配置（已忽略）
```

`exp.yaml` 里 `db:` 相同的实验共用一份已写入的记忆（写入是幂等的），所以只改检索开关的实验不用重新写入。

## 怎么运行（Windows，在项目根目录）

跑某一个版本（前台运行，能看到进度条）：

```bash
.venv\Scripts\python eval\experiments\v4-window-rerank\run.py --judge deepseek
```

常用参数：

| 参数 | 含义 |
|---|---|
| `--judge deepseek` | 回答和评分模型，默认 deepseek；可选 `qwen-local`（很慢）、`gpt-4o-mini`（花钱） |
| `--only locomo` / `--only lme` | 只跑一个数据集 |
| `--locomo-per-conv 70` | LoCoMo 每段对话抽多少题，共 5 段，默认 70，共 350 题 |
| `--lme-per-type 10` | LongMemEval-S 每种题型抽多少题，共 6 种，默认 10，共 60 题 |
| `--force` | 已有结果也重跑 |
| `--port 8300` | 本地临时服务的端口 |

一次跑多个版本：

```bash
.venv\Scripts\python eval\runner.py v3-window v4-window-rerank v4a-chrono --judge deepseek
```

对比结果（第一个是基准，逐题统计胜负）：

```bash
.venv\Scripts\python eval\compare.py v4-window-rerank v4a-chrono v4b-cut40 --judge deepseek
```

## 运行时你会看到什么

1. `[实验名] server up on :8300`：本地临时检索服务已启动。
2. 每个数据集一个 tqdm 进度条：写入进度、答题进度，进度条后面显示当前累计准确率 `acc=`。
3. 数据集跑完后打印召回率、准确率、返回 token 数的汇总。
4. 结束时自动关闭临时服务。

调用细节（每个请求的路径、状态码、耗时）在 `eval/results/<实验>/<judge>/server.log`，可以另开一个终端 `Get-Content -Wait` 查看。

## 注意

- 样本量：LoCoMo 350 题加 LongMemEval-S 60 题，噪声约 ±2.5%，**只能分辨 5 个百分点以上的差异**；差距更小时看 `compare.py` 里的胜负题数，不要下结论。
- `v5-notes-gpt4omini` 会在写入阶段调用 OpenRouter 的 gpt-4o-mini，消耗额度；其余实验不调用 LLM 组件。
- 密钥只放在项目根目录的 `.env`（已被 git 忽略），实验目录里不放任何密钥。

## 通宵实验（免费 Qwen，一条命令）

```bash
bash eval/overnight.sh
```

依次跑开发集（v4、v6 到 v11 共 7 个实验）和留出验证（v4 对 v11），最后打印两张对比表（总表加分类表），
并保存到 `eval/results/summary_qwen-local.md` 和 `summary_qwen-local-holdout.md`。已完成的实验会跳过，
中断后重新运行同一条命令即可续跑。
