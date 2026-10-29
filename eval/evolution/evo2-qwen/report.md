# Evolution evo2-qwen (judge: qwen-local)

| experiment | round | change | dev acc | LoCoMo | LME | vs parent (+/-) | net |
|---|---|---|---|---|---|---|---|
| r0-c4036e **best** | 0 | reference (deployed 0.2.yq: window +-2, rerank) | 0.745 | 0.736 | 0.792 | - | - |
| r1-0a8370 | 1 | window: 2,2 -> 1,1 | 0.738 | 0.736 | 0.750 | +5/-6 | -1 |
| r1-5a4952 | 1 | min_relative_score: 0.0 -> 0.2 | 0.745 | 0.736 | 0.792 | +0/-0 | 0 |
| r1-86b0f4 | 1 | rerank_weight: 0.7 -> 0.85 | 0.738 | 0.744 | 0.708 | +4/-5 | -1 |
| r1-a52334 | 1 | window: 2,2 -> 3,3 | 0.738 | 0.752 | 0.667 | +4/-5 | -1 |
| r1-ae475c | 1 | candidate_units: 60 -> 90 | 0.745 | 0.760 | 0.667 | +6/-6 | 0 |
| r1-c74269 | 1 | candidate_units: 60 -> 45 | 0.738 | 0.728 | 0.792 | +5/-6 | -1 |
| r1-e97f73 | 1 | rerank_weight: 0.7 -> 0.5 | 0.758 | 0.760 | 0.750 | +5/-3 | 2 |

## Decisions

- round 1 (parent r0-c4036e): winners none

## Held-out confirmation

not run (the search kept the reference)

## Recommended `retrieval:` overrides for config.local.yaml

```yaml
retrieval:
  rerank: true
```

Caveats: the judge is not the platform's judge, and the dev set is small (about 410 questions,
noise ~ +-2.5%); only adopt a change if the held-out result is also non-negative.
