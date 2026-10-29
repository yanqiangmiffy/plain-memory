"""Offline LongMemEval-S harness (stratified subset): Add every session, Search, answer, judge.

usage: python eval/lme.py --base-url http://127.0.0.1:8100 --token T --per-type 6
"""

import argparse
import concurrent.futures as cf
import datetime as dt
import importlib.util
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx
import tiktoken
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import llm_client  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/longmemeval/longmemeval_s_cleaned.json"
spec = importlib.util.spec_from_file_location(
    "pipe", ROOT / "agent-memory-leaderboard/data/longmemeval-s/pipeline.py"
)
pipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipe)
enc = tiktoken.get_encoding("cl100k_base")


def parse_date(text: str) -> int:
    d = dt.datetime.strptime(text.split(" (")[0] + " " + text.split(") ")[1], "%Y/%m/%d %H:%M")
    return int(d.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


def batches(messages):
    cur, words = [], 0
    for m in messages:
        w = len(m["content"].split())
        if cur and (len(cur) >= 20 or words + w > 2000):
            yield cur
            cur, words = [], 0
        cur.append(m)
        words += w
    if cur:
        yield cur


def ingest(client, item, run, headers):
    uid = f"eval:{run}:lme:{item['question_id']}"
    jobs = []
    for si, (sid, date, session) in enumerate(
        zip(item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"])
    ):
        ts = parse_date(date)
        for bi, b in enumerate(batches(session)):
            jobs.append(
                {
                    "request_id": f"eval:{run}:lme:{item['question_id']}:{si}:chunk-{bi}",
                    "user_id": uid,
                    "session_id": f"eval:{run}:{item['question_id']}:{sid}",
                    "messages": [
                        {"role": m["role"], "timestamp": ts, "content": m["content"]} for m in b
                    ],
                }
            )

    def post(body):
        r = None
        for _ in range(120):
            r = client.post("/add", json=body, headers=headers, timeout=1800)
            if r.status_code == 200:
                return
            time.sleep(1 if r.status_code == 429 else 3)
        raise RuntimeError(f"add failed {r.status_code} {r.text[:200]}")

    with cf.ThreadPoolExecutor(6) as ex:
        list(ex.map(post, jobs))
    return uid


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def one(client, headers, item, run, args):
    t0 = time.time()
    if args.skip_ingest:
        uid = f"eval:{run}:lme:{item['question_id']}"
    else:
        uid = ingest(client, item, run, headers)
    t_ing = time.time() - t0
    t1 = time.time()
    r = client.post(
        "/search",
        json={"query": item["question"], "user_id": uid, "top_k": 100},
        headers=headers,
        timeout=1800,
    )
    r.raise_for_status()
    items = r.json()["data"]
    lat = time.time() - t1
    blob = norm("\n".join(i["content"] for i in items))
    ev = [
        norm(t["content"])[:200]
        for s in item["haystack_sessions"]
        for t in s
        if t.get("has_answer")
    ]
    hit = [e in blob for e in ev]
    mem, used = [], 0
    for i in items:
        n = len(enc.encode(i["content"], disallowed_special=()))
        if used + n > 117760:
            break
        mem.append(i["content"])
        used += n
    res = {
        "qid": item["question_id"],
        "type": item["question_type"],
        "n_items": len(items),
        "recall_all": all(hit) if hit else None,
        "recall_frac": sum(hit) / len(hit) if hit else None,
        "mem_tokens": used,
        "latency": lat,
        "ingest": t_ing,
    }
    if not args.recall_only:
        q = {
            "id": item["question_id"],
            "question": item["question"],
            "speaker_1_name": "user",
            "speaker_2_name": "assistant",
            "speaker_1_memories": "\n\n".join(mem),
            "speaker_2_memories": "",
        }
        gen = llm_client.chat(pipe.render_answer_prompt(q), 256)
        gold = str(item["answer"])
        jp = (
            pipe.ACCURACY_PROMPT.replace("{question}", item["question"])
            .replace("{gold_answer}", gold)
            .replace("{generated_answer}", gen)
        )
        try:
            label = pipe.parse_judge_label(llm_client.chat(jp, 200))
        except Exception:
            label = "WRONG"
        res.update(generated=gen, gold=gold, correct=label == "CORRECT")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--token", required=True)
    ap.add_argument("--per-type", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--recall-only", action="store_true")
    ap.add_argument("--run", default=None)
    ap.add_argument("--skip-ingest", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    run = args.run or "r" + str(int(time.time()))
    headers = {"Authorization": f"Bearer {args.token}"}
    data = json.load(open(DATA, encoding="utf-8"))
    rnd = random.Random(args.seed)
    by_type = defaultdict(list)
    for d in data:
        by_type[d["question_type"]].append(d)
    chosen = []
    for t in sorted(by_type):
        chosen += rnd.sample(by_type[t], args.per_type)
    del data
    client = httpx.Client(base_url=args.base_url, trust_env=False)  # local server: never via proxy
    results = []
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(one, client, headers, it, run, args) for it in chosen]
        bar = tqdm(cf.as_completed(futures), total=len(futures), desc="LongMemEval-S questions")
        for future in bar:
            results.append(future.result())
            if not args.recall_only:
                bar.set_postfix(acc=f"{sum(r['correct'] for r in results) / len(results):.3f}")
    by = defaultdict(list)
    for r in results:
        by[r["type"]].append(r)
        by["all"].append(r)

    def avg(rs, k):
        v = [r[k] for r in rs if r.get(k) is not None]
        return sum(v) / len(v) if v else float("nan")

    print(f"run={run} n={len(results)}")
    for k in sorted(by):
        rs = by[k]
        line = (
            f"{k:>24} n={len(rs):3d} recall_all={avg(rs, 'recall_all'):.3f} "
            f"items={avg(rs, 'n_items'):.0f} memtok={avg(rs, 'mem_tokens'):.0f} "
            f"lat={avg(rs, 'latency'):.2f}s ingest={avg(rs, 'ingest'):.0f}s"
        )
        if not args.recall_only:
            line += f" ACC={avg(rs, 'correct'):.3f}"
        print(line)
    if args.out:
        Path(args.out).write_text(
            json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
