"""Offline LoCoMo-Refined harness: simulates AML Add/Search + answer + judge.

usage: python eval/locomo.py --base-url http://127.0.0.1:8000 --token T --convs conv-26 --per-conv 40
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
DATA = ROOT / "datasets/locomo-refined/data/public"
spec = importlib.util.spec_from_file_location(
    "pipe", ROOT / "agent-memory-leaderboard/data/locomo-refined/pipeline.py"
)
pipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipe)
enc = tiktoken.get_encoding("cl100k_base")


def parse_date(text: str) -> int:
    # "1:56 pm on 8 May, 2023"
    d = dt.datetime.strptime(text.replace(",", ""), "%I:%M %p on %d %B %Y")
    return int(d.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


def msg_text(m):
    t = f"{m['speaker']}: {m['text']}"
    if m.get("blip_caption"):
        t += f" [shared image: {m['blip_caption']}]"
    return t


def batches(messages):
    cur, words = [], 0
    for m in messages:
        w = len(msg_text(m).split())
        if cur and (len(cur) >= 20 or words + w > 2000):
            yield cur
            cur, words = [], 0
        cur.append(m)
        words += w
    if cur:
        yield cur


def ingest(client, conv, run, headers):
    uid = f"eval:{run}:locomo:{conv['sample_id']}"
    jobs = []
    for s in conv["sessions"]:
        ts = parse_date(s["date_time"])
        for i, b in enumerate(batches(s["messages"])):
            body = {
                "request_id": f"eval:{run}:locomo_refined:{conv['sample_id']}:s{s['session_index']}:chunk-{i}",
                "user_id": uid,
                "session_id": f"eval:{run}:{conv['sample_id']}:session-{s['session_index']}",
                "messages": [
                    {
                        "role": "user" if m["speaker"] == conv["speaker_a"] else "assistant",
                        "timestamp": ts,
                        "content": msg_text(m),
                    }
                    for m in b
                ],
            }
            jobs.append(body)

    def post(body):
        r = None
        for _ in range(8):
            r = client.post("/add", json=body, headers=headers, timeout=1800)
            if r.status_code == 200:
                return
            time.sleep(2)
        raise RuntimeError(f"add failed {r.status_code} {r.text[:200]}")

    with cf.ThreadPoolExecutor(8) as ex:
        for _ in tqdm(
            ex.map(post, jobs), total=len(jobs), desc=f"ingest {conv['sample_id']}", leave=False
        ):
            pass
    return uid


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def run_question(client, headers, uid, q, args):
    t0 = time.time()
    r = client.post(
        "/search",
        json={"query": q["question"], "user_id": uid, "top_k": 100},
        headers=headers,
        timeout=1800,
    )
    r.raise_for_status()
    items = r.json()["data"]
    lat = time.time() - t0
    blob = norm("\n".join(i["content"] for i in items))
    ev = [norm(m["text"]) for m in q["evidence_messages"]]
    hit = [e in blob for e in ev]
    res = {
        "qa_id": q["qa_id"],
        "category": q["category"],
        "n_items": len(items),
        "recall_all": all(hit) if hit else None,
        "recall_frac": sum(hit) / len(hit) if hit else None,
        "latency": lat,
    }
    # budget-limited memory text, in returned order
    mem, used = [], 0
    for i in items:
        n = len(enc.encode(i["content"], disallowed_special=()))
        if used + n > 117760:
            break
        mem.append(i["content"])
        used += n
    res["mem_tokens"] = used
    if not args.recall_only:
        item = {
            "id": q["qa_id"],
            "question": q["question"],
            "speaker_1_name": q["speaker_a"],
            "speaker_2_name": q["speaker_b"],
            "speaker_1_memories": "\n\n".join(mem),
            "speaker_2_memories": "",
        }
        gen = llm_client.chat(pipe.render_answer_prompt(item), 256)
        gold = " | ".join(q["answer"]) if isinstance(q["answer"], list) else str(q["answer"])
        jp = (
            pipe.ACCURACY_PROMPT.replace("{question}", q["question"])
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
    ap.add_argument("--convs", default="conv-26")
    ap.add_argument("--per-conv", type=int, default=0, help="subset size, 0=all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recall-only", action="store_true")
    ap.add_argument("--run", default=None)
    ap.add_argument("--skip-ingest", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    run = args.run or "r" + str(int(time.time()))
    headers = {"Authorization": f"Bearer {args.token}"}
    convs = {
        c["sample_id"]: c
        for c in map(json.loads, open(DATA / "conversations.jsonl", encoding="utf-8"))
    }
    qs = [json.loads(line) for line in open(DATA / "questions.jsonl", encoding="utf-8")]
    client = httpx.Client(base_url=args.base_url, trust_env=False)  # local server: never via proxy
    results = []
    rnd = random.Random(args.seed)
    for cid in args.convs.split(","):
        t0 = time.time()
        uid = f"eval:{run}:locomo:{cid}"
        if not args.skip_ingest:
            ingest(client, convs[cid], run, headers)
        print(f"[{cid}] ingest {time.time() - t0:.1f}s", flush=True)
        cq = [q for q in qs if q["sample_id"] == cid]
        if args.per_conv and args.per_conv < len(cq):
            cq = rnd.sample(cq, args.per_conv)
        done = []
        with cf.ThreadPoolExecutor(args.workers) as ex:
            futures = [ex.submit(run_question, client, headers, uid, q, args) for q in cq]
            bar = tqdm(cf.as_completed(futures), total=len(futures), desc=f"{cid} questions")
            for future in bar:
                done.append(future.result())
                if not args.recall_only:
                    bar.set_postfix(acc=f"{sum(r['correct'] for r in done) / len(done):.3f}")
        results += done
    by = defaultdict(list)
    for r in results:
        by[r["category"]].append(r)
        by["all"].append(r)

    def avg(rs, k):
        v = [r[k] for r in rs if r.get(k) is not None]
        return sum(v) / len(v) if v else float("nan")

    print(f"run={run} n={len(results)}")
    for k in sorted(by):
        rs = by[k]
        line = (
            f"cat {k:>3} n={len(rs):4d} recall_all={avg(rs, 'recall_all'):.3f} "
            f"recall_frac={avg(rs, 'recall_frac'):.3f} items={avg(rs, 'n_items'):.0f} "
            f"memtok={avg(rs, 'mem_tokens'):.0f} lat={avg(rs, 'latency'):.2f}s"
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
