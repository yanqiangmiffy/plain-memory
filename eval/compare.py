"""Paired comparison of experiments (same questions, same judge), with per-category detail.

    python eval/compare.py v4-window-rerank v6-rule-notes v7-date-boost --judge qwen-local
    python eval/compare.py v4-window-rerank v11-all-features --judge qwen-local --tag holdout

The first experiment (or --base) is the reference. delta / win / loss are counted per question
over the questions both runs answered. Output is also saved to eval/results/summary_<judge>.md.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

EVAL = Path(__file__).resolve().parent
LOCOMO_CATEGORIES = {
    "1": "locomo single-hop",
    "2": "locomo temporal",
    "3": "locomo inference",
    "4": "locomo open",
}


def load(name: str, folder_name: str):
    folder = EVAL / "results" / name / folder_name
    result = {}
    for dataset, key, label in (("locomo", "qa_id", "category"), ("lme", "qid", "type")):
        path = folder / f"{dataset}.json"
        if path.exists():
            rows = json.loads(path.read_text(encoding="utf-8"))
            result[dataset] = {r[key]: bool(r["correct"]) for r in rows}
            result[dataset + "_group"] = {
                r[key]: (
                    LOCOMO_CATEGORIES.get(str(r[label]), str(r[label]))
                    if dataset == "locomo"
                    else "lme " + r[label]
                )
                for r in rows
            }
            result[dataset + "_tokens"] = sum(r["mem_tokens"] for r in rows) / len(rows)
    return result


def paired(run, base, select=None):
    pairs = []
    for dataset in ("locomo", "lme"):
        data, ref = run.get(dataset), base.get(dataset)
        if not data or not ref:
            continue
        for k, ok in data.items():
            if k in ref and (select is None or run[dataset + "_group"][k] == select):
                pairs.append((ok, ref[k]))
    return pairs


def stats(pairs):
    if not pairs:
        return None
    acc = sum(a for a, _ in pairs) / len(pairs)
    ref = sum(b for _, b in pairs) / len(pairs)
    wins = sum(a and not b for a, b in pairs)
    loss = sum(b and not a for a, b in pairs)
    z = (wins - loss) / math.sqrt(wins + loss) if wins + loss else 0.0
    return acc, acc - ref, wins, loss, z, len(pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="+")
    parser.add_argument("--judge", default="deepseek")
    parser.add_argument("--tag")
    parser.add_argument("--base")
    args = parser.parse_args()
    folder = args.judge + (f"-{args.tag}" if args.tag else "")
    runs = {name: load(name, folder) for name in args.experiments}
    base_name = args.base or args.experiments[0]
    base = runs[base_name]
    lines = [
        f"judge: {folder}   reference: {base_name}",
        "",
        "| experiment | LoCoMo | LongMemEval | all | delta | win/loss | z | avg tokens |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, run in runs.items():
        cells = [
            f"{sum(run[d].values()) / len(run[d]):.3f} ({len(run[d])})" if d in run else "-"
            for d in ("locomo", "lme")
        ]
        s = stats(paired(run, base))
        tokens = [run[d + "_tokens"] for d in ("locomo", "lme") if d in run]
        avg = f"{sum(tokens) / len(tokens):.0f}" if tokens else "-"
        if s:
            lines.append(
                f"| {name} | {cells[0]} | {cells[1]} | {s[0]:.3f} | {s[1]:+.3f} | {s[2]}/{s[3]} | {s[4]:+.1f} | {avg} |"
            )
        else:
            lines.append(f"| {name} | {cells[0]} | {cells[1]} | - | - | - | - | {avg} |")
    groups = defaultdict(set)
    for run in runs.values():
        for d in ("locomo", "lme"):
            for group in run.get(d + "_group", {}).values():
                groups[d].add(group)
    names = sorted(groups["locomo"]) + sorted(groups["lme"])
    lines += [
        "",
        "Per category: accuracy (delta vs reference, wins/losses)",
        "",
        "| experiment | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for name, run in runs.items():
        cells = []
        for group in names:
            s = stats(paired(run, base, group))
            cells.append(f"{s[0]:.2f} ({s[1]:+.2f}, +{s[2]}/-{s[3]})" if s else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    print(text)
    (EVAL / "results" / f"summary_{folder}.md").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
