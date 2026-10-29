"""Self-evolving ablation search for the search-time retrieval settings.

    .venv\\Scripts\\python eval\\evolve.py --id evo1 --judge deepseek [--parallel 3]
    .venv\\Scripts\\python eval\\evolve.py --id evo1 --status

Loop (greedy hill-climbing on the paired dev comparison):
  round 0   run the deployed configuration (v4: window +-2, rerank) as the reference
  round k   build one candidate per knob neighbour of the current best, run them in parallel,
            keep those that beat the best by a paired win/loss test, try the union of the top
            winners, adopt the best of {top single, union}; stop when nothing improves
  confirm   re-run reference and final best on held-out conversations/histories
  report    eval/evolution/<id>/report.md with every experiment, decisions and the final config

Everything is resumable: finished experiments are read back from disk, never recomputed.
All experiments share one ingested memory (search-time knobs only), so no re-ingestion.
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import queue
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import yaml
from tqdm.auto import tqdm

EVAL = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL))
import runner  # noqa: E402

DEFAULTS = {
    "window_before": 2,
    "window_after": 2,
    "candidate_units": 60,
    "rerank_weight": 0.7,
    "min_relative_score": 0.0,
    "context_tokens": 60000,
    "order": "relevance",
    "max_block_units": 12,
    "rerank_candidates": 50,
    "rrf_k": 60,
}
BASE = {**DEFAULTS, "rerank": True}  # the deployed 0.2.yq configuration

# knob -> ordered ladder; every ladder entry sets the listed keys. Neighbours are +-1 steps.
LADDERS = {
    "window": [{"window_before": n, "window_after": n} for n in (0, 1, 2, 3, 4, 5)],
    "candidate_units": [{"candidate_units": n} for n in (15, 30, 45, 60, 90, 120)],
    "rerank_weight": [{"rerank_weight": n} for n in (0.3, 0.5, 0.7, 0.85, 1.0)],
    "min_relative_score": [{"min_relative_score": n} for n in (0.0, 0.2, 0.4, 0.6)],
    "context_tokens": [{"context_tokens": n} for n in (10000, 20000, 40000, 60000)],
    "order": [{"order": "relevance"}, {"order": "chronological"}],
    "max_block_units": [{"max_block_units": n} for n in (6, 8, 12, 20)],
    "rerank_candidates": [{"rerank_candidates": n} for n in (20, 30, 50, 80)],
    "rrf_k": [{"rrf_k": n} for n in (20, 60, 100)],
}
HOLDOUT_CONVS = "conv-30,conv-44,conv-47,conv-49,conv-50"
MIN_NET, MIN_Z = 3, 1.0


def ladder_index(eff: dict, knob: str) -> int:
    for i, entry in enumerate(LADDERS[knob]):
        if all(eff[k] == v for k, v in entry.items()):
            return i
    raise ValueError(f"{knob}: value not on ladder")


def neighbours(eff: dict):
    """Yield (label, new_effective_config) for every one-step change of one knob."""
    for knob, ladder in LADDERS.items():
        i = ladder_index(eff, knob)
        for j in (i - 1, i + 1):
            if 0 <= j < len(ladder):
                yield (
                    f"{knob}: {_show(ladder[i])} -> {_show(ladder[j])}",
                    {**eff, **ladder[j]},
                    knob,
                )


def _show(entry: dict) -> str:
    return ",".join(f"{v}" for v in entry.values())


def digest(eff: dict) -> str:
    return hashlib.sha1(json.dumps(eff, sort_keys=True).encode()).hexdigest()[:6]


class Evolution:
    def __init__(self, args):
        self.args = args
        self.root = EVAL / "evolution" / args.id
        (self.root / "experiments").mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.state = (
            json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state_path.exists()
            else {"experiments": {}, "rounds": [], "best": None, "holdout": None}
        )
        self.lock = threading.Lock()
        self.keys = runner.secrets()
        self.ports = queue.Queue()
        for i in range(args.parallel):
            self.ports.put(args.port + i)
        self.dev = SimpleNamespace(
            judge=args.judge,
            locomo_per_conv=args.dev_locomo,
            lme_per_type=args.dev_lme,
            only=None,
            workers=12,
            port=args.port,
            force=False,
            convs=None,
            seed=None,
            lme_seed=None,
        )
        self.holdout = SimpleNamespace(
            judge=args.judge,
            locomo_per_conv=args.dev_locomo,
            lme_per_type=args.dev_lme,
            only=None,
            workers=12,
            port=args.port,
            force=False,
            convs=HOLDOUT_CONVS,
            seed=7,
            lme_seed=7,
        )

    # ---- bookkeeping -------------------------------------------------------------------
    def log(self, message: str):
        line = f"{time.strftime('%H:%M:%S')}  {message}"
        tqdm.write(line)
        with (self.root / "progress.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def save(self):
        self.state_path.write_text(json.dumps(self.state, indent=1), encoding="utf-8")

    def experiment(
        self, eff: dict, label: str, round_no: int, parent: str | None, suffix: str = ""
    ) -> str:
        name = f"r{round_no}-{digest(eff)}{suffix}"
        known = self.state["experiments"]
        for existing, record in known.items():  # same configuration already run -> reuse
            if record["eff"] == eff and record.get("suffix", "") == suffix:
                return existing
        folder = self.root / "experiments" / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "exp.yaml").write_text(
            yaml.safe_dump({"description": label, "db": "ctx-on", "retrieval": dict(eff)}),
            encoding="utf-8",
        )
        known[name] = {
            "eff": eff,
            "label": label,
            "round": round_no,
            "parent": parent,
            "suffix": suffix,
        }
        return name

    def run(self, name: str, holdout: bool = False) -> dict:
        """Run one experiment (blocking) and return {dataset: {question: correct}}."""
        port = self.ports.get()
        try:
            for attempt in (
                1,
                2,
            ):  # a crashed harness is retried once; the LLM cache keeps it cheap
                runner.run_experiment(
                    name,
                    self.holdout if holdout else self.dev,
                    self.keys,
                    root=self.root,
                    port=port,
                    quiet=True,
                )
                try:
                    return self.load(name)
                except RuntimeError:
                    if attempt == 2:
                        raise
                    self.log(f"  {name}: incomplete results, retrying once")
                    time.sleep(5)
        finally:
            self.ports.put(port)

    def load(self, name: str) -> dict:
        folder = self.root / "results" / name / self.args.judge
        out = {}
        for dataset, key in (("locomo", "qa_id"), ("lme", "qid")):
            path = folder / f"{dataset}.json"
            if not path.exists():
                raise RuntimeError(
                    f"{name}: no {dataset} results (see {folder / (dataset + '.log')})"
                )
            out[dataset] = {
                r[key]: bool(r["correct"]) for r in json.loads(path.read_text(encoding="utf-8"))
            }
        return out

    @staticmethod
    def accuracy(result: dict) -> float:
        values = [v for d in result.values() for v in d.values()]
        return sum(values) / len(values)

    @staticmethod
    def paired(a: dict, b: dict) -> dict:
        pairs = [(a[d][k], b[d][k]) for d in a for k in a[d] if k in b.get(d, {})]
        wins = sum(x and not y for x, y in pairs)
        losses = sum(y and not x for x, y in pairs)
        net = wins - losses
        z = net / math.sqrt(wins + losses) if wins + losses else 0.0
        return {"wins": wins, "losses": losses, "net": net, "z": round(z, 2), "n": len(pairs)}

    def record(self, name: str, result: dict, parent_result: dict | None):
        entry = self.state["experiments"][name]
        entry["acc"] = round(self.accuracy(result), 4)
        entry["locomo"] = round(sum(result["locomo"].values()) / len(result["locomo"]), 4)
        entry["lme"] = round(sum(result["lme"].values()) / len(result["lme"]), 4)
        if parent_result is not None:
            entry["vs_parent"] = self.paired(result, parent_result)
        self.save()

    # ---- the search --------------------------------------------------------------------
    def ensure(self, names: list[str], parent: str, parent_result: dict, title: str) -> dict:
        """Results for `names`, running the unfinished ones in parallel and comparing every one
        with `parent` (re-comparing experiments that were first measured against another parent)."""
        experiments = self.state["experiments"]
        todo = [n for n in names if "acc" not in experiments[n]]
        results = {}
        if todo:
            with cf.ThreadPoolExecutor(self.args.parallel) as pool:
                futures = {pool.submit(self.run, n): n for n in todo}
                for future in tqdm(cf.as_completed(futures), total=len(futures), desc=title):
                    name = futures[future]
                    try:
                        results[name] = future.result()
                    except Exception as exc:  # one failed experiment must not stop the search
                        self.log(f"  {name}: FAILED ({exc})")
                        continue
                    with self.lock:
                        experiments[name]["parent"] = parent
                        self.record(name, results[name], parent_result)
                    self.describe(name)
        for name in names:
            if name not in results and "acc" in experiments[name]:
                results[name] = self.load(name)
                if experiments[name].get("parent") != parent:
                    experiments[name]["parent"] = parent
                    self.record(name, results[name], parent_result)
        self.save()
        return results

    def describe(self, name: str):
        entry = self.state["experiments"][name]
        versus = entry.get("vs_parent")
        tail = (
            f"  vs parent: +{versus['wins']}/-{versus['losses']} net {versus['net']:+d} z={versus['z']}"
            if versus
            else ""
        )
        self.log(
            f"  done {name}  acc={entry['acc']:.3f} (locomo {entry['locomo']:.3f}, "
            f"lme {entry['lme']:.3f})  [{entry['label']}]{tail}"
        )

    def improved(self, name: str) -> bool:
        versus = self.state["experiments"][name].get("vs_parent")
        return bool(versus) and versus["net"] >= MIN_NET and versus["z"] >= MIN_Z

    def search(self):
        args = self.args
        experiments = self.state["experiments"]
        best_eff = dict(BASE)
        base_name = self.experiment(
            best_eff, "reference (deployed 0.2.yq: window +-2, rerank)", 0, None
        )
        self.state["base"] = base_name
        self.log(
            f"=== evolution {args.id}: judge={args.judge}, dev = LoCoMo {args.dev_locomo}/conv x5 + "
            f"LongMemEval {args.dev_lme}/type x6, parallel={args.parallel}"
        )
        self.log(f"round 0: reference {base_name} (ingests the shared memory first)")
        base_result = self.run(base_name)
        self.record(base_name, base_result, None)
        self.log(f"  reference acc={experiments[base_name]['acc']:.3f}")
        best_name, best_result = base_name, base_result
        spent = 1
        for round_no in range(1, args.max_rounds + 1):
            cands = {}
            for label, eff, knob in neighbours(best_eff):
                name = self.experiment(eff, label, round_no, best_name)
                if name != best_name:
                    cands[name] = knob
            unfinished = [n for n in cands if "acc" not in experiments[n]]
            room = args.max_experiments - spent
            if room <= 0 and unfinished:
                self.log("experiment budget exhausted")
                break
            for dropped in unfinished[room:]:
                del cands[dropped]
            self.log(f"round {round_no}: {len(cands)} candidates around {best_name}")
            results = self.ensure(list(cands), best_name, best_result, f"round {round_no}")
            spent += len([n for n in unfinished if n in cands])
            winners = sorted(
                (n for n in results if self.improved(n)),
                key=lambda n: -experiments[n]["vs_parent"]["net"],
            )
            self.state["rounds"].append(
                {"round": round_no, "parent": best_name, "winners": winners}
            )
            self.save()
            if not winners:
                self.log(f"round {round_no}: no candidate beat {best_name}; stopping")
                break
            choice, choice_result = winners[0], results[winners[0]]
            picked = {}
            for n in winners:
                picked.setdefault(cands[n], n)  # best candidate per knob
            top = list(picked.values())[:3]
            if len(top) >= 2 and spent < args.max_experiments:
                merged = dict(best_eff)
                for n in top:
                    merged.update(
                        {k: v for k, v in experiments[n]["eff"].items() if v != best_eff[k]}
                    )
                label = "union of " + ", ".join(experiments[n]["label"] for n in top)
                union = self.experiment(merged, label, round_no, best_name, suffix="u")
                self.log(f"round {round_no}: testing the union of {len(top)} winners")
                if "acc" not in experiments[union]:
                    spent += 1
                union_result = self.ensure(
                    [union], best_name, best_result, f"round {round_no} union"
                )
                if union in union_result and self.improved(union):
                    if (
                        experiments[union]["vs_parent"]["net"]
                        > experiments[choice]["vs_parent"]["net"]
                    ):
                        choice, choice_result = union, union_result[union]
            self.log(f"round {round_no}: adopting {choice} [{experiments[choice]['label']}]")
            best_name, best_result = choice, choice_result
            best_eff = dict(experiments[choice]["eff"])
            self.state["best"] = best_name
            self.save()
        self.state["best"] = best_name
        self.save()
        return base_name, best_name

    def confirm(self, base_name: str, best_name: str):
        if base_name == best_name:
            self.log("confirm: search kept the reference configuration; nothing to confirm")
            return
        self.log("confirm: re-running reference and best on held-out conversations / histories")
        copies = {}
        for name in (base_name, best_name):
            held = f"{name}-holdout"
            folder = self.root / "experiments" / held
            folder.mkdir(parents=True, exist_ok=True)
            source = self.root / "experiments" / name / "exp.yaml"
            (folder / "exp.yaml").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            copies[name] = held
        base_result = self.run(copies[base_name], holdout=True)  # first one ingests the new data
        best_result = self.run(copies[best_name], holdout=True)
        versus = self.paired(best_result, base_result)
        self.state["holdout"] = {
            "base_acc": round(self.accuracy(base_result), 4),
            "best_acc": round(self.accuracy(best_result), 4),
            **versus,
        }
        self.save()
        self.log(
            f"  holdout: reference {self.state['holdout']['base_acc']:.3f} -> best "
            f"{self.state['holdout']['best_acc']:.3f}  (+{versus['wins']}/-{versus['losses']}, n={versus['n']})"
        )

    def report(self):
        experiments = self.state["experiments"]
        best = self.state.get("best")
        lines = [f"# Evolution {self.args.id} (judge: {self.args.judge})", ""]
        lines += [
            "| experiment | round | change | dev acc | LoCoMo | LME | vs parent (+/-) | net |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, e in sorted(experiments.items(), key=lambda kv: (kv[1]["round"], kv[0])):
            if "acc" not in e:
                continue
            v = e.get("vs_parent")
            mark = " **best**" if name == best else ""
            lines.append(
                f"| {name}{mark} | {e['round']} | {e['label']} | {e['acc']:.3f} | {e['locomo']:.3f} | "
                f"{e['lme']:.3f} | {('+%d/-%d' % (v['wins'], v['losses'])) if v else '-'} | "
                f"{v['net'] if v else '-'} |"
            )
        lines += ["", "## Decisions", ""]
        for r in self.state["rounds"]:
            lines.append(
                f"- round {r['round']} (parent {r['parent']}): winners {r['winners'] or 'none'}"
            )
        holdout = self.state.get("holdout")
        lines += ["", "## Held-out confirmation", ""]
        lines.append(
            f"reference {holdout['base_acc']:.3f} -> best {holdout['best_acc']:.3f}, "
            f"+{holdout['wins']}/-{holdout['losses']} over {holdout['n']} questions"
            if holdout
            else "not run (the search kept the reference)"
        )
        if best:
            diff = {k: v for k, v in experiments[best]["eff"].items() if v != BASE.get(k)}
            lines += [
                "",
                "## Recommended `retrieval:` overrides for config.local.yaml",
                "",
                "```yaml",
                yaml.safe_dump({"retrieval": {"rerank": True, **diff}}).rstrip(),
                "```",
            ]
        lines += [
            "",
            "Caveats: the judge is not the platform's judge, and the dev set is small (about 410 questions,",
            "noise ~ +-2.5%); only adopt a change if the held-out result is also non-negative.",
        ]
        (self.root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.log(f"report written: {self.root / 'report.md'}")

    def status(self):
        experiments = self.state["experiments"]
        finished = [n for n, e in experiments.items() if "acc" in e]
        print(
            f"{self.args.id}: {len(finished)} finished / {len(experiments)} defined; best = {self.state.get('best')}"
        )
        log = self.root / "progress.log"
        if log.exists():
            print("".join(log.read_text(encoding="utf-8").splitlines(keepends=True)[-12:]), end="")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="evo1")
    parser.add_argument("--judge", default="deepseek")
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-experiments", type=int, default=30)
    parser.add_argument("--dev-locomo", type=int, default=70)
    parser.add_argument("--dev-lme", type=int, default=10)
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    evolution = Evolution(args)
    if args.status:
        evolution.status()
        return
    base_name, best_name = evolution.search()
    evolution.confirm(base_name, best_name)
    evolution.report()


if __name__ == "__main__":
    main()
