"""Run ablation experiments: one folder per version under eval/experiments/.

    python eval/runner.py v3-window v4-window-rerank --judge deepseek

For every experiment it generates a private config (eval/.run/, git-ignored), starts a local
PlainMemory server, ingests + searches + answers + judges LoCoMo-Refined and LongMemEval-S
subsets, and writes eval/results/<experiment>/<judge>/{locomo.json,lme.json,*.log}.
Experiments that share a `db:` reuse the same ingested memory (ingest is idempotent), so
search-time ablations cost no re-ingestion. Finished results are skipped unless --force.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml

EVAL = Path(__file__).resolve().parent
ROOT = EVAL.parent
LOCOMO_CONVS = "conv-26,conv-41,conv-42,conv-43,conv-48"


def read_env_file(path: Path) -> dict:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip("'")
    return values


def secrets() -> dict:
    text = (ROOT / ".env").read_text(encoding="utf-8")
    return dict(re.findall(r"^([A-Z_]+_API_KEY)=(\S+)", text, re.M))


def make_config(exp: dict, name: str, keys: dict, no_rerank: bool = False) -> Path:
    base = yaml.safe_load((ROOT / "config.local.yaml").read_text(encoding="utf-8"))
    cfg = {k: base[k] for k in ("api_token", "embedding", "chunk_tokens", "overlap_tokens")}
    cfg["database"] = f"../../data/eval/{exp.get('db', name)}.sqlite3"
    cfg["embedding"]["cache_path"] = "../../data/eval/embed-cache.sqlite3"
    cfg["add_timeout_seconds"] = 600
    cfg["search_timeout_seconds"] = 120
    cfg["rerank"] = {
        "base_url": "https://llm-et2ipodeqk0ea8be.cn-beijing.maas.aliyuncs.com/api/v1",
        "model": "qwen3-rerank",
        "api_key": keys["DASHSCOPE_API_KEY"],
        "cache_path": "../../data/eval/rerank-cache.sqlite3",  # identical (query, excerpt) pairs are free
    }
    if exp.get("llm") == "gpt-4o-mini":
        cfg["llm"] = {
            "base_url": "https://openrouter.ai/api/v1",
            "model": "openai/gpt-4o-mini",
            "api_key": keys["OPENROUTER_API_KEY"],
            "concurrency": 16,
        }
    cfg["retrieval"] = dict(exp.get("retrieval", {}))
    if no_rerank:
        cfg["retrieval"]["rerank"] = False
    out = EVAL / ".run" / f"{name}.yaml"
    out.parent.mkdir(exist_ok=True)
    out.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return out


def start_server(config: Path, port: int, log: Path, env: dict) -> subprocess.Popen:
    handle = log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "plain_memory",
            "--config",
            str(config),
            "serve",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        env=env,
        stdout=handle,
        stderr=handle,
    )
    for _ in range(90):
        try:
            if (
                httpx.get(f"http://127.0.0.1:{port}/health", timeout=2, trust_env=False).status_code
                == 200
            ):
                return process
        except httpx.HTTPError:
            pass
        if process.poll() is not None:
            raise SystemExit(f"server exited early; see {log}")
        time.sleep(1)
    process.kill()
    raise SystemExit("server did not become healthy")


def tee(cmd: list, env: dict, log_path: Path, quiet: bool = False):
    """Run cmd in the foreground: raw output (tqdm bars included) goes to the terminal, and a
    cleaned copy (last state of every progress line) goes to the log file."""
    process = subprocess.Popen(
        cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    buffer = b""
    with log_path.open("w", encoding="utf-8") as log:
        while True:
            chunk = os.read(process.stdout.fileno(), 4096)
            if not chunk:
                break
            if not quiet:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                log.write(line.rstrip(b"\r").split(b"\r")[-1].decode("utf-8", "replace") + "\n")
        if buffer:
            log.write(buffer.rstrip(b"\r").split(b"\r")[-1].decode("utf-8", "replace") + "\n")
    process.wait()


def run_experiment(
    name: str, args, keys: dict, root: Path = EVAL, port: int | None = None, quiet: bool = False
):
    exp = yaml.safe_load((root / "experiments" / name / "exp.yaml").read_text(encoding="utf-8"))
    out_dir = (
        root
        / "results"
        / name
        / (
            args.judge
            + (f"-{args.tag}" if getattr(args, "tag", None) else "")
            + ("-norerank" if getattr(args, "no_rerank", False) else "")
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [
        d
        for d in (args.only or ["locomo", "lme"])
        if args.force or not (out_dir / f"{d}.json").exists()
    ]
    if not todo:
        if not quiet:
            print(f"[{name}] already done for judge {args.judge} (use --force to redo)")
        return
    if exp.get("llm") and not keys.get("OPENROUTER_API_KEY"):
        raise SystemExit("this experiment needs OPENROUTER_API_KEY in .env")
    judge_env = read_env_file(EVAL / "judges" / f"{args.judge}.env")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", **keys)
    no_proxy = "localhost,127.0.0.1,.aliyuncs.com,.deepseek.com,10.208.62.156"
    env["NO_PROXY"] = env["no_proxy"] = no_proxy + (
        "," + env["NO_PROXY"] if env.get("NO_PROXY") else ""
    )
    config = make_config(exp, name, keys, getattr(args, "no_rerank", False))
    token = yaml.safe_load(config.read_text(encoding="utf-8"))["api_token"]
    port = port or args.port
    server = start_server(config, port, out_dir / "server.log", env)
    if not quiet:
        print(
            f"[{name}] server up on :{port}; judge={args.judge}; db={exp.get('db', name)}",
            flush=True,
        )
    run_id = exp.get("db", name)
    try:
        for dataset in todo:
            common = [
                "--base-url",
                f"http://127.0.0.1:{port}",
                "--token",
                token,
                "--run",
                run_id,
                "--out",
                str(out_dir / f"{dataset}.json"),
            ]
            if dataset == "locomo":
                cmd = [
                    sys.executable,
                    str(EVAL / "lib" / "locomo.py"),
                    *common,
                    "--convs",
                    getattr(args, "convs", None) or LOCOMO_CONVS,
                    "--per-conv",
                    str(args.locomo_per_conv),
                    "--workers",
                    str(args.workers),
                    *(
                        ["--seed", str(args.seed)]
                        if getattr(args, "seed", None) is not None
                        else []
                    ),
                ]
            else:
                cmd = [
                    sys.executable,
                    str(EVAL / "lib" / "lme.py"),
                    *common,
                    "--per-type",
                    str(args.lme_per_type),
                    "--workers",
                    "4",
                    *(
                        ["--seed", str(args.lme_seed)]
                        if getattr(args, "lme_seed", None) is not None
                        else []
                    ),
                ]
            if not quiet:
                print(f"[{name}] --- {dataset} ---", flush=True)
            tee(cmd, {**env, **judge_env}, out_dir / f"{dataset}.log", quiet)
            ok = (out_dir / f"{dataset}.json").exists()
            if not quiet:
                print(
                    f"[{name}] {dataset}: {'ok' if ok else 'FAILED (see ' + dataset + '.log)'}",
                    flush=True,
                )
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


def main(default_experiments=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="*", default=default_experiments)
    parser.add_argument(
        "--judge", default="deepseek", help="file name in eval/judges/ (without .env)"
    )
    parser.add_argument("--locomo-per-conv", type=int, default=70)
    parser.add_argument("--lme-per-type", type=int, default=10)
    parser.add_argument("--only", nargs="+", choices=["locomo", "lme"])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--port", type=int, default=8300)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--no-rerank", action="store_true", help="turn the reranker off (no DashScope rerank cost)"
    )
    parser.add_argument("--tag", help="suffix for the results folder, e.g. holdout")
    parser.add_argument("--convs", help="LoCoMo conversations, comma separated")
    parser.add_argument("--seed", type=int, help="LoCoMo question sample seed")
    parser.add_argument("--lme-seed", type=int, help="LongMemEval sample seed")
    args = parser.parse_args()
    if not args.experiments:
        parser.error("name at least one experiment")
    keys = secrets()
    for name in args.experiments:
        run_experiment(name, args, keys)


if __name__ == "__main__":
    main()
