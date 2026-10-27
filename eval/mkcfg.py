"""Generate a private eval config: python eval/mkcfg.py NAME key=value ... (dotted keys).

e.g. python eval/mkcfg.py B retrieval.add_notes=true retrieval.show_notes=true
Keys are read from .env; output config.eval<NAME>.local.yaml (git-ignored) with its own DB.
"""

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
env = dict(re.findall(r"^([A-Z_]+_API_KEY)=(\S+)", (ROOT / ".env").read_text(encoding="utf-8"), re.M))
cfg = yaml.safe_load((ROOT / "config.local.yaml").read_text(encoding="utf-8"))
name = sys.argv[1]
cfg["database"] = f"data/eval-{name}.sqlite3"
cfg["llm"] = {
    "base_url": "https://openrouter.ai/api/v1",
    "model": "openai/gpt-4o-mini",
    "api_key": env["OPENROUTER_API_KEY"],
    "concurrency": 16,
}
cfg["rerank"] = {
    "base_url": "https://llm-et2ipodeqk0ea8be.cn-beijing.maas.aliyuncs.com/api/v1",
    "model": "qwen3-rerank",
    "api_key": env["DASHSCOPE_API_KEY"],
}
cfg["retrieval"] = {}
for arg in sys.argv[2:]:
    key, value = arg.split("=", 1)
    target = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = yaml.safe_load(value)
out = ROOT / f"config.eval{name}.local.yaml"
out.write_text(yaml.safe_dump(cfg), encoding="utf-8")
print(out.name, cfg["api_token"][:4] + "...")
