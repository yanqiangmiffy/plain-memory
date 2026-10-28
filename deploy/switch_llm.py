"""Switch config.local.yaml's llm section to DeepSeek unless that changes the database identity.

Run in the service directory (where config.local.yaml lives), with the service's own Python:
    read -rsp 'DeepSeek key: ' K; echo; K="$K" .venv/bin/python switch_llm.py
Backs up the old file to config.local.yaml.bak; nothing is changed if the check fails.
The LLM model name is part of the store identity when retrieval.add_notes is on, so an existing
database built with gpt-4o-mini cannot be reused with another model.
"""

import os
import shutil
import sqlite3
import sys
from pathlib import Path

import yaml

from plain_memory.config import load_settings
from plain_memory.store import canonical

src, new = Path("config.local.yaml"), Path("config.new.yaml")
cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
old = (cfg.get("llm") or {}).get("model")
cfg["llm"] = {
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-flash",
    "api_key": os.environ["K"],
    "concurrency": 16,
}
new.write_text(yaml.safe_dump(cfg), encoding="utf-8")
new.chmod(0o600)
s = load_settings(new.resolve())
r = s.retrieval
print(
    f"database={s.database.name} add_notes={r.add_notes} query_expansion={r.query_expansion} "
    f"rerank={r.rerank} llm: {old} -> deepseek-flash"
)
try:
    meta = sqlite3.connect(s.database)
    row = meta.execute("select value from meta where key='identity'").fetchone()
except sqlite3.OperationalError:
    row = None
if row and row[0] != canonical(s.store_identity()):
    new.unlink()
    sys.exit(
        "STOP: existing database was built with a different LLM/notes identity; "
        "config NOT changed. Point `database:` at a new file (it starts empty) or keep the old llm."
    )
shutil.copy(src, "config.local.yaml.bak")
new.replace(src)
print("OK: config.local.yaml updated (backup: config.local.yaml.bak)")
