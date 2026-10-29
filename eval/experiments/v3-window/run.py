r"""Entry point for experiment v3-window (see exp.yaml).

    .venv\Scripts\python eval\experiments\v3-window\run.py --judge deepseek

Flags: --judge qwen-local|gpt-4o-mini  --only locomo|lme  --locomo-per-conv N  --lme-per-type N  --force
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import runner  # noqa: E402

runner.main(default_experiments=[Path(__file__).resolve().parent.name])
