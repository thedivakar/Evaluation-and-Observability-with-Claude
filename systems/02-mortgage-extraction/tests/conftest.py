"""Pytest config: load .env so recording tests can hit the live API on first run."""
from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv() -> None:
    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()
