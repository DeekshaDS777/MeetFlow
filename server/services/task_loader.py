from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

TASK_DIR = Path(__file__).resolve().parents[2] / "tasks"


def load_task_bank() -> Dict[str, List[dict]]:
    bank: Dict[str, List[dict]] = {}
    for name in ("easy", "medium", "hard"):
        with open(TASK_DIR / f"{name}.json", "r", encoding="utf-8") as f:
            bank[name] = json.load(f)
    return bank
