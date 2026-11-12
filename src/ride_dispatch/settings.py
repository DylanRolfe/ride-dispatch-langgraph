"""The one place every threshold is read from: config/settings.json.

The agent (graph.py) and the tests both call load(), so no number is typed anywhere else.
Set RIDE_DISPATCH_SETTINGS to a different JSON file to try other values without editing the real one.
"""
import json
import os
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.json"
REQUIRED = ("rain_pickup_margin_minutes", "max_candidates", "max_status_age_minutes", "max_reason_chars")


def load() -> dict:
    path = Path(os.environ.get("RIDE_DISPATCH_SETTINGS", DEFAULT_PATH))
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        raise KeyError(f"{path} is missing settings: {', '.join(missing)}")
    for k in REQUIRED:
        if not isinstance(data[k], (int, float)) or isinstance(data[k], bool) or data[k] < 0:
            raise ValueError(f"{path}: {k} must be a number that is not negative, got {data[k]!r}")
    return data
