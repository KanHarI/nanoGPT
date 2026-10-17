"""An explicit active-condition list shared by the sweep and its VM monitor."""

import json
from pathlib import Path


CONDITIONS = ("curriculum", "hard", "mixture")


def active_conditions(directory, default=CONDITIONS):
    path = Path(directory) / "run_plan.json"
    if not path.exists():
        return tuple(default)
    values = json.loads(path.read_text())["conditions"]
    if (not isinstance(values, list) or not values
            or any(value not in CONDITIONS for value in values)
            or len(set(values)) != len(values)):
        raise ValueError("run_plan.json must list unique known active conditions")
    return tuple(values)
