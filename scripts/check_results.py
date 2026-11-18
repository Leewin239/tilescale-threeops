"""Recompute the published paired speedups from stored per-update evidence."""

import json
from pathlib import Path
from statistics import median


def main():
    evidence = Path(__file__).resolve().parents[1] / "results" / "long201.json"
    study = json.loads(evidence.read_text(encoding="utf-8"))
    for name, block in study["blocks"].items():
        groups = {}
        for run, record in block["runs"].items():
            updates = record["training"]
            if len(updates) != 201 or any(u["skipped"] or u["nan"] for u in updates):
                raise ValueError(f"Invalid historical run: {run}")
            mode = run.split("-", 1)[1]
            groups.setdefault(mode, []).append([u["step_ms"] for u in updates[20:]])
        paired = {
            mode: [sum(values) / len(values) for values in zip(*runs)]
            for mode, runs in groups.items()
        }
        speedups = {}
        for mode, expected in block["over_original"].items():
            actual = median(a / b for a, b in zip(paired["original"], paired[mode]))
            if abs(actual - expected) > 1e-8:
                raise ValueError(f"Summary mismatch: {name}, {mode}")
            speedups[mode] = actual
        print(json.dumps({"block": name, "verified_speedups": speedups}, sort_keys=True))


if __name__ == "__main__":
    main()
