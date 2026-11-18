import math

import json

from pathlib import Path

import re

from datetime import datetime

from statistics import mean, median

import sys

LAYER_TIME = re.compile(
    r"^MEGATRON_LAYER_TIME sample=(\d+) forward_ms=(\S+) backward_ms=(\S+) layer_ms=(\S+)$", re.M
)

TRAIN_STEP = re.compile(
    r"iteration\s+(\d+)/.*?lm loss: (\S+).*?"
    r"number of skipped iterations:\s+(\d+).*?number of nan iterations:\s+(\d+)"
)

STEP_TIME = re.compile(r"iteration\s+(\d+)/.*?elapsed time per iteration \(ms\): (\S+)")

TRAINING_EVIDENCE = "TILELANG_REGION_TRAINING "


def read_run(path, *, steps=11, require_check=True, warmup=3):
    content = path.read_text()
    events = [
        (int(step), float(fw), float(bw), float(ms))
        for step, fw, bw, ms in LAYER_TIME.findall(content)
    ]
    updates = [
        (int(step), float(loss), int(skipped), int(nan))
        for step, loss, skipped, nan in TRAIN_STEP.findall(content)
    ]
    if (
        [step for step, *_ in events] != list(range(steps))
        or [step for step, *_ in updates] != list(range(1, steps + 1))
        or any(not math.isfinite(ms) or ms <= 0 for _, *times in events for ms in times)
        or any(not math.isfinite(loss) or skipped or nan for _, loss, skipped, nan in updates)
        or "[after training is done]" not in content
        or "Traceback (most recent call last)" in content
        or "TILELANG_REGION_PLACEMENT_FALLBACK " in content
        or (require_check and "TILELANG_MLP_CHECK " not in content)
    ):
        raise ValueError(f"incomplete or invalid Megatron run: {path}")
    for grid, stage in (("qkv_grid", "QKV"), ("proj_grid", "PROJ")):
        if re.search(rf"^TILELANG_MLP .*\b{grid}=[1-9]\d*\b", content, re.M):
            if f"TILELANG_{stage}_CHECK " not in content:
                raise ValueError(f"missing TileLang {stage} reference check: {path}")
    return (
        [ms for _, _, _, ms in events],
        [loss for _, loss, _, _ in updates],
        [median(event[column] for event in events[warmup:]) for column in (1, 2, 3)],
    )


def losses_match(reference, other):
    return all(math.isclose(a, b, rel_tol=0.02, abs_tol=0.02) for a, b in zip(reference, other))


def read_threeop_run(path, mode, *, steps, layers, packed_ag, joint_rs=False, resnorm=False):
    _, losses, _ = read_run(path, steps=steps, warmup=min(3, steps - 1))
    log = path.read_text(encoding="utf-8")
    updates = [(int(i), float(t)) for i, t in STEP_TIME.findall(log)]
    if [i for i, _ in updates] != list(range(1, steps + 1)) or any(
        not math.isfinite(t) or t <= 0 for _, t in updates
    ):
        raise ValueError(f"incomplete optimizer step timing: {path}")

    expected_layers = list(range(1, layers + 1))
    checks = ("MLP", "PROJ", "RESNORM") if resnorm else ("MLP", "PROJ")
    for check in checks:
        seen = sorted(map(int, re.findall(rf"^TILELANG_{check}_CHECK layer=(\d+)\s", log, re.M)))
        if seen != expected_layers:
            raise ValueError(f"incomplete {check} checks: {path}")

    installed = re.findall(r"^TILELANG_MLP layer=(\d+) .* packed_ag=([01])$", log, re.M)
    if sorted((int(layer), int(packed)) for layer, packed in installed) != [
        (layer, int(packed_ag)) for layer in expected_layers
    ]:
        raise ValueError(f"wrong AG layout or incomplete layer installation: {path}")
    modes = re.findall(r"^TILELANG_THREEOP_ABLATION mode=(\S+)(?: joint_rs=([01]))?$", log, re.M)
    if [(name, int(joint or "0")) for name, joint in modes] != [(mode, int(joint_rs))] * layers:
        raise ValueError(f"wrong three-op ablation: {path}")
    return [t for _, t in updates], losses


def check_timing_overlap(report, log, warmup):
    events = report.read_text(encoding="utf-8").splitlines()
    if not events:
        return
    samples = re.findall(r"^foreign_gpu_process=\d+ observed_at=(\d+)$", "\n".join(events), re.M)
    if len(samples) != len(events):
        raise ValueError(f"GPU process overlap or incomplete monitoring data: {report}")
    timestamps = [
        (int(step), datetime.fromisoformat(at).timestamp())
        for at, step in re.findall(
            r"^\s*\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]\s+iteration\s+(\d+)/",
            log.read_text(encoding="utf-8"),
            re.M,
        )
    ]
    if not timestamps or warmup not in dict(timestamps):
        raise ValueError(f"missing timed training window: {log}")
    # Megatron prints timestamps to the second, including the last timed step.
    start = dict(timestamps)[warmup]
    end = timestamps[-1][1]
    if any(start <= int(at) <= end for at in samples):
        raise ValueError(f"GPU process overlap during measured training steps: {report}")


def compare_threeop_runs(root, *, steps, warmup, layers, path_scope):
    # The shell launcher uses the same devices, model and input in mirrored order.
    if not 0 <= warmup < steps:
        raise ValueError("expected at least one timed training step")
    for report in root.glob("*.gpu.log"):
        check_timing_overlap(
            report, root / report.name.removesuffix(".gpu.log") / "train.log", warmup
        )
    arms = ("ag", "rs", "ar") if path_scope == "path-singles" else ("ag_rs", "ag_ar", "rs_ar")
    sequence = ("original", *("paths_" + arm for arm in arms), "paths_ag_rs_ar")
    modes = {
        arm: (arm, "ag" in arm.split("_"), False, "ar" in arm.split("_")) for arm in sequence
    }
    order = sequence + sequence[::-1]
    runs = {
        (slot, arm): read_threeop_run(
            root / f"{slot}-{arm}" / "train.log",
            modes[arm][0],
            steps=steps,
            layers=layers,
            packed_ag=modes[arm][1],
            joint_rs=modes[arm][2],
            resnorm=modes[arm][3],
        )
        for slot, arm in enumerate(order, 1)
    }
    anchor_losses = runs[1, order[0]][1]
    if not all(losses_match(anchor_losses, losses) for _, losses in runs.values()):
        raise ValueError(f"mismatched training loss trajectories: {root}")

    repeats = {}
    for slot, arm in enumerate(sequence, 1):
        first, last = runs[slot, arm][0], runs[len(order) - slot + 1, arm][0]
        repeats[arm] = first[warmup:], last[warmup:]
    step_times = {arm: [mean(t) for t in zip(*samples)] for arm, samples in repeats.items()}
    result = {
        "median_ms": {arm: median(t) for arm, t in step_times.items()},
        "repeat_median_ms": {arm: [median(t) for t in samples] for arm, samples in repeats.items()},
    }
    baseline, *parts, combined = sequence
    result["over_original"] = {
        arm: median(a / b for a, b in zip(step_times[baseline], step_times[arm]))
        for arm in (*parts, combined)
    }
    result["combined_over_path"] = {
        arm: median(a / b for a, b in zip(step_times[arm], step_times[combined]))
        for arm in parts
    }
    result["repeat_speedup_over_original"] = {
        arm: [
            median(a / b for a, b in zip(left, right))
            for left, right in zip(repeats[baseline], repeats[arm])
        ]
        for arm in (*parts, combined)
    }
    result["repeat_combined_over_path"] = {
        arm: [
            median(a / b for a, b in zip(left, right))
            for left, right in zip(repeats[arm], repeats[combined])
        ]
        for arm in parts
    }
    return result


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("scope", choices=("path-singles", "path-doubles"))
    p.add_argument("root", type=Path)
    p.add_argument("--steps", type=int, default=201)
    p.add_argument("--warmup", type=int, default=20)
    a = p.parse_args()
    print(
        json.dumps(
            compare_threeop_runs(
                a.root,
                steps=a.steps,
                warmup=a.warmup,
                layers=28,
                path_scope=a.scope,
            ),
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
