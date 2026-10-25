#!/usr/bin/env python
"""Summarise the temporal-branch intervention sweep.

Reads the JSON files written by ``tools/diag_sgta_temporal.py run`` and prints

  * the accuracy of every intervention against the paired ``none`` baseline
    (paired per sequence, plus the naive mean);
  * the numerical floor measured by the ``repeat`` control, so every
    per-frame influence number can be judged against it;
  * the module-level predictor diagnostics.

Everything is paired on the sequence names actually present in every file, so
a partial run cannot silently change the comparison set.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="RGBT_workspace/diag/*_sweep.json")
    parser.add_argument("--baseline", default="none")
    parser.add_argument("--metrics", default="PR,SS,AO")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"no files matched {args.glob}")

    runs: Dict[str, dict] = {}
    for path in paths:
        data = load(path)
        runs[data["intervention"]] = data

    common: List[str] | None = None
    for data in runs.values():
        names = set(data["sequences"])
        common = names if common is None else (common & names)
    common = sorted(common or [])
    if not common:
        raise SystemExit("no sequences shared by all runs")
    if args.baseline not in runs:
        raise SystemExit(f"baseline '{args.baseline}' not found in {sorted(runs)}")

    index = {
        mode: {row["seq_name"]: row for row in data["metric_rows"]}
        for mode, data in runs.items()
    }
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    print("=" * 92)
    print(f"sequences shared by all runs: {len(common)}")
    print("=" * 92)
    header = f"{'intervention':<16}{'n':>4}" + "".join(f"{m + ' mean':>11}{m + ' d':>10}" for m in metrics)
    print(header)
    print("-" * len(header))
    base_rows = index[args.baseline]
    for mode in sorted(runs, key=lambda m: -np.mean([index[m][s]["PR"] for s in common])):
        rows = index[mode]
        cells = ""
        for metric in metrics:
            values = np.array([rows[s][metric] for s in common], dtype=float)
            deltas = np.array([rows[s][metric] - base_rows[s][metric] for s in common], dtype=float)
            cells += f"{values.mean():>11.4f}{deltas.mean():>+10.4f}"
        print(f"{mode:<16}{len(common):>4}{cells}")

    print()
    print("=" * 92)
    print("per-frame influence (input-matched double forward; delta / baseline)")
    print("=" * 92)
    floor = runs["repeat"]["frame_deltas"] if "repeat" in runs else {}
    print(f"numerical floor (repeat control): rel_score_map="
          f"{floor.get('rel_score_map', float('nan')):.3e}  "
          f"box_shift_px={floor.get('box_shift_px', float('nan')):.3e}  n={floor.get('n', 0)}")
    header = f"{'intervention':<16}{'n':>7}{'rel_score_map':>16}{'x floor':>10}{'max_|dSM|':>12}{'box_shift_px':>14}{'x floor':>10}"
    print(header)
    for mode in sorted(runs):
        d = runs[mode]["frame_deltas"]
        if not d:
            continue
        sm = d.get("rel_score_map", float("nan"))
        px = d.get("box_shift_px", float("nan"))
        f_sm = floor.get("rel_score_map", float("nan"))
        f_px = floor.get("box_shift_px", float("nan"))
        print(f"{mode:<16}{d.get('n', 0):>7}{sm:>16.3e}{sm / (f_sm + 1e-30):>10.1f}"
              f"{d.get('max_score_map', float('nan')):>12.3e}{px:>14.3e}{px / (f_px + 1e-30):>10.1f}")

    print()
    print("=" * 92)
    print("module-level predictor diagnostics")
    print("=" * 92)
    keys = ("n", "cos_pred_running_mean", "cos_pred_t2", "cos_pred_copy_t2",
            "rel_dev_from_copy", "cos_pred_zerostate", "rel_pred_zerostate",
            "t2_norm_per_token", "state_norm")
    header = f"{'intervention':<16}" + "".join(f"{k:>21}" for k in keys)
    print(header)
    for mode in sorted(runs):
        d = runs[mode]["predictor_diag"]
        if not d:
            continue
        cells = "".join(f"{d.get(k, float('nan')):>21.6g}" for k in keys)
        print(f"{mode:<16}{cells}")

    out = os.path.join(os.path.dirname(paths[0]), "intervention_summary.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "common_sequences": common,
                "metrics": {
                    mode: {m: float(np.mean([index[mode][s][m] for s in common]))
                           for m in metrics}
                    for mode in runs
                },
                "frame_deltas": {mode: runs[mode]["frame_deltas"] for mode in runs},
                "predictor_diag": {mode: runs[mode]["predictor_diag"] for mode in runs},
            },
            fh, indent=2,
        )
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
