#!/usr/bin/env python3
"""Ablation ladder summary from the stored evaluation histories.

Reads `<results_root>/<ns>/eval_history.csv` (aggregate metrics per epoch) and
merges in `<results_root>/<ns>/official_eval_history.csv` when the namespace has
one (A4 stores its official numbers there rather than inline).

Reports three views:

  1. coverage        - how many epochs each variant has, official vs internal
  2. best-per-variant- the best epoch of each variant (what "take the best run"
                       means) with the official and internal metrics
  3. fixed-epoch     - the same metrics at epochs every variant has reached, so
                       the ladder can be read without epoch selection

Metric vocabulary (LasHeR, 245 sequences):
  oPR / oNPR / oSR  official evaluator (20 px precision, normalised precision,
                    success rate)               -> the numbers to report
  PR / NPS / SS     this repo's own computation   -> available for every run,
                    including A4 which has no per-sequence official values

Usage:
  python tools/analyze_ablation.py
  python tools/analyze_ablation.py --select-metric oSR --out /tmp/ladder.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

RESULTS_ROOT = "RGBT_workspace/results"

# (label, results namespace, checkpoint subdirectory, what the rung adds)
# The order follows the paper (Table 3): M -> S -> R -> L.
VARIANTS: List[Tuple[str, str, str, str]] = [
    ("A0", "lasher_a0", "lasher_ablation_a0",
     "baseline: fixed averaging, static templates, no SGTA module"),
    ("A1", "lasher_a1", "lasher_ablation_a1",
     "+ M: pre-head modality mixing (MoE)"),
    ("A2", "lasher_a2", "lasher_ablation_a2",
     "+ S: evidence-conditioned Mamba-2 spatial enhancement"),
    ("A3", "lasher_a3", "lasher_ablation_a3",
     "+ R: paired template scoring + dynamic template"),
    ("A4", "lasher", "lasher_train",
     "+ L: state-dependent update permission (full model)"),
]

# display name -> CSV column in eval_history.csv
INTERNAL: List[Tuple[str, str]] = [("PR", "seq_PR"), ("NPS", "seq_NPS"), ("SS", "seq_SS")]
OFFICIAL: List[Tuple[str, str]] = [("oPR", "seq_oPR"), ("oNPR", "seq_oNPR"), ("oSR", "seq_oSR")]

_EPOCH_IN_CKPT = re.compile(r"ep(\d+)")
_EPOCH_IN_TAG = re.compile(r"_(\d+)$")


def _as_float(raw: object) -> Optional[float]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _epoch_of(row: Dict[str, str]) -> Optional[int]:
    match = _EPOCH_IN_CKPT.search(row.get("checkpoint", ""))
    if match:
        return int(match.group(1))
    match = _EPOCH_IN_TAG.search(row.get("ckpt_tag", "").strip())
    if match:
        return int(match.group(1))
    return None


def read_eval_history(path: str) -> Dict[int, Dict[str, Optional[float]]]:
    """epoch -> {column: value} for every numeric column we care about."""
    wanted = [column for _, column in INTERNAL + OFFICIAL]
    out: Dict[int, Dict[str, Optional[float]]] = {}
    if not os.path.isfile(path):
        return out
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            epoch = _epoch_of(row)
            if epoch is None:
                continue
            record: Dict[str, Optional[float]] = {}
            for column in wanted:
                record[column] = _as_float(row.get(column))
            record["_tag"] = row.get("ckpt_tag", "") or ""
            out[epoch] = record
    return out


def read_official_history(path: str) -> Dict[int, Dict[str, Optional[float]]]:
    """epoch -> official PR/NPR/SR from a separate official_eval_history.csv."""
    out: Dict[int, Dict[str, Optional[float]]] = {}
    if not os.path.isfile(path):
        return out
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            epoch = _as_float(row.get("epoch"))
            if epoch is None:
                continue
            out[int(epoch)] = {
                "seq_oPR": _as_float(row.get("PR")),
                "seq_oNPR": _as_float(row.get("NPR")),
                "seq_oSR": _as_float(row.get("SR")),
            }
    return out


def load_variant(results_root: str, namespace: str
                 ) -> Dict[int, Dict[str, Optional[float]]]:
    """Merge the inline history with the standalone official history (if any)."""
    base_dir = os.path.join(results_root, namespace)
    history = read_eval_history(os.path.join(base_dir, "eval_history.csv"))
    official = read_official_history(os.path.join(base_dir, "official_eval_history.csv"))
    for epoch, values in official.items():
        record = history.setdefault(epoch, {column: None for _, column in
                                            INTERNAL + OFFICIAL})
        for column, value in values.items():
            # A standalone official history is the more authoritative source.
            if value is not None:
                record[column] = value
    return history


def has_official(history: Dict[int, Dict[str, Optional[float]]]) -> bool:
    return any(record.get("seq_oPR") is not None for record in history.values())


def best_epoch(history: Dict[int, Dict[str, Optional[float]]], column: str
               ) -> Optional[Tuple[int, float]]:
    candidates = [(epoch, record[column]) for epoch, record in history.items()
                  if record.get(column) is not None]
    if not candidates:
        return None
    epoch, value = max(candidates, key=lambda item: item[1])
    return epoch, value


def value_at(history: Dict[int, Dict[str, Optional[float]]], epoch: int, column: str
             ) -> Optional[float]:
    record = history.get(epoch)
    return None if record is None else record.get(column)


def _fmt(value: Optional[float], width: int = 7, digits: int = 4) -> str:
    if value is None:
        return " " * (width - 1) + "-"
    return f"{value:>{width}.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument("--select-metric", default="seq_oPR",
                        help="column used to pick the best epoch (default seq_oPR, "
                             "falls back to seq_PR when a variant has no official numbers)")
    parser.add_argument("--out", default="", help="write the best-per-variant table here")
    parser.add_argument("--epochs", nargs="+", type=int, default=None,
                        help="print the fixed-epoch ladder at exactly these epochs "
                             "(default: first / middle / last common epoch)")
    parser.add_argument("--last-n", type=int, default=3,
                        help="size of the trailing window for the protocol-compliant "
                             "mean (docs/SGTA_ABLATION_PLAN.md sec 7 risk 4: report "
                             "the mean of the last N epochs, never the best epoch)")
    args = parser.parse_args()

    histories: Dict[str, Dict[int, Dict[str, Optional[float]]]] = {}
    for label, namespace, _, _ in VARIANTS:
        histories[label] = load_variant(args.results_root, namespace)

    # ---------------------------------------------------------------- coverage
    print("=" * 104)
    print("1) coverage per variant")
    print("=" * 104)
    print(f"{'rung':<5} {'namespace':<12} {'epochs':>7} {'official':>9} "
          f"{'first':>6} {'last':>6}  {'adds'}")
    print("-" * 104)
    for label, namespace, _, adds in VARIANTS:
        history = histories[label]
        epochs = sorted(history)
        official_epochs = sorted(e for e, r in history.items()
                                 if r.get("seq_oPR") is not None)
        span = f"{epochs[0]}..{epochs[-1]}" if epochs else "-"
        print(f"{label:<5} {namespace:<12} {len(epochs):>7} {len(official_epochs):>9} "
              f"{(epochs[0] if epochs else 0):>6} {(epochs[-1] if epochs else 0):>6}  {adds}")
        if not epochs:
            print(f"      !! no eval rows found under {args.results_root}/{namespace}")
            continue
        if official_epochs and len(official_epochs) != len(epochs):
            print(f"      note: official numbers cover epochs {official_epochs[0]}.."
                  f"{official_epochs[-1]} ({len(official_epochs)}/{len(epochs)})"
                  f"  [span {span}]")

    # ------------------------------------------------- best epoch per variant
    print()
    print("=" * 104)
    print("2) best epoch per variant  (what 'take the best run' would report)")
    print("=" * 104)
    print(f"{'rung':<5} {'select':<8} {'ep':>4} {'oPR':>8} {'oNPR':>8} {'oSR':>8} "
          f"{'|':>2} {'PR':>8} {'NPS':>8} {'SS':>8} {'|':>2} {'last':>6} {'PR@last':>8}")
    print("-" * 104)

    selected: Dict[str, Dict[str, object]] = {}
    for label, namespace, _, _ in VARIANTS:
        history = histories[label]
        if not history:
            continue
        column = args.select_metric
        if not has_official(history):
            column = "seq_PR"  # no official numbers -> fall back to internal PR
        picked = best_epoch(history, column)
        if picked is None:
            continue
        epoch, _ = picked
        last_epoch = max(history)
        record = history[epoch]
        last_record = history[last_epoch]
        print(f"{label:<5} {'official' if has_official(history) else 'internal':<8} "
              f"{epoch:>4} {_fmt(record.get('seq_oPR'))} {_fmt(record.get('seq_oNPR'))} "
              f"{_fmt(record.get('seq_oSR'))} {'|':>2} {_fmt(record.get('seq_PR'))} "
              f"{_fmt(record.get('seq_NPS'))} {_fmt(record.get('seq_SS'))} {'|':>2} "
              f"{last_epoch:>6} {_fmt(last_record.get('seq_PR'))}")
        selected[label] = {
            "namespace": namespace,
            "criterion": column,
            "epoch": epoch,
            **{column_name: record.get(column_name)
               for _, column_name in INTERNAL + OFFICIAL},
            "last_epoch": last_epoch,
            "last_oPR": last_record.get("seq_oPR"),
            "last_PR": last_record.get("seq_PR"),
        }

    # also: best by official SR, to see whether the same epoch wins
    print()
    print(f"{'rung':<5} best-by-oPR: ep / value    best-by-oSR: ep / value    "
          f"agree?   oPR spread over epochs (best-median-last)")
    print("-" * 104)
    for label, _, _, _ in VARIANTS:
        history = histories[label]
        if not history:
            continue
        by_pr = best_epoch(history, "seq_oPR")
        by_sr = best_epoch(history, "seq_oSR")
        if by_pr is None:
            by_pr = best_epoch(history, "seq_PR")
            by_sr = best_epoch(history, "seq_SS")
            tag = "(internal PR/SS)"
        else:
            tag = "(official)"
        values = sorted((r[("seq_oPR" if tag.startswith("(official") else "seq_PR")]
                         for r in history.values()
                         if r.get("seq_oPR" if tag.startswith("(official") else "seq_PR")
                         is not None))
        median = values[len(values) // 2] if values else float("nan")
        agree = "yes" if by_pr and by_sr and by_pr[0] == by_sr[0] else "no"
        print(f"{label:<5} {tag:<13} ep{by_pr[0]:>3} / {by_pr[1]:.4f}        "
              f"ep{by_sr[0]:>3} / {by_sr[1]:.4f}        {agree:<8} "
              f"{values[-1] - median:+.4f} / {values[-1] - values[0]:+.4f} "
              f"(min {values[0]:.4f}, med {median:.4f}, max {values[-1]:.4f}, n={len(values)})")

    # ------------------------------- protocol-compliant trailing-window mean
    print()
    print("=" * 104)
    print(f"2b) protocol-compliant view: mean of the LAST {args.last_n} epochs "
          f"(docs/SGTA_ABLATION_PLAN.md sec 7 risk 4)")
    print("=" * 104)
    columns = [column for _, column in INTERNAL + OFFICIAL]
    print(f"{'rung':<5} {'window':<12} {'oPR':>8} {'oNPR':>8} {'oSR':>8} {'|':>2} "
          f"{'PR':>8} {'NPS':>8} {'SS':>8}")
    print("-" * 104)
    trailing: Dict[str, Dict[str, Optional[float]]] = {}
    for label, _, _, _ in VARIANTS:
        history = histories[label]
        if not history:
            continue
        epochs = sorted(history)[-max(1, args.last_n):]
        means: Dict[str, Optional[float]] = {}
        for column in columns:
            values = [history[e].get(column) for e in epochs]
            values = [v for v in values if v is not None]
            means[column] = sum(values) / len(values) if values else None
        trailing[label] = means
        window = f"{epochs[0]}-{epochs[-1]}"
        print(f"{label:<5} {window:<12} {_fmt(means.get('seq_oPR'))} "
              f"{_fmt(means.get('seq_oNPR'))} {_fmt(means.get('seq_oSR'))} {'|':>2} "
              f"{_fmt(means.get('seq_PR'))} {_fmt(means.get('seq_NPS'))} "
              f"{_fmt(means.get('seq_SS'))}")

    # ------------------------------------------------------- fixed-epoch views
    common = None
    for label, _, _, _ in VARIANTS:
        epochs = set(histories[label])
        common = epochs if common is None else (common & epochs)
    common = sorted(common or [])
    if common:
        print()
        print("=" * 104)
        print(f"3) fixed-epoch ladder (epochs every variant has: {common[0]}..{common[-1]}, "
              f"n={len(common)})")
        print("=" * 104)
        for target in (args.epochs if args.epochs else
                       (common[0], common[len(common) // 2], common[-1])):
            print(f"-- epoch {target} --")
            print(f"   {'rung':<5} {'oPR':>8} {'oNPR':>8} {'oSR':>8} {'|':>2} "
                  f"{'PR':>8} {'NPS':>8} {'SS':>8}")
            for label, _, _, _ in VARIANTS:
                history = histories[label]
                record = history.get(target)
                if record is None:
                    print(f"   {label:<5} {'missing':>8}")
                    continue
                print(f"   {label:<5} {_fmt(record.get('seq_oPR'))} "
                      f"{_fmt(record.get('seq_oNPR'))} {_fmt(record.get('seq_oSR'))} "
                      f"{'|':>2} {_fmt(record.get('seq_PR'))} "
                      f"{_fmt(record.get('seq_NPS'))} {_fmt(record.get('seq_SS'))}")
            print()

    if args.out and selected:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        fields = ["rung", "namespace", "criterion", "epoch",
                  *[column for _, column in INTERNAL + OFFICIAL],
                  "last_epoch", "last_oPR", "last_PR",
                  *[f"last{args.last_n}_{column}" for column in columns]]
        with open(args.out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for label, _, _, _ in VARIANTS:
                entry = selected.get(label)
                if not entry:
                    continue
                trailing_means = trailing.get(label, {})
                writer.writerow({
                    "rung": label, **entry,
                    **{f"last{args.last_n}_{column}": trailing_means.get(column)
                       for column in columns},
                })
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
