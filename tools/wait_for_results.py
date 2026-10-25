#!/usr/bin/env python3
"""Block until a tracking run has produced results for every expected sequence.

Long RGBT evaluations run for hours in tmux. This helper waits for the per
sequence ``.txt`` files that ``RGBT_workspace/test_sgta.py`` writes and exits
once each requested dataset is complete, so a following evaluation step (or a
`wait && ...` chain) can be started without guessing at timing.

Usage:
    python tools/wait_for_results.py --tracker_tag lasher_train_15 \
        --datasets gtot rgbt210 rgbt234 --interval 60 --timeout 21600

Exit status is 1 on timeout, 2 on configuration errors, else 0.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

PRJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# dataset -> number of sequences in the official toolkit sequence list
EXPECTED_SEQS = {"gtot": 50, "rgbt210": 210, "rgbt234": 234, "lasher": 245}


def count_results(result_dir: str) -> int:
    if not os.path.isdir(result_dir):
        return 0
    return sum(1 for name in os.listdir(result_dir) if name.endswith(".txt"))


def main() -> int:
    parser = argparse.ArgumentParser("wait for tracking results")
    parser.add_argument("--result_root", default=os.path.join(PRJ_ROOT, "RGBT_workspace", "results"))
    parser.add_argument("--tracker_tag", required=True)
    parser.add_argument("--datasets", nargs="+", required=True, choices=sorted(EXPECTED_SEQS))
    parser.add_argument("--interval", type=float, default=60.0, help="poll interval in seconds")
    parser.add_argument("--timeout", type=float, default=21600.0, help="give up after N seconds")
    parser.add_argument("--quiet", action="store_true", help="only print the final line")
    args = parser.parse_args()

    targets = {key: EXPECTED_SEQS[key] for key in args.datasets}
    started = time.time()

    while True:
        state = {}
        for key, expected in targets.items():
            done = count_results(os.path.join(args.result_root, key, args.tracker_tag))
            state[key] = (done, expected)

        if not args.quiet:
            stamp = time.strftime("%H:%M:%S")
            parts = " ".join(f"{key}={done}/{exp}" for key, (done, exp) in state.items())
            print(f"[{stamp}] {parts}", flush=True)

        if all(done >= exp for done, exp in state.values()):
            total = sum(done for done, _ in state.values())
            print(f"[DONE] all datasets complete ({total} sequences): "
                  + ", ".join(f"{k}={v[0]}" for k, v in state.items()), flush=True)
            return 0

        if args.timeout and time.time() - started > args.timeout:
            print(f"[TIMEOUT] after {args.timeout:.0f}s: "
                  + ", ".join(f"{k}={v[0]}/{v[1]}" for k, v in state.items()), flush=True)
            return 1

        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
