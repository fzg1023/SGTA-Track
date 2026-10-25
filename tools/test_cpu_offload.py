#!/usr/bin/env python
"""Verify that the CPU offload / restore round-trip is numerically exact.

A bug in this path would only show up hours into a multi-day run (at the first
per-epoch evaluation), so it is checked here on a toy model instead:

  * the loss trajectory of a run with an offload/restore in the middle must
    match the run without one;
  * parameters must really be on CPU while parked, and back on CUDA after;
  * device memory allocated by the model must drop while parked.

Run:  python tools/test_cpu_offload.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lib.train.trainers.base_trainer import (  # noqa: E402
    offload_training_state,
    restore_training_state,
)

STEPS = 8
PARK_AFTER = 3


def make_model():
    return nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
    )


def run(with_offload: bool):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model = make_model().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(32, 64, device="cuda")
    targets = torch.randn(32, 1, device="cuda")

    losses = []
    facts = {}
    for step in range(STEPS):
        output = model(inputs)
        loss = ((output - targets) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()

        if with_offload and step == PARK_AFTER:
            before = torch.cuda.memory_allocated()
            # the invariant is "every tensor is back where it was", not
            # "everything is on CUDA": AdamW keeps some bookkeeping tensors on
            # CPU by construction
            opt_before = [
                v.device
                for state in optimizer.state.values()
                for v in state.values()
                if torch.is_tensor(v)
            ]
            token = offload_training_state(model, optimizer)
            facts["params_on_cpu"] = all(p.device.type == "cpu" for p in model.parameters())
            facts["buffers_on_cpu"] = all(b.device.type == "cpu" for b in model.buffers())
            facts["opt_state_on_cpu"] = all(
                v.device.type == "cpu"
                for state in optimizer.state.values()
                for v in state.values()
                if torch.is_tensor(v)
            )
            facts["opt_state_tensors"] = len(opt_before)
            facts["opt_state_devices_before"] = sorted({str(d) for d in opt_before})
            parked = torch.cuda.memory_allocated()
            facts["memory_freed_mb"] = (before - parked) / 2 ** 20
            restore_training_state(model, optimizer, token)
            facts["params_on_cuda_after"] = all(p.device.type == "cuda" for p in model.parameters())
            opt_after = [
                v.device
                for state in optimizer.state.values()
                for v in state.values()
                if torch.is_tensor(v)
            ]
            facts["opt_state_devices_after"] = sorted({str(d) for d in opt_after})
            facts["opt_state_devices_restored"] = (
                len(opt_before) == len(opt_after)
                and all(a == b for a, b in zip(opt_before, opt_after))
            )

        optimizer.step()
        losses.append(loss.item())
    return losses, facts


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available; cannot run this test")
        raise SystemExit(2)

    base, _ = run(with_offload=False)
    test, facts = run(with_offload=True)

    print("losses without offload:", " ".join(f"{v:.8f}" for v in base))
    print("losses with offload   :", " ".join(f"{v:.8f}" for v in test))
    worst = max(abs(a - b) for a, b in zip(base, test))
    print(f"max |difference|      : {worst:.3e}")
    print("facts                 :", facts)

    checks = [
        ("trajectory identical (< 1e-6)", worst < 1e-6),
        ("parameters parked on CPU", facts.get("params_on_cpu") is True),
        ("buffers parked on CPU", facts.get("buffers_on_cpu") is True),
        ("optimizer state parked on CPU", facts.get("opt_state_on_cpu") is True),
        ("every optimizer tensor back on its original device",
         facts.get("opt_state_devices_restored") is True),
        ("parameters restored to CUDA", facts.get("params_on_cuda_after") is True),
        ("device memory actually freed", facts.get("memory_freed_mb", 0) > 0),
        ("optimizer state existed", facts.get("opt_state_tensors", 0) > 0),
    ]
    print()
    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print("\nRESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
