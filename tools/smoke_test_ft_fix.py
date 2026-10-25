"""Smoke test for the train/test mismatch fix.

Verifies that after the fix:
  1. TemporalMambaMemory.state_proj receives gradient in training mode
     (pseudo state from the warm-up pass is NOT detached).
  2. LiquidTemplateUpdateController.hidden_proj.weight receives gradient
     when a second pass is run with a non-zero detached state.
Also checks the test-time (stateful) path still runs.

Run: CUDA_VISIBLE_DEVICES=0 python tools/smoke_test_ft_fix.py
"""
import torch

from lib.models.bat.sgta_modules import (
    TemporalMambaMemory,
    LiquidTemplateUpdateController,
)

torch.manual_seed(0)
B, L, C = 4, 64, 768
dev = "cuda"
t1 = torch.randn(B, L, C, device=dev)
t2 = torch.randn(B, L, C, device=dev)
t3 = torch.randn(B, L, C, device=dev)

# ---- 1. TemporalMambaMemory ----
m = TemporalMambaMemory(C, memory_tokens=64, loss_weight=1.0).to(dev)

# training mode, state=None -> warm-up pseudo state path
pred, loss, state = m(t1, t2, target=t3, train=True)
print("TMM train: pred", tuple(pred.shape), "state", tuple(state.shape), "loss", float(loss))
assert m.state_proj.weight.grad is None  # before backward
loss.backward()
g = m.state_proj.weight.grad
print("TMM state_proj.weight.grad after backward:", None if g is None else "NOT None, mean|g|=%.3g" % float(g.abs().mean()))

# test mode, state carried across frames (layout must match train: [state, t1, t2])
with torch.no_grad():
    pred2, loss2, state2 = m(t1, t2, state=state, train=False)
print("TMM test : pred", tuple(pred2.shape), "state", tuple(state2.shape))

# ---- 2. LiquidTemplateUpdateController ----
liq = LiquidTemplateUpdateController(C * 3 + 6, hidden_dim=64).to(dev)
feats = torch.randn(B, C * 3 + 6, device=dev)

out1 = liq(feats)                      # first pass: zero state (hidden_proj gets no grad)
out2 = liq(feats, state=out1["state"])  # second pass: non-zero detached state
g1 = None if liq.hidden_proj.weight.grad is None else liq.hidden_proj.weight.grad.clone()
(0.01 * out2["update_gate"].mean()).backward()
g2 = liq.hidden_proj.weight.grad
print("Liquid hidden_proj.weight.grad after backward:",
      None if g2 is None else "NOT None, mean|g|=%.3g" % float(g2.abs().mean()))

assert m.state_proj.weight.grad is not None, "state_proj still receives no gradient!"
assert g2 is not None, "hidden_proj.weight still receives no gradient!"
print("ALL CHECKS PASSED")
