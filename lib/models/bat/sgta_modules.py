import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba, Mamba2
from timm.models.layers import DropPath, Mlp


class TemplateRouter(nn.Module):
    def __init__(self, L, C):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(C, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, template1, template2):
        template1 = template1.permute(0, 2, 1)
        template2 = template2.permute(0, 2, 1)
        template1 = self.gap(template1.unsqueeze(2)).view(template1.size(0), -1)
        template2 = self.gap(template2.unsqueeze(2)).view(template2.size(0), -1)
        scores = torch.cat([self.fc(template1), self.fc(template2)], dim=1)
        return scores, self.softmax(scores)


def build_mamba_layer(dim, version=2, d_state=64, d_conv=4, expand=2, headdim=64):
    if int(version) == 2:
        return Mamba2(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )
    return Mamba(
        d_model=dim,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
    )


def _mean_evidence(attn):
    if attn.dim() == 4:
        return attn.mean(1).mean(1)
    if attn.dim() == 3:
        return attn.mean(1)
    return attn


class SpatialMambaBlock(nn.Module):
    def __init__(self, dim, version=2, d_state=64, d_conv=4, expand=2,
                 headdim=64, drop_path=0.0, mlp_ratio=2.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba_row = build_mamba_layer(dim, version, d_state, d_conv, expand, headdim)
        self.mamba_col = build_mamba_layer(dim, version, d_state, d_conv, expand, headdim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=0.0)

    def forward(self, tokens):
        bsz, length, channels = tokens.shape
        side = int(math.sqrt(length))
        residual = tokens
        x = self.norm(tokens)

        if side * side == length:
            feat = x.view(bsz, side, side, channels)
            row_tokens = feat.reshape(bsz, length, channels)
            row_out = self.mamba_row(row_tokens)

            col_tokens = feat.transpose(1, 2).reshape(bsz, length, channels)
            col_out = self.mamba_col(col_tokens).view(bsz, side, side, channels).transpose(1, 2)
            col_out = col_out.reshape(bsz, length, channels)
            x = 0.5 * (row_out + col_out)
        else:
            x = self.mamba_row(x)

        x = residual + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class UncertaintyGuidedMambaFusion(nn.Module):
    def __init__(self, dim, version=2, d_state=64, d_conv=4, expand=2,
                 headdim=64, drop_path=0.0):
        super().__init__()
        self.softplus = nn.Softplus()
        self.pre_fuse = nn.Linear(dim * 4 + 3, dim)
        self.spatial_mamba = SpatialMambaBlock(
            dim, version=version, d_state=d_state, d_conv=d_conv,
            expand=expand, headdim=headdim, drop_path=drop_path)
        self.gate = nn.Sequential(
            nn.LayerNorm(dim * 3 + 3),
            nn.Linear(dim * 3 + 3, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat_rgb, feat_tir, attn_rgb, attn_tir):
        evidence_rgb = _mean_evidence(attn_rgb)
        evidence_tir = _mean_evidence(attn_tir)

        alpha_rgb = self.softplus(evidence_rgb) + 1.0
        alpha_tir = self.softplus(evidence_tir) + 1.0
        s_rgb = torch.sum(alpha_rgb, dim=-1, keepdim=True)
        s_tir = torch.sum(alpha_tir, dim=-1, keepdim=True)

        u_rgb = feat_rgb.size(1) / (s_rgb + 1e-6)
        u_tir = feat_tir.size(1) / (s_tir + 1e-6)
        u_fused = 2 * u_rgb * u_tir / (u_rgb + u_tir + 1e-6)

        weights = 0.5 * (alpha_rgb + alpha_tir)
        min_v = weights.amin(dim=-1, keepdim=True)
        max_v = weights.amax(dim=-1, keepdim=True)
        weights = 0.01 + (weights - min_v) / (max_v - min_v + 1e-6) * 0.99

        w_rgb = u_fused / (u_rgb + 1e-6)
        w_tir = u_fused / (u_tir + 1e-6)
        uncertainty = torch.cat([u_rgb, u_tir, u_fused], dim=-1)
        uncertainty_tokens = uncertainty.unsqueeze(1).expand(-1, feat_rgb.size(1), -1)

        base_fused = (feat_rgb * w_rgb.unsqueeze(-1) + feat_tir * w_tir.unsqueeze(-1))
        base_fused = base_fused * weights.unsqueeze(-1)
        mamba_in = torch.cat(
            [feat_rgb, feat_tir, feat_rgb - feat_tir, base_fused, uncertainty_tokens],
            dim=-1,
        )
        mamba_in = self.pre_fuse(mamba_in)
        mamba_out = self.spatial_mamba(mamba_in)

        gate_in = torch.cat([feat_rgb, feat_tir, mamba_out, uncertainty_tokens], dim=-1)
        gate = self.gate(gate_in)
        fused = gate * mamba_out + (1.0 - gate) * base_fused
        return fused, [u_fused.unsqueeze(-1), u_rgb.unsqueeze(-1), u_tir.unsqueeze(-1)], weights


class TemporalMambaMemory(nn.Module):
    def __init__(self, dim, version=2, d_state=64, d_conv=4, expand=2,
                 headdim=64, memory_tokens=64, loss_weight=1.0):
        super().__init__()
        self.memory_tokens = memory_tokens
        self.loss_weight = loss_weight
        self.norm = nn.LayerNorm(dim)
        self.mamba = build_mamba_layer(dim, version, d_state, d_conv, expand, headdim)
        self.pred = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.state_proj = nn.Linear(dim, dim)

    def forward(self, t1, t2, target=None, state=None, train=True):
        if state is not None and state.shape[0] == t1.shape[0] and state.shape[-1] == t1.shape[-1]:
            tokens = torch.cat([state, t1, t2], dim=1)
        elif train:
            # warm-up pass: summarize t1 into a pseudo memory state so the
            # training-time token layout matches test time. No detach here:
            # gradients flow through state_proj back into the first scan
            # (fixes state_proj never receiving any gradient).
            out0 = self.mamba(self.norm(t1))
            state = self.state_proj(out0[:, -self.memory_tokens:])
            tokens = torch.cat([state, t1, t2], dim=1)
        else:
            tokens = torch.cat([t1, t2], dim=1)

        out = self.mamba(self.norm(tokens))
        pred = F.normalize(t2 + self.pred(out[:, -t2.shape[1]:]), dim=-1)
        # NOTE: no .detach() here. The tracker detaches the whole
        # dynamic_template dict at test time (detach_state in
        # vit_ce_adapter.forward); during training the state is unused.
        new_state = self.state_proj(out[:, -self.memory_tokens:])

        if target is not None and train:
            target = F.normalize(target, dim=-1)
            mse = F.mse_loss(pred, target)
            cosine = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
            loss = self.loss_weight * (mse + 0.1 * cosine)
        else:
            loss = pred.new_tensor(0.0)
        return pred, loss, new_state


class LiquidTemplateUpdateController(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.tau_proj = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, features, state=None, dt=1.0):
        if state is None or state.shape[0] != features.shape[0]:
            state = torch.zeros(features.shape[0], self.hidden_proj.in_features,
                                device=features.device, dtype=features.dtype)

        tau = F.softplus(self.tau_proj(features)) + 1e-3
        candidate = torch.tanh(self.input_proj(features) + self.hidden_proj(state))
        decay = torch.exp(-torch.as_tensor(dt, device=features.device, dtype=features.dtype) / tau)
        new_state = decay * state + (1.0 - decay) * candidate

        logits = self.out(new_state)
        update_gate = torch.sigmoid(logits[:, :1])
        modal_gate = torch.softmax(logits[:, 1:3], dim=-1)
        fuse_gate = torch.sigmoid(logits[:, 3:4])
        motion_scale = torch.tanh(logits[:, 4:5])
        return {
            "update_gate": update_gate,
            "modal_gate": modal_gate,
            "fuse_gate": fuse_gate,
            "motion_scale": motion_scale,
            "state": new_state.detach(),
        }


class MoEWhenToFuseDecision(nn.Module):
    def __init__(self, dim, hidden_dim=256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(dim * 3 + 3),
            nn.Linear(dim * 3 + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, rgb_tokens, tir_tokens, fused_tokens, uncertainty=None):
        rgb_pool = rgb_tokens.mean(dim=1)
        tir_pool = tir_tokens.mean(dim=1)
        fused_pool = fused_tokens.mean(dim=1)
        if uncertainty is None:
            uncertainty = rgb_pool.new_zeros(rgb_pool.shape[0], 3)
        gate_in = torch.cat([rgb_pool, tir_pool, fused_pool, uncertainty], dim=-1)
        gates = torch.softmax(self.gate(gate_in), dim=-1)
        out = (
            gates[:, 0:1, None] * rgb_tokens
            + gates[:, 1:2, None] * tir_tokens
            + gates[:, 2:3, None] * fused_tokens
        )
        return out, gates
