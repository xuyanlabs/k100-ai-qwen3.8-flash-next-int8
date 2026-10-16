#!/usr/bin/env python3
# repro_moe_realw.py — 用真实 checkpoint 权重验证 MoE int8 全链路数值
# 卡4单卡: 加载 layer0 全部 512 专家的 gate/up/down int8+scale, 走 fused_experts
# 对照 = CPU fp32 反量化参考(只对每个 token 的 top10 专家算, 省内存)
import json
import os
import sys

import torch

M = int(sys.argv[1]) if len(sys.argv) > 1 else 4
LAYER = 0
H = 2560
NI = 640
E = 512
TOPK = 10
IDX = "/data2/qwen38-int8-w8a8/model.safetensors.index.json"
DIR = "/data2/qwen38-int8-w8a8"

from safetensors import safe_open

print("加载索引...", flush=True)
idx = json.load(open(IDX))["weight_map"]
prefix = f"model.language_model.layers.{LAYER}.mlp."

# 找 gate(路由) 权重键名
gate_keys = [k for k in idx if k.startswith(prefix) and "mlp.gate" in k]
print("router 键:", gate_keys[:4], flush=True)

# 收集 expert 键
e0 = [k for k in idx if k.startswith(prefix + "experts.0.")]
print("expert.0 键:", sorted(e0), flush=True)


def load_tensor(key):
    f = safe_open(os.path.join(DIR, idx[key]), framework="pt")
    return f.get_tensor(key)


print("加载 router gate ...", flush=True)
gw = load_tensor(f"{prefix}gate.weight")          # bf16 [512, 2560] (router 不量化)
print(f"gate: {gw.shape} {gw.dtype} (无 scale, bf16)", flush=True)

print("加载 512 专家 (约 2.5GB int8)...", flush=True)
w1q = torch.empty(E, 2 * NI, H, dtype=torch.int8)   # gate_proj;up_proj
w2q = torch.empty(E, H, NI, dtype=torch.int8)       # down_proj
w1s = torch.empty(E, 2 * NI, 1, dtype=torch.float32)
w2s = torch.empty(E, H, 1, dtype=torch.float32)
for e in range(E):
    g = load_tensor(f"{prefix}experts.{e}.gate_proj.weight")
    gs = load_tensor(f"{prefix}experts.{e}.gate_proj.weight_scale")
    u = load_tensor(f"{prefix}experts.{e}.up_proj.weight")
    us = load_tensor(f"{prefix}experts.{e}.up_proj.weight_scale")
    d = load_tensor(f"{prefix}experts.{e}.down_proj.weight")
    ds = load_tensor(f"{prefix}experts.{e}.down_proj.weight_scale")
    w1q[e, :NI], w1q[e, NI:] = g, u
    w1s[e, :NI], w1s[e, NI:] = gs, us
    w2q[e], w2s[e] = d, ds
    if e % 128 == 0:
        print(f"  expert {e}/512", flush=True)
print("加载完毕", flush=True)

torch.manual_seed(7)
hidden = torch.randn(M, H, dtype=torch.bfloat16) * 0.5

# ---- GPU 路径: 引擎同款 ----
dev = "cuda"
print("GPU fused_experts ...", flush=True)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_quant_int8,
)

hq, hs_ = per_token_quant_int8(hidden.to(dev))
gate_bf16 = gw.to(torch.bfloat16).to(dev)
router_logits = torch.nn.functional.linear(hidden.to(dev), gate_bf16)
scores = torch.sigmoid(router_logits.float())
tw, ti = torch.topk(scores, TOPK, dim=-1)
tw = tw / (tw.sum(dim=-1, keepdim=True) + 1e-20)
print(f"路由 top{TOPK}: ids[0]={ti[0].tolist()}", flush=True)

qc = FusedMoEQuantConfig.make(
    quant_dtype=torch.int8,
    per_act_token_quant=True,
    per_out_ch_quant=True,
    w1_scale=w1s.to(dev),
    w2_scale=w2s.to(dev),
)
out_gpu = fused_experts(
    hidden.to(dev), w1q.to(dev), w2q.to(dev),
    tw.to(torch.bfloat16), ti.to(torch.int32),
    activation=MoEActivation.SILU,
    quant_config=qc,
)
torch.cuda.synchronize()
print("GPU 完成", flush=True)

# ---- CPU fp32 参考 ----
print("CPU fp32 参考 (仅 top10 专家)...", flush=True)
out_ref = torch.zeros(M, H, dtype=torch.float32)
for m in range(M):
    for j in range(TOPK):
        e = int(ti[m, j])
        g = (w1q[e, :NI].float() * w1s[e, :NI])
        u = (w1q[e, NI:].float() * w1s[e, NI:])
        d = (w2q[e].float() * w2s[e])
        x = hidden[m].float()
        gu = torch.nn.functional.linear(x, g)
        uu = torch.nn.functional.linear(x, u)
        act = torch.nn.functional.silu(gu) * uu
        o = torch.nn.functional.linear(act, d)
        out_ref[m] += float(tw[m, j]) * o
print("参考完成", flush=True)

err = (out_gpu.float().cpu() - out_ref).abs()
rel = (err / out_ref.abs().clamp(min=1e-2)).max()
print(f"OUT_GPU[0,:4]={out_gpu[0,:4].float().cpu().tolist()}")
print(f"OUT_REF [0,:4]={out_ref[0,:4].tolist()}")
print(f"MAX_ABS={err.max().item():.5f} REF_ABSMAX={out_ref.abs().max().item():.3f} MAX_REL={rel.item():.4f}")
ok = err.max().item() < 0.05 * max(out_ref.abs().max().item(), 1.0)
print("MOE_REALW_PASS" if ok else "MOE_REALW_FAIL", flush=True)
