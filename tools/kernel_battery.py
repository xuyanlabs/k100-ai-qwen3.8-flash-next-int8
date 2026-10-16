#!/usr/bin/env python3
# kernel_battery.py — 逐个验证 vLLM 热点 _C 算子是否在 gfx928 上装死
import torch
import torch.nn.functional as F

import vllm._C  # noqa: F401

dev = "cuda"
torch.manual_seed(0)
M, N = 8, 256
eps = 1e-6


def rep(name, got, ref, extra=""):
    d = (got.float() - ref.float()).abs().max().item()
    scale = max(ref.float().abs().max().item(), 1e-6)
    print(f"[{name}] maxdiff={d:.6g} ref_absmax={scale:.4g} rel={d / scale:.4g} {extra}",
          flush=True)


print("### 1) rms_norm ###", flush=True)
x = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
w = torch.randn(N, device=dev, dtype=torch.bfloat16)
xf, wf = x.float(), w.float()
ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * wf
out = torch.zeros_like(x)
try:
    torch.ops._C.rms_norm(out, x, w, eps)
    torch.cuda.synchronize()
    rep("rms_norm(out,x,w,eps)", out, ref, f"out_absmax={out.float().abs().max().item():.4g}")
except Exception as e:
    print("[rms_norm] EXC", type(e).__name__, str(e)[:200], flush=True)

print("### 2) fused_add_rms_norm ###", flush=True)
a = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
r = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
a2, r2 = a.clone(), r.clone()
res = a2.float() + r2.float()
ref_out = res * torch.rsqrt(res.pow(2).mean(-1, keepdim=True) + eps) * wf
try:
    torch.ops._C.fused_add_rms_norm(a2, r2, w, eps)
    torch.cuda.synchronize()
    rep("fused_add_rms_norm: x_out", a2, ref_out)
    rep("fused_add_rms_norm: residual", r2, res)
    print(f"    x unchanged? {bool(torch.equal(a2, a))}  residual unchanged? "
          f"{bool(torch.equal(r2, r))}", flush=True)
except Exception as e:
    print("[fused_add_rms_norm] EXC", type(e).__name__, str(e)[:200], flush=True)

print("### 3) rotary_embedding (no-op 检测) ###", flush=True)
try:
    head_size, num_heads, num_tokens_, max_pos, base = 128, 8, 16, 512, 10000.0
    rotary_dim = head_size
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32,
                                             device=dev) / rotary_dim))
    t = torch.arange(max_pos, dtype=torch.float32, device=dev)
    freqs = torch.outer(t, inv_freq)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.float32)
    pos = torch.randint(0, max_pos, (num_tokens_,), device=dev, dtype=torch.int64)
    q = torch.randn(num_tokens_, num_heads * head_size, device=dev, dtype=torch.bfloat16)
    k = torch.randn(num_tokens_, num_heads * head_size, device=dev, dtype=torch.bfloat16)
    q0, k0 = q.clone(), k.clone()

    def ref_rope(t_, heads, hs, cache_, positions, is_neox=True):
        tt = t_.float().view(-1, heads, hs)
        half = hs // 2
        c = cache_[positions][:, :half].unsqueeze(1)   # [T,1,half]
        s = cache_[positions][:, half:].unsqueeze(1)
        o = torch.empty_like(tt)
        o[:, :, :half] = tt[:, :, :half] * c - tt[:, :, half:] * s
        o[:, :, half:] = tt[:, :, half:] * c + tt[:, :, :half] * s
        return o.view(-1, heads * hs)

    torch.ops._C.rotary_embedding(pos, q, k, head_size, cache, True)
    torch.cuda.synchronize()
    rep("rotary_embedding: q", q, ref_rope(q0, num_heads, head_size, cache, pos))
    rep("rotary_embedding: k", k, ref_rope(k0, num_heads, head_size, cache, pos))
    print(f"    q unchanged? {bool(torch.equal(q, q0))}  "
          f"k unchanged? {bool(torch.equal(k, k0))}", flush=True)
except Exception as e:
    print("[rotary_embedding] EXC", type(e).__name__, str(e)[:250], flush=True)

print("### 4) gelu_and_mul / gelu_tanh_and_mul ###", flush=True)
gx = torch.randn(64, 2 * 128, device=dev, dtype=torch.bfloat16)
gg, gu = gx[:, :128], gx[:, 128:]
for nm, fn, ref_fn in (
    ("gelu_and_mul", "_C", lambda: F.gelu(gg.float()) * gu.float()),
    ("gelu_tanh_and_mul", "_C", lambda: F.gelu(gg.float(), approximate="tanh") * gu.float()),
):
    try:
        o = torch.zeros(64, 128, device=dev, dtype=torch.bfloat16)
        getattr(torch.ops._C, nm)(o, gx)
        torch.cuda.synchronize()
        rep(nm, o, ref_fn(), f"nz={int((o != 0).sum())}/{o.numel()}")
    except Exception as e:
        print(f"[{nm}] EXC", type(e).__name__, str(e)[:200], flush=True)

print("### 5) moe_sum (确认已修) ###", flush=True)
inp = torch.randn(4, 10, 256, device=dev, dtype=torch.bfloat16)
o = torch.zeros(4, 256, device=dev, dtype=torch.bfloat16)
from vllm import _custom_ops as cops
cops.moe_sum(inp, o)
torch.cuda.synchronize()
rep("moe_sum", o, inp.float().sum(dim=1))

print("BATTERY_DONE", flush=True)
