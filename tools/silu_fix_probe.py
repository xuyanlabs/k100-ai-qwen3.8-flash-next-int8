#!/usr/bin/env python3
# silu_fix_probe.py — 正确 Triton silu_and_mul + 尝试全局覆盖 _C.silu_and_mul
import torch
import torch.nn.functional as F

import vllm._C  # noqa: F401

dev = "cuda"
torch.manual_seed(1234)
NC = 128
x = torch.randn(64, 2 * NC, device=dev, dtype=torch.bfloat16)
g, u = x[:, :NC], x[:, NC:]
ref = (F.silu(g.float()) * u.float()).to(torch.bfloat16)

import triton
import triton.language as tl


@triton.jit
def _silu_mul_kernel(o_ptr, x_ptr, total, NC, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < total
    row = off // NC
    col = off % NC
    gp = tl.load(x_ptr + row * 2 * NC + col, mask=m, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + row * 2 * NC + NC + col, mask=m, other=0.0).to(tl.float32)
    r = gp * tl.sigmoid(gp) * up
    tl.store(o_ptr + off, r.to(o_ptr.dtype.element_ty), mask=m)


def triton_silu_and_mul(out, inp):
    xc = inp.reshape(-1, inp.shape[-1])
    oc = out.reshape(-1, out.shape[-1])
    total = oc.numel()
    ncol = oc.shape[-1]
    BLOCK = 256
    _silu_mul_kernel[(triton.cdiv(total, BLOCK),)](
        oc, xc, total, ncol, BLOCK=BLOCK
    )
    return out


print("=== A. 自写 Triton 版数值校验 ===", flush=True)
ot = torch.zeros(64, NC, device=dev, dtype=torch.bfloat16)
triton_silu_and_mul(ot, x)
torch.cuda.synchronize()
print(f"triton absmax={ot.float().abs().max().item():.4f} nz={int((ot != 0).sum())}/{ot.numel()}"
      f" maxdiff={(ot.float() - ref.float()).abs().max().item():.6f}", flush=True)

print("=== B. 尝试 torch.library 覆盖 _C.silu_and_mul(CUDA) ===", flush=True)
try:
    lib = torch.library.Library("_C", "IMPL", "CUDA")
    lib.impl("silu_and_mul", triton_silu_and_mul)
    print("   override registered OK", flush=True)
    o2 = torch.zeros(64, NC, device=dev, dtype=torch.bfloat16)
    torch.ops._C.silu_and_mul(o2, x)
    torch.cuda.synchronize()
    print(f"   after override: absmax={o2.float().abs().max().item():.4f} "
          f"nz={int((o2 != 0).sum())}/{o2.numel()} "
          f"maxdiff={(o2.float() - ref.float()).abs().max().item():.6f}", flush=True)
except Exception as e:
    print("   override EXC:", type(e).__name__, str(e)[:400], flush=True)

print("=== C. SilentAndMul 端到端 (用 default_vllm_config) ===", flush=True)
try:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.activation import SiluAndMul

    with set_current_vllm_config(VllmConfig()):
        m = SiluAndMul().to(dev)
        y = m(x)
    torch.cuda.synchronize()
    print(f"   SiluAndMul absmax={y.float().abs().max().item():.4f} "
          f"nz={int((y != 0).sum())}/{y.numel()} "
          f"maxdiff={(y.float() - ref.float()).abs().max().item():.6f}", flush=True)
except Exception as e:
    print("   EXC:", type(e).__name__, str(e)[:300], flush=True)

print("PROBE4_DONE", flush=True)
