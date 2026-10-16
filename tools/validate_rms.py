#!/usr/bin/env python3
# validate_rms.py — 融合 RMSNorm 对拍 + 图捕获
import torch

from vllm.ir.ops import layernorm as L

dev = "cuda"


def cmp(name, got, ref):
    g = got.double()
    r = ref.double()
    d = (g - r).abs().max().item()
    scale = max(r.abs().max().item(), 1e-9)
    print(f"  {name:34s} maxdiff={d:.3e}  rel={d / scale:.3e}")
    return d / scale


print("=== rms_norm 对拍 (vs 原 native 实现) ===")
worst = 0.0
for N in (128, 256, 2560):
    for M in (1, 2, 4, 16, 64, 256):
        torch.manual_seed(N + M)
        x = (torch.randn(M, N, device=dev, dtype=torch.bfloat16) * 2)
        w = torch.randn(N, device=dev, dtype=torch.bfloat16)
        eps = 1e-6
        fast = L.rms_norm(x, w, eps)
        ref = L._k100_native_rms(x, w, eps, None)
        torch.cuda.synchronize()
        rel = cmp(f"N={N} M={M}", fast, ref)
        worst = max(worst, rel)
print(f"  => 最大相对误差 {worst:.3e}")

print()
print("=== fused_add_rms_norm 对拍 ===")
worst2 = 0.0
for N in (256, 2560):
    for M in (1, 4, 16, 64):
        torch.manual_seed(N * 3 + M)
        x = (torch.randn(M, N, device=dev, dtype=torch.bfloat16) * 2)
        r = (torch.randn(M, N, device=dev, dtype=torch.bfloat16) * 2)
        w = torch.randn(N, device=dev, dtype=torch.bfloat16)
        eps = 1e-6
        fo, fr = L.fused_add_rms_norm(x, r, w, eps)
        no, nr = L._k100_native_fused(x, r, w, eps, None)
        torch.cuda.synchronize()
        a = cmp(f"N={N} M={M} out", fo, no)
        b = cmp(f"N={N} M={M} residual", fr, nr)
        worst2 = max(worst2, a, b)
print(f"  => 最大相对误差 {worst2:.3e}")

print()
print("=== CUDA graph 捕获 ===")
for N, M in ((2560, 1), (2560, 8), (2560, 64), (256, 64)):
    x = (torch.randn(M, N, device=dev, dtype=torch.bfloat16) * 2)
    r = (torch.randn(M, N, device=dev, dtype=torch.bfloat16) * 2)
    w = torch.randn(N, device=dev, dtype=torch.bfloat16)
    out = torch.empty_like(x)
    res_out = torch.empty_like(r)

    def fn():
        L._k100_launch(x, w, out, r, res_out, 1e-6)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        g.replay()
        torch.cuda.synchronize()
        no, nr = L._k100_native_fused(x, r, w, 1e-6, None)
        d = (out.double() - no.double()).abs().max().item()
        rel = d / max(no.double().abs().max().item(), 1e-9)
        print(f"  N={N} M={M}: 捕获 OK  replay 相对误差={rel:.3e}")
    except Exception as e:
        print(f"  N={N} M={M}: 捕获失败 {type(e).__name__}: {str(e)[:140]}")

print()
print("=== 单核耗时 (图内) ===")


def time_graph(fn, iters=300):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1000.0


for N, M in ((2560, 1), (2560, 8), (2560, 64), (2560, 256)):
    x = (torch.randn(M, N, device=dev, dtype=torch.bfloat16) * 2)
    w = torch.randn(N, device=dev, dtype=torch.bfloat16)
    out = torch.empty_like(x)
    a = time_graph(lambda: L._k100_native_rms(x, w, 1e-6, None))
    b = time_graph(lambda: L._k100_launch(x, w, out, None, None, 1e-6))
    print(f"  N={N} M={M:4d}: native={a:7.2f}us  triton={b:7.2f}us  {a / b:.2f}x")
print("VALIDATE_RMS_DONE", flush=True)
