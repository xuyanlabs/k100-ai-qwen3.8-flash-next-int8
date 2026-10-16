#!/usr/bin/env python3
# validate_router.py — 验证接入后的融合路由: 数值对拍 + padding mask + 图捕获
import torch

from vllm.model_executor.layers.fused_moe.router import fused_topk_router as R

dev = "cuda"
E, K = 512, 10


def run(scoring, M, renormalize, use_triton, pad=None):
    torch.manual_seed(M * 13 + (0 if scoring == "softmax" else 1))
    g = torch.randn(M, E, device=dev, dtype=torch.bfloat16)
    tw = torch.empty(M, K, device=dev, dtype=torch.float32)
    ti = torch.empty(M, K, device=dev, dtype=torch.int32)
    te = torch.empty(M, K, device=dev, dtype=torch.int32)
    old = R._ROUTER_USE_TRITON
    R._ROUTER_USE_TRITON = use_triton
    try:
        if use_triton:
            ok = R._k100_router_triton(scoring, tw, ti, te, g, renormalize, pad)
            if not ok:
                raise RuntimeError("fast path declined")
        else:
            s = (torch.sigmoid(g.float()) if scoring == "sigmoid"
                 else torch.softmax(g.float(), dim=-1))
            R._torch_topk_fill(s, tw, ti, te, renormalize, pad)
        torch.cuda.synchronize()
    finally:
        R._ROUTER_USE_TRITON = old
    return g, tw, ti, te


print("=== 数值对拍 (经公开入口 vllm_topk_softmax / sigmoid) ===")
worst = 0.0
for scoring, fn in (("softmax", R.vllm_topk_softmax), ("sigmoid", R.vllm_topk_sigmoid)):
    for renorm in (True, False):
        for M in (1, 2, 4, 8, 16, 64, 128):
            torch.manual_seed(M * 13 + 7)
            g = torch.randn(M, E, device=dev, dtype=torch.bfloat16)
            res = {}
            for use in (True, False):
                tw = torch.empty(M, K, device=dev, dtype=torch.float32)
                ti = torch.empty(M, K, device=dev, dtype=torch.int32)
                te = torch.empty(M, K, device=dev, dtype=torch.int32)
                old = R._ROUTER_USE_TRITON
                R._ROUTER_USE_TRITON = use
                try:
                    fn(tw, ti, te, g, renorm)
                    torch.cuda.synchronize()
                finally:
                    R._ROUTER_USE_TRITON = old
                res[use] = (tw.clone(), ti.clone())
            d = (res[True][0].double() - res[False][0].double()).abs().max().item()
            rel = d / max(res[False][0].double().abs().max().item(), 1e-9)
            worst = max(worst, rel)
            if rel > 1e-6:
                print(f"  !! {scoring} renorm={renorm} M={M} rel={rel:.2e}")
print(f"  最大相对误差 = {worst:.3e}  (阈值 1e-6)")

print()
print("=== padding mask 语义 ===")
for M in (4, 16):
    pad = torch.zeros(M, device=dev, dtype=torch.bool)
    pad[0] = True
    pad[M - 1] = True
    _, wa, _, _ = run("softmax", M, True, True, pad)
    _, wb, _, _ = run("softmax", M, True, False, pad)
    d = (wa.double() - wb.double()).abs().max().item()
    zero_ok = bool((wa[0].abs().max().item() == 0) and (wa[M - 1].abs().max().item() == 0))
    print(f"  M={M}: 与 torch 差={d:.3e}  被标记行的权重已清零={zero_ok}")

print()
print("=== CUDA graph 捕获 ===")
for M in (1, 4, 16, 64):
    g = torch.randn(M, E, device=dev, dtype=torch.bfloat16)
    tw = torch.empty(M, K, device=dev, dtype=torch.float32)
    ti = torch.empty(M, K, device=dev, dtype=torch.int32)
    te = torch.empty(M, K, device=dev, dtype=torch.int32)

    def fn():
        R._k100_router_triton("softmax", tw, ti, te, g, True, None)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    try:
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            fn()
        gr.replay()
        torch.cuda.synchronize()
        # 用同一份 g 算参考 (不要重新 seed)
        s_ref = torch.softmax(g.float(), dim=-1)
        wr, _ir = torch.topk(s_ref, K, dim=-1)
        wr = wr / (wr.sum(dim=-1, keepdim=True) + 1e-20)
        d = (tw.double() - wr.double()).abs().max().item()
        rel = d / max(wr.double().abs().max().item(), 1e-9)
        print(f"  M={M}: 捕获 OK  replay 相对误差={rel:.3e}")
    except Exception as e:
        print(f"  M={M}: 捕获失败 {type(e).__name__}: {str(e)[:150]}")
print("VALIDATE_ROUTER_DONE", flush=True)
