#!/usr/bin/env python3
# cg_capture_test.py — 直接检验 _torch_moe_align_block_size 能否被 CUDA graph 捕获
import torch

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    _torch_moe_align_block_size,
)

dev = "cuda"
E, B = 512, 16


def check(M, topk, pad=False):
    ids = torch.randint(0, E, (M, topk), device=dev, dtype=torch.int32)
    if M >= 2:
        ids[-1, :] = -1                      # 尾部填充行

    # 参考 (eager)
    ref = _torch_moe_align_block_size(ids.clone(), B, E, pad)

    # warmup on side stream
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            _torch_moe_align_block_size(ids, B, E, pad)
    torch.cuda.current_stream().wait_stream(s)

    static = ids.clone()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            got = _torch_moe_align_block_size(static, B, E, pad)
        g.replay()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[M={M} topk={topk} pad={pad}] CAPTURE FAILED: "
              f"{type(e).__name__}: {str(e)[:220]}", flush=True)
        return
    npp = int(got[2])
    ok = (int(ref[2]) == npp
          and torch.equal(ref[0][:npp].cpu(), got[0][:npp].cpu())
          and torch.equal(ref[1].cpu(), got[1].cpu()))
    print(f"[M={M} topk={topk} pad={pad}] CAPTURE OK  npp={npp} "
          f"len={got[0].numel()} 与 eager 一致={ok}", flush=True)


for M, topk, pad in [(1, 10, False), (4, 10, False), (4, 10, True),
                     (64, 10, False), (16, 1, False), (300, 10, False)]:
    check(M, topk, pad)
print("CG_TEST_DONE", flush=True)
