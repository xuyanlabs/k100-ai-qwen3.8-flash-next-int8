#!/usr/bin/env python3
# repro_moe_stages.py — fused_moe int8 分级数值解剖(合成权重, 真实形状)
# M=5 -> naive 路径; M=64 -> aligned 路径
import sys
import torch

dev = "cuda"
torch.manual_seed(0)
H, NI, E, TOPK = 2560, 640, 512, 10

from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_triton_kernel,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_quant_int8,
)
from vllm.triton_utils import tl


def run(M):
    print(f"\n===== M={M} =====", flush=True)
    hidden = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    w1q = torch.randint(-127, 128, (E, 2 * NI, H), device=dev, dtype=torch.int8)
    w2q = torch.randint(-127, 128, (E, H, NI), device=dev, dtype=torch.int8)
    w1s = torch.rand(E, 2 * NI, 1, device=dev, dtype=torch.float32) * 0.004 + 1e-4
    w2s = torch.rand(E, H, 1, device=dev, dtype=torch.float32) * 0.004 + 1e-4
    logits = torch.randn(M, E, device=dev, dtype=torch.float32)
    tw, ti = torch.topk(logits, TOPK, dim=-1)
    tw = torch.softmax(tw, dim=-1)

    A_q, A_s = per_token_quant_int8(hidden)
    torch.cuda.synchronize()
    print(f"A_q nonzero={int((A_q != 0).sum())} A_s[0,0]={float(A_s[0,0]):.5f}", flush=True)

    # 与引擎一致的 align/naive 选择
    naive = M * TOPK * 4 <= E
    if naive:
        sorted_ids, expert_ids, npp = None, ti.view(-1), torch.full(
            (1,), ti.numel() * 16, dtype=torch.int32, device=dev)
        print("路径=naive", flush=True)
    else:
        sorted_ids, expert_ids, npp = moe_align_block_size(ti, 16, E, None)
        print(f"路径=aligned npp={int(npp[0])}", flush=True)

    C1 = torch.zeros(M, TOPK, 2 * NI, device=dev, dtype=torch.bfloat16)
    config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
              "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2}
    invoke_fused_moe_triton_kernel(
        A_q, w1q, C1, A_s, w1s,
        None, sorted_ids, expert_ids, npp,
        False, TOPK, config, tl.bfloat16,
        False, True, False, False, True, None,
    )
    torch.cuda.synchronize()
    c1f = C1.float()
    print(f"C1(w1 out): nonzero={int((C1 != 0).sum())}/{C1.numel()} "
          f"absmax={c1f.abs().max().item():.4f} nan={bool(c1f.isnan().any())}", flush=True)

    # 参考: expert 238... 用 ti[0,0] 的专家, token0, 前 4 个输出
    e0 = int(ti[0, 0])
    ref = (hidden[0].float() @ (w1q[e0].float() * w1s[e0]).t())[:4]
    got = C1[0, 0, :4].float()
    print(f"ref tok0/exp{e0}[:4]={ref.tolist()}", flush=True)
    print(f"got tok0/exp{e0}[:4]={got.tolist()}", flush=True)

    if bool((C1 != 0).any()):
        # silu + 第二段(走 fork 自带的 apply_moe_activation, 与引擎完全一致)
        from vllm.model_executor.layers.fused_moe.activation import (
            MoEActivation,
            apply_moe_activation,
        )
        inter = torch.empty(M * TOPK, NI, device=dev, dtype=torch.bfloat16)
        apply_moe_activation(
            MoEActivation.SILU, inter, C1.view(M * TOPK, 2 * NI)
        )
        torch.cuda.synchronize()
        print(f"inter(silu): nonzero={int((inter != 0).sum())} "
              f"absmax={inter.float().abs().max().item():.4f} "
              f"nan={bool(inter.float().isnan().any())}", flush=True)
        A2_q, A2_s = per_token_quant_int8(inter)
        C2 = torch.zeros(M, 1, H, device=dev, dtype=torch.bfloat16)
        if naive:
            s2, e2 = None, ti.view(-1)
            npp2 = torch.full((1,), M * TOPK * 16, dtype=torch.int32, device=dev)
        else:
            s2, e2, npp2 = moe_align_block_size(ti, 16, E, None)
        invoke_fused_moe_triton_kernel(
            A2_q, w2q, C2, A2_s, w2s,
            tw.view(-1, 1), s2, e2, npp2,
            True, 1, config, tl.bfloat16,
            False, True, False, False, True, None,
        )
        torch.cuda.synchronize()
        c2f = C2.float()
        print(f"C2(w2 out): nonzero={int((C2 != 0).sum())} "
              f"absmax={c2f.abs().max().item():.4f} nan={bool(c2f.isnan().any())}", flush=True)

        # 末段: moe_sum 加权归并(预编译核嫌疑)
        out = torch.zeros(M, H, device=dev, dtype=torch.bfloat16)
        from vllm import _custom_ops as _ops
        _ops.moe_sum(C2, out)
        torch.cuda.synchronize()
        print(f"moe_sum out: nonzero={int((out != 0).sum())} "
              f"absmax={out.float().abs().max().item():.4f} "
              f"nan={bool(out.float().isnan().any())}", flush=True)
        # 参考归并: 手工 sum
        ref_sum = (C2.view(M, 1, H).float()).sum(dim=1)
        print(f"手工sum absmax={ref_sum.abs().max().item():.4f} "
              f"vs moe_sum 一致={torch.allclose(out.float(), ref_sum, atol=1e-2)}", flush=True)


run(5)
run(64)
print("DONE", flush=True)
