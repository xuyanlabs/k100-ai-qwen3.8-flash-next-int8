#!/usr/bin/env python3
# repro_moe.py — 单卡复现 fused_moe_kernel VMFault，按预设配置逐个试
# 用法: python3 repro_moe.py <case>
#   case: stage0 | bf16 | conservative | nomfma | nopipe | tiny
import sys

import torch
import triton
import triton.language as tl

PRESETS = {
    # 与服务器 /moe_configs 里相同的保守档
    "conservative": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64,
                     "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
                     "waves_per_eu": 0, "matrix_instr_nonkdim": 16, "kpack": 1},
    # 关掉 MFMA 矩阵指令（纯 FMA 路径）
    "nomfma": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64,
               "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
               "waves_per_eu": 0, "matrix_instr_nonkdim": 0, "kpack": 1},
    # 关软件流水
    "nopipe": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64,
               "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 1,
               "waves_per_eu": 0, "matrix_instr_nonkdim": 16, "kpack": 1},
    # 最小最笨
    "tiny": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 32,
             "GROUP_SIZE_M": 1, "num_warps": 2, "num_stages": 1,
             "waves_per_eu": 0, "matrix_instr_nonkdim": 0, "kpack": 1},
}


def stage0():
    """baseline: 裸 Triton int8 tl.dot，不碰 vLLM"""

    @triton.jit
    def dot_kernel(a_ptr, b_ptr, c_ptr,
                   M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        offs_m = tl.arange(0, BM)
        offs_n = tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        acc = tl.dot(a, b)
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

    M = N = K = 32
    a = torch.randint(-8, 8, (M, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-8, 8, (K, N), dtype=torch.int8, device="cuda")
    c = torch.empty((M, N), dtype=torch.int32, device="cuda")
    dot_kernel[(1,)](a, b, c, M, N, K, 32, 32, 32)
    torch.cuda.synchronize()
    ref = a.cpu().to(torch.int32) @ b.cpu().to(torch.int32)
    ok = torch.equal(c.cpu(), ref)
    print("stage0 裸 int8 tl.dot:", "OK" if ok else "结果错误(编译过了但算错)")


def stage1(cfg, bf16, slack=0, clamp=False):
    """直接调 vllm 的 invoke_fused_moe_triton_kernel，形状=线上每 rank:
    E_local=128, N=160, K=2560, top_k=10, M=4096
    slack: B/B_scale 多分配几个 expert 行（验证越界读假设）
    clamp: 发射前把 expert_ids 钳到 [0,E-1]（验证修复方案）"""
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        invoke_fused_moe_triton_kernel,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    M, E, N, K, TOPK = 4096, 128, 160, 2560, 10
    dev = "cuda"
    torch.manual_seed(0)
    if bf16:
        A = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        B = torch.randn(E + slack, 2 * N, K, dtype=torch.bfloat16, device=dev) * 0.02
        A_scale = None
        B_scale = None
        use_int8 = False
    else:
        A = torch.randint(-8, 8, (M, K), dtype=torch.int8, device=dev)
        A_scale = torch.rand(M, 1, dtype=torch.float32, device=dev) * 0.02
        B = torch.randint(-8, 8, (E + slack, 2 * N, K), dtype=torch.int8, device=dev)
        B_scale = torch.rand(E + slack, 2 * N, dtype=torch.float32, device=dev) * 0.02
        use_int8 = True
    C = torch.empty(M, TOPK, 2 * N, dtype=torch.bfloat16, device=dev)
    topk_ids = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=dev)
    topk_weights = torch.rand(M, TOPK, dtype=torch.float32, device=dev)
    sorted_ids, expert_ids, npp = moe_align_block_size(
        topk_ids, cfg["BLOCK_SIZE_M"], E, None
    )
    if clamp:
        expert_ids = expert_ids.clamp(max=E - 1)
    invoke_fused_moe_triton_kernel(
        A, B, C, A_scale, B_scale, topk_weights,
        sorted_ids, expert_ids, npp,
        False, TOPK, cfg, tl.bfloat16,
        False, use_int8, False, False, True, None, None,
    )
    torch.cuda.synchronize()
    print("stage1 bf16=%s slack=%d clamp=%s 通过: %s" % (bf16, slack, clamp, cfg))


def stage2(cfg):
    """下行投影（第二 GEMM）忠实形状：A int8 [M*TOPK, 160]，B=w2 int8 [128, 2560, 160]，
    B_scale [128, 2560]，C [M, TOPK, 2560]，K=160 与 BK 的整除关系是变量。"""
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        invoke_fused_moe_triton_kernel,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    M, E, N_OUT, K_IN, TOPK = 4096, 128, 2560, 160, 10
    dev = "cuda"
    torch.manual_seed(0)
    A = torch.randint(-8, 8, (M * TOPK, K_IN), dtype=torch.int8, device=dev)
    A_scale = torch.rand(M * TOPK, 1, dtype=torch.float32, device=dev) * 0.02
    B = torch.randint(-8, 8, (E, N_OUT, K_IN), dtype=torch.int8, device=dev)
    B_scale = torch.rand(E, N_OUT, dtype=torch.float32, device=dev) * 0.02
    C = torch.empty(M, TOPK, N_OUT, dtype=torch.bfloat16, device=dev)
    topk_ids = torch.randint(0, E, (M, TOPK), dtype=torch.int32, device=dev)
    topk_weights = torch.rand(M, TOPK, dtype=torch.float32, device=dev)
    sorted_ids, expert_ids, npp = moe_align_block_size(
        topk_ids, cfg["BLOCK_SIZE_M"], E, None
    )
    expert_ids = expert_ids.clamp(max=E - 1)
    invoke_fused_moe_triton_kernel(
        A, B, C, A_scale, B_scale, topk_weights,
        sorted_ids, expert_ids, npp,
        True, TOPK, cfg, tl.bfloat16,
        False, True, False, False, True, None, None,
    )
    torch.cuda.synchronize()
    print("stage2 down-proj 通过: %s" % cfg)


def stage3(clamp):
    """引擎 TP4 的忠实复刻:E_global=512, E_local=128, expert_map 四分之三 -1。
    手动展开 moe_align_block_size 内部两步（op + gather），逐步定位炸点。"""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        invoke_fused_moe_triton_kernel,
    )
    import triton as _tr

    M, E_G, E_L, N, K, TOPK = 8192, 512, 128, 160, 2560, 10
    cfg = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64,
           "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
           "waves_per_eu": 0, "matrix_instr_nonkdim": 16, "kpack": 1}
    dev = "cuda"
    torch.manual_seed(0)
    A = torch.randint(-8, 8, (M, K), dtype=torch.int8, device=dev)
    A_scale = torch.rand(M, 1, dtype=torch.float32, device=dev) * 0.02
    B = torch.randint(-8, 8, (E_L, 2 * N, K), dtype=torch.int8, device=dev)
    B_scale = torch.rand(E_L, 2 * N, dtype=torch.float32, device=dev) * 0.02
    C = torch.empty(M, TOPK, 2 * N, dtype=torch.bfloat16, device=dev)
    topk_ids = torch.randint(0, E_G, (M, TOPK), dtype=torch.int32, device=dev)
    topk_weights = torch.rand(M, TOPK, dtype=torch.float32, device=dev)
    expert_map = torch.full((E_G,), -1, dtype=torch.int32, device=dev)
    expert_map[:E_L] = torch.arange(E_L, dtype=torch.int32, device=dev)

    # —— 手动展开 moe_align_block_size(wrapper 不传 map 给 op) ——
    BM = cfg["BLOCK_SIZE_M"]
    max_num_tokens_padded = topk_ids.numel() + E_G * (BM - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=dev)
    max_num_m_blocks = _tr.cdiv(max_num_tokens_padded, BM)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=dev)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=dev)
    print("step1: align op 发射, expert_ids len=%d" % max_num_m_blocks, flush=True)
    ops.moe_align_block_size(topk_ids, E_G, BM, sorted_ids, expert_ids,
                             num_tokens_post_pad, None)
    torch.cuda.synchronize()
    npp = num_tokens_post_pad.item()
    blocks_used = (npp + BM - 1) // BM
    used = expert_ids[:blocks_used]
    tail = expert_ids[blocks_used:]
    bad_used = int((used >= E_G).sum().item()) + int((used < 0).sum().item())
    bad_tail = int((tail >= E_G).sum().item()) + int((tail < 0).sum().item())
    print("step1 OK: post_padded=%d blocks_used=%d/%d" % (npp, blocks_used, max_num_m_blocks), flush=True)
    print("step1 值域: 有效区野值=%d 个, 尾部区野值=%d 个, min=%d max=%d" % (
        bad_used, bad_tail, expert_ids.min().item(), expert_ids.max().item()), flush=True)

    print("step2: gather expert_map[expert_ids]", flush=True)
    expert_ids = expert_map[expert_ids]
    torch.cuda.synchronize()
    print("step2 OK: gather 完成; min=%d max=%d" % (
        expert_ids.min().item(), expert_ids.max().item()), flush=True)

    if clamp:
        expert_ids = expert_ids.clamp(min=-1, max=E_L - 1)

    print("step3: fused_moe_kernel 发射", flush=True)
    invoke_fused_moe_triton_kernel(
        A, B, C, A_scale, B_scale, topk_weights,
        sorted_ids, expert_ids, num_tokens_post_pad,
        False, TOPK, cfg, tl.bfloat16,
        False, True, False, False, True, None, None,
    )
    torch.cuda.synchronize()
    print("stage3 emap clamp=%s 通过" % clamp)


def stage3(clamp, M=8192):
    """引擎 TP4 忠实路径：走（已打补丁的）moe_align_block_size wrapper，
    E_global=512, E_local=128, expert_map 四分之三 -1。"""
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        invoke_fused_moe_triton_kernel,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    E_G, E_L, N, K, TOPK = 512, 128, 160, 2560, 10
    cfg = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64,
           "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
           "waves_per_eu": 0, "matrix_instr_nonkdim": 16, "kpack": 1}
    dev = "cuda"
    torch.manual_seed(0)
    A = torch.randint(-8, 8, (M, K), dtype=torch.int8, device=dev)
    A_scale = torch.rand(M, 1, dtype=torch.float32, device=dev) * 0.02
    B = torch.randint(-8, 8, (E_L, 2 * N, K), dtype=torch.int8, device=dev)
    B_scale = torch.rand(E_L, 2 * N, dtype=torch.float32, device=dev) * 0.02
    C = torch.empty(M, TOPK, 2 * N, dtype=torch.bfloat16, device=dev)
    topk_ids = torch.randint(0, E_G, (M, TOPK), dtype=torch.int32, device=dev)
    topk_weights = torch.rand(M, TOPK, dtype=torch.float32, device=dev)
    expert_map = torch.full((E_G,), -1, dtype=torch.int32, device=dev)
    expert_map[:E_L] = torch.arange(E_L, dtype=torch.int32, device=dev)

    sorted_ids, expert_ids, npp = moe_align_block_size(
        topk_ids, cfg["BLOCK_SIZE_M"], E_G, expert_map
    )
    torch.cuda.synchronize()
    print("align(wrapper): npp=%d expert_ids min=%d max=%d" % (
        npp.item(), expert_ids.min().item(), expert_ids.max().item()), flush=True)
    assert expert_ids.min().item() >= -1 and expert_ids.max().item() < E_L, \
        "expert_ids 越界!"

    if clamp:
        expert_ids = expert_ids.clamp(min=-1, max=E_L - 1)

    print("step3: fused_moe_kernel 发射", flush=True)
    invoke_fused_moe_triton_kernel(
        A, B, C, A_scale, B_scale, topk_weights,
        sorted_ids, expert_ids, npp,
        False, TOPK, cfg, tl.bfloat16,
        False, True, False, False, True, None, None,
    )
    torch.cuda.synchronize()
    print("stage3 emap clamp=%s 通过" % clamp)


def stage3_negpad():
    """topk_ids 含 -1 填充行（引擎 cudagraph pad 场景），走补丁 wrapper + kernel。"""
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        invoke_fused_moe_triton_kernel,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    M, E_G, E_L, N, K, TOPK = 1024, 512, 128, 160, 2560, 10
    cfg = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64,
           "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
           "waves_per_eu": 0, "matrix_instr_nonkdim": 16, "kpack": 1}
    dev = "cuda"
    torch.manual_seed(0)
    A = torch.randint(-8, 8, (M, K), dtype=torch.int8, device=dev)
    A_scale = torch.rand(M, 1, dtype=torch.float32, device=dev) * 0.02
    B = torch.randint(-8, 8, (E_L, 2 * N, K), dtype=torch.int8, device=dev)
    B_scale = torch.rand(E_L, 2 * N, dtype=torch.float32, device=dev) * 0.02
    C = torch.empty(M, TOPK, 2 * N, dtype=torch.bfloat16, device=dev)
    topk_ids = torch.randint(0, E_G, (M, TOPK), dtype=torch.int32, device=dev)
    topk_ids[M // 2:, :] = -1  # 后半=填充行
    topk_weights = torch.rand(M, TOPK, dtype=torch.float32, device=dev)
    expert_map = torch.full((E_G,), -1, dtype=torch.int32, device=dev)
    expert_map[:E_L] = torch.arange(E_L, dtype=torch.int32, device=dev)

    sorted_ids, expert_ids, npp = moe_align_block_size(
        topk_ids, cfg["BLOCK_SIZE_M"], E_G, expert_map
    )
    torch.cuda.synchronize()
    print("negpad align: npp=%d expert_ids min=%d max=%d" % (
        npp.item(), expert_ids.min().item(), expert_ids.max().item()), flush=True)
    assert -1 <= expert_ids.min().item() and expert_ids.max().item() < E_L
    invoke_fused_moe_triton_kernel(
        A, B, C, A_scale, B_scale, topk_weights,
        sorted_ids, expert_ids, npp,
        False, TOPK, cfg, tl.bfloat16,
        False, True, False, False, True, None, None,
    )
    torch.cuda.synchronize()
    print("stage3_negpad 通过")


def sampler_case(do_scatter, dtype=torch.bfloat16):
    """采样器崩点复刻:[64, 248320] 上 sort + scatter_/gather 回填。"""
    M, V = 64, 248320
    dev = "cuda"
    torch.manual_seed(0)
    x = torch.randn(M, V, dtype=dtype, device=dev)
    print("dtype=%s sort 中..." % dtype, flush=True)
    srt, idx = x.sort(dim=-1)
    torch.cuda.synchronize()
    print("sort OK", flush=True)
    if do_scatter:
        out = torch.empty_like(x)
        out.scatter_(dim=-1, index=idx, src=srt)
        torch.cuda.synchronize()
        print("scatter_ OK", flush=True)
    else:
        inv = idx.argsort(dim=-1)
        out = srt.gather(-1, inv)
        torch.cuda.synchronize()
        print("gather 回填 OK", flush=True)
    # 数值抽查
    ok = torch.allclose(out.float(), x.float(), rtol=1e-2, atol=1e-2)
    print("数值一致:" , ok)


def sampler_vllm_case(strided):
    """直接调 vllm 采样器的 apply_top_k_top_p(引擎同款函数),
    strided=True 时 logits 用非连续视图(模拟引擎切片 buffer)。"""
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

    M, V = 64, 248320
    dev = "cuda"
    torch.manual_seed(0)
    if strided:
        base = torch.randn(M, V + 1024, dtype=torch.float32, device=dev)
        logits = base[:, :V]
        print("logits 非连续:", not logits.is_contiguous(), flush=True)
    else:
        logits = torch.randn(M, V, dtype=torch.float32, device=dev)
    # 引擎语义:temperature=0.3 已提前除过;top_k=None, top_p=1.0
    p = torch.ones(M, dtype=torch.float32, device=dev)
    print("调 apply_top_k_top_p(None, 1.0)...", flush=True)
    out = apply_top_k_top_p(logits, None, p)
    torch.cuda.synchronize()
    print("apply_top_k_top_p OK", flush=True)
    # 再试 top_k=50 的分支(走 gather 路径)
    k = torch.full((M,), 50, dtype=torch.int32, device=dev)
    print("调 apply_top_k_top_p(k=50, 1.0)...", flush=True)
    out = apply_top_k_top_p(logits, k, p)
    torch.cuda.synchronize()
    print("top_k 分支 OK", flush=True)


def chain_case():
    """逐步复刻 apply_top_k_top_p_pytorch 的算子链,每步 sync+打印,找挂点。"""
    M, V = 64, 248320
    dev = "cuda"
    torch.manual_seed(0)
    logits = torch.randn(M, V, dtype=torch.float32, device=dev)

    def step(name, fn):
        print("step:", name, flush=True)
        r = fn()
        torch.cuda.synchronize()
        print("step OK:", name, flush=True)
        return r

    logits_sort, logits_idx = step("sort", lambda: logits.sort(dim=-1, descending=False))
    step("gather", lambda: logits_sort.gather(1, (V - 50) * torch.ones(M, 1, dtype=torch.long, device=dev)))
    top_k_mask = step("mask_cmp", lambda: logits_sort < logits_sort[:, -50:].clamp_min(0).max(dim=1, keepdim=True).values)
    step("masked_fill_", lambda: logits_sort.masked_fill_(top_k_mask, -float("inf")))
    probs_sort = step("softmax", lambda: logits_sort.softmax(dim=-1))
    probs_sum = step("cumsum", lambda: torch.cumsum(probs_sort, dim=-1))
    top_p_mask = step("cumsum_cmp", lambda: probs_sum <= 0.0)
    top_p_mask[:, -1] = False
    step("masked_fill_2", lambda: logits_sort.masked_fill_(top_p_mask, -float("inf")))
    step("scatter_", lambda: logits.scatter_(dim=-1, index=logits_idx, src=logits_sort))
    print("chain 全过")


def mm_case():
    """int8 scaled_mm(稠密 GEMM)在 decode 形状下的复现:
    引擎路径 = ops.scaled_int8_quant(x) -> triton_scaled_mm。
    扫 M∈{1,4,4096} × N∈{2560,7680,512}, K=2560。"""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
        triton_scaled_mm,
    )

    dev = "cuda"
    K = 2560
    torch.manual_seed(0)
    for M in (1, 4, 4096):
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        x_q, x_s, _ = ops.scaled_int8_quant(x.contiguous(), None, None, symmetric=True)
        for N in (2560, 7680, 512):
            w_q = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=dev)
            w_s = torch.rand(N, 1, dtype=torch.float32, device=dev) * 0.02
            print(f"MM M={M} N={N} K={K} 发射...", flush=True)
            out = triton_scaled_mm(x_q, w_q, scale_a=x_s, scale_b=w_s,
                                   out_dtype=torch.bfloat16)
            torch.cuda.synchronize()
            print(f"MM M={M} N={N} OK out={tuple(out.shape)}", flush=True)
    print("mm 全过")


def verify():
    """数值正确性端到端核验：小尺寸 + expert_map,核对 kernel 输出 vs CPU 参考。
    -1 专家块必须写零；本地专家必须数值吻合（bf16 舍入容差内）。"""
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        invoke_fused_moe_triton_kernel,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    M, E_G, E_L, N2, K, TOPK = 64, 8, 2, 64, 32, 2
    cfg = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32,
           "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
           "waves_per_eu": 0, "matrix_instr_nonkdim": 16, "kpack": 1}
    dev = "cuda"
    torch.manual_seed(7)
    A = torch.randint(-8, 8, (M, K), dtype=torch.int8, device=dev)
    A_scale = torch.rand(M, 1, dtype=torch.float32, device=dev) * 0.05 + 0.001
    B = torch.randint(-8, 8, (E_L, N2, K), dtype=torch.int8, device=dev)
    B_scale = torch.rand(E_L, N2, dtype=torch.float32, device=dev) * 0.05 + 0.001
    C = torch.full((M, TOPK, N2), float("nan"), dtype=torch.bfloat16, device=dev)
    topk_ids = torch.randint(0, E_G, (M, TOPK), dtype=torch.int32, device=dev)
    topk_weights = torch.rand(M, TOPK, dtype=torch.float32, device=dev)
    expert_map = torch.tensor([0, 1, -1, -1, -1, -1, -1, -1],
                              dtype=torch.int32, device=dev)

    sorted_ids, expert_ids, npp = moe_align_block_size(
        topk_ids, cfg["BLOCK_SIZE_M"], E_G, expert_map
    )
    invoke_fused_moe_triton_kernel(
        A, B, C, A_scale, B_scale, topk_weights,
        sorted_ids, expert_ids, npp,
        False, TOPK, cfg, tl.bfloat16,
        False, True, False, False, True, None, None,
    )
    torch.cuda.synchronize()

    # CPU 参考
    Ac = A.cpu().to(torch.float64)
    Bc = B.cpu().to(torch.float64)
    As = A_scale.cpu().to(torch.float64)
    Bs = B_scale.cpu().to(torch.float64)
    ids = topk_ids.cpu()
    Cp = C.cpu().to(torch.float64)
    bad = 0
    for t in range(M):
        for k in range(TOPK):
            g = int(ids[t, k])
            l = int(expert_map.cpu()[g])
            got = Cp[t, k]
            if l < 0:
                ok = bool((got == 0).all())
            else:
                ref = (Ac[t] * As[t, 0]) @ (Bc[l] * Bs[l, :, None]).T
                ok = bool(torch.allclose(got, ref, rtol=2e-2, atol=2e-2))
            if not ok:
                bad += 1
                if bad <= 3:
                    print("  MISMATCH t=%d k=%d g=%d l=%d" % (t, k, g, l))
    if bad == 0:
        print("verify: 全部 %d 个 (token,topk) 数值吻合" % (M * TOPK))
    else:
        print("verify: 失败, %d/%d 处不符" % (bad, M * TOPK))
        sys.exit(3)


def main():
    case = sys.argv[1]
    print("torch", torch.__version__, "| device:", torch.cuda.get_device_name(0))
    if case == "stage0":
        stage0()
    elif case == "verify":
        verify()
    elif case == "bf16":
        stage1(PRESETS["conservative"], bf16=True)
    elif case == "slack":
        # B/B_scale 多 8 行冗余 → 越界读落在界内
        stage1(PRESETS["conservative"], bf16=False, slack=8)
    elif case == "clamp":
        # 张量精确尺寸，发射前钳 expert_ids → 验证修复
        stage1(PRESETS["conservative"], bf16=False, clamp=True)
    elif case == "down":
        # 下行投影形状：A[M*10, 160] int8, B[128, 2560, 160], K=160, BK=64 不整除
        stage2({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
                "waves_per_eu": 0, "matrix_instr_nonkdim": 16, "kpack": 1})
    elif case == "down32":
        # 同上但 BK=32 整除 160
        stage2({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
                "waves_per_eu": 0, "matrix_instr_nonkdim": 16, "kpack": 1})
    elif case == "negpad":
        # 引擎 CUDA-graph 补齐场景：topk_ids 后半部分为 -1
        stage3_negpad()
    elif case == "scatter":
        # 采样器现场:[64, 248320] fp32 全词表 scatter_（引擎崩点原样复刻）
        sampler_case(do_scatter=True)
    elif case == "gather":
        # 同形状 gather 替代（候选修法）
        sampler_case(do_scatter=False)
    elif case == "sampler_vllm":
        sampler_vllm_case(strided=False)
    elif case == "sampler_vllm_view":
        sampler_vllm_case(strided=True)
    elif case == "chain":
        # vllm top-k/top-p pytorch 链逐步:sort→gather→mask→softmax→cumsum→masked_fill_→scatter_
        chain_case()
    elif case == "decode1":
        # decode 步形状:M=1, topk=10(引擎炸点);补丁 clamp 在挂载的 fused_moe 里生效
        stage3(False, M=1)
    elif case == "mm":
        mm_case()
    elif case in ("emap", "emapclamp"):
        # 忠实模拟引擎 TP4:全局 512 专家，本 rank 持 128，expert_map 3/4 为 -1
        # moe_align_block_size 内部会 expert_map[expert_ids] 重映射（含越界 gather）
        # emap=不钳位（现状复现），emapclamp=双向钳位（验证修复）
        stage3(clamp=(case == "emapclamp"))
    elif case in PRESETS:
        stage1(PRESETS[case], bf16=False)
    else:
        print("未知 case:", case)
        sys.exit(2)
    print("VERDICT: PASS", case)


if __name__ == "__main__":
    main()
