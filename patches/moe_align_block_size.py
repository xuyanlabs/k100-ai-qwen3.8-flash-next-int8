# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for Hygon K100-AI (gfx928) compatibility -- see patches/NOTICE.md

import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import triton
from vllm.utils.math_utils import round_up


def _torch_moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    pad_sorted_ids: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-torch drop-in for ops.moe_align_block_size.

    [patch] gfx928/CDNA2: the prebuilt HIP kernel leaves expert_ids and
    num_tokens_post_pad partially unfilled on this platform; with torch.empty
    the stale bytes (e.g. float bit patterns like 1065353182) become wild
    indices that VMFault the expert_map gather or the MoE kernel itself.
    This implementation produces identical outputs using only portable torch
    ops and never synchronizes with the host (everything stays on-device).
    """
    dev = topk_ids.device
    flat_all = topk_ids.flatten().to(torch.int64)
    raw_numel = flat_all.numel()
    # [patch] 引擎在 CUDA-graph 补齐时会给填充行 topk_ids=-1；原 C++ 核对负值
    # 的语义是"不参与计数"。先过滤再统计，哨兵取 raw_numel（= kernel 的
    # num_valid_tokens 阈值），保证填充槽位被 token_mask 正确屏蔽。
    flat = flat_all[(flat_all >= 0) & (flat_all < num_experts)]
    numel = flat.numel()

    max_num_tokens_padded = raw_numel + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if raw_numel < num_experts:
        max_num_tokens_padded = min(raw_numel * block_size, max_num_tokens_padded)
    # 缓冲区至少要装下有效分配（过滤后可能比 raw 公式小，兜底放大）
    need = numel + num_experts * (block_size - 1)
    if need > max_num_tokens_padded:
        max_num_tokens_padded = need

    # per-expert token counts and their padded segment sizes
    counts = torch.bincount(flat, minlength=num_experts)
    seg = (counts + (block_size - 1)) // block_size * block_size
    npp = seg.sum().to(torch.int32).reshape(1)
    seg_start = torch.cumsum(seg, 0) - seg

    # scatter each (token, k) assignment into its expert's padded segment
    order = torch.argsort(flat, stable=True)
    sorted_flat = flat[order]
    raw_start = torch.cumsum(counts, 0) - counts
    rank_in_expert = torch.arange(numel, device=dev) - raw_start[sorted_flat]
    dest = seg_start[sorted_flat] + rank_in_expert
    sorted_ids = torch.full(
        (max_num_tokens_padded,), raw_numel, dtype=torch.int32, device=dev
    )
    if numel > 0:
        sorted_ids[dest] = order.to(torch.int32)

    # per-block expert id via the padded-block prefix sums (no host sync);
    # blocks past the filled range get the num_experts sentinel, matching the
    # original kernel's contract (the GEMM early-returns before reading them)
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    blocks_per_expert = seg // block_size
    cum_blocks = torch.cumsum(blocks_per_expert, 0)
    block_idx = torch.arange(max_num_m_blocks, device=dev)
    expert_ids = torch.searchsorted(cum_blocks, block_idx, right=True).to(torch.int32)

    return sorted_ids, expert_ids, npp


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Note: In the case of expert_parallel, moe_align_block_size initially
    considers all experts as valid and aligns all tokens appropriately.
    Before the function returns it marks the experts_ids that are not in
    the current GPU rank as -1 so the MoE matmuls could skip those blocks.
    This requires the num_experts input arg to be the num global experts.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.
    - expert_map: A tensor of shape [num_experts] that maps the expert index
        from the global space to the local index space of the current
        expert parallel shard. If the expert is not in the current expert
        parallel shard, the mapping is set to -1.
    - pad_sorted_ids: A flag indicating whether the sorted_token_ids length
        should be padded to a multiple of block_size,
    - ignore_invalid_experts: A flag indicating whether to ignore invalid
        experts. When False, all expert_ids in topk_ids will participate in
        counting and ranking, but invalid experts in expert_ids will be marked
        as -1. When True, all invalid expert_ids in topk_ids will be ignored
        and will not participate in counting or ranking, and there will be no
        -1 in expert_ids.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    # [patch] gfx928/CDNA2: ops.moe_align_block_size 的 HIP 移植在此平台
    # 填表不全（实测 num_tokens_post_pad 都是野值），改走纯 torch 实现。
    # ignore_invalid_experts=True 的路径未被本模型使用（TritonExperts 默认
    # False），如需启用请先验证。
    if ignore_invalid_experts:
        raise NotImplementedError(
            "torch fallback moe_align_block_size does not support "
            "ignore_invalid_experts=True on this platform"
        )
    sorted_ids, expert_ids, num_tokens_post_pad = _torch_moe_align_block_size(
        topk_ids, block_size, num_experts, pad_sorted_ids
    )

    if expert_map is not None:
        # clamp 把尾部哨兵收进 map 界内；尾部块 GEMM 早退永远不会被读
        expert_ids = expert_map[expert_ids.clamp(max=num_experts - 1)]

    return sorted_ids, expert_ids, num_tokens_post_pad


def batched_moe_align_block_size(
    max_tokens_per_batch: int, block_size: int, expert_num_tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given num_batches, max_tokens_per_batch, block_size and the number of
    valid-tokens in each batch, prepare sorted_token_ids, expert_ids and
    num_tokens_post_pad. sorted_token_ids, expert_ids and num_tokens_post_pad
    have the same semantics as in moe_align_block_size.

    This function is intended to be a drop in replacement for
    moe_align_batch_size for the batched case.

    Parameters:
    - max_tokens_per_batch (int): Number of tokens in each batch (both
        valid and invalid).
    - block_size (int): block_size to align the data to.
    - expert_num_tokens (torch.Tensor): expert_num_tokens[i], indicates
        the number of valid tokens in batch i.

    Returns:
    - sorted_token_ids (torch.Tensor): Torch tensor of size
        (num_batches * max_tokens_per_batch) indicating the token indices for
        that block.
    - expert_ids (torch.Tensor): Torch tensor of size
        ceil((num_batches * max_tokens_per_batch) / block_size) indicating
        what expert to use for each block.
    - num_tokens_post_pad (torch.Tensor): Torch tensor of size 1
        indicating the number of valid blocks with actual data to
        process. This is represented in terms of num tokens.
    Example:
    Let num_batches=5, max_tokens_per_batch=8, block_size=4, and
    expert_num_tokens=[2, 3, 0, 6, 8]. This expert_num_tokens tensor
    indicates that,
     - The first 2 tokens in the 0th batch are valid and the rest 6 are
     invalid (i.e. in the 2D hidden_states tensor of shape,
     [num_batches * max_tokens_per_batch, K], indices 0, 1 are valid)
     - The first 3 tokens in the 1st batch are valid. i.e. indices 8, 9, 10
     - 0 tokens in the 2nd batch are valid
     - first 6 tokens in the  3rd batch are valid. i.e. indices,
     24, 25, 26, 27, 28, 29
     - so on ...

     In this case,
      sorted_token_ids will be [0, 1, 40, 40,
                                8, 9, 10, 40,
                                24, 25, 26, 27,
                                28, 29, 40, 40,
                                32, 33, 34, 35,
                                36, 37, 38, 39,
                                40, 40, 40, 40,
                                (rest all 40, 40, 40, 40)
                                ...]
      Here, 40 represents an invalid index. as there is no token index 40.
      The gemm kernel using this sorted_token_ids is expected to skip the
      gemm computation when it encounters this invalid index.

      expert_ids will be [0, 1, 3, 3, 4, 5, 5, -1, -1, (rest all -1) ...]
      Here, -1 represents an invalid expert. The gemm kernel using this
      expert_ids is expected to skip the gemm computation when it encounters
      an expert of id -1.

      num_tokens_post_pad will be 24 as sorted_token_ids has valid entries
      until 24.
    """

    B = expert_num_tokens.size(0)
    device = expert_num_tokens.device

    # Round up so each batch can be split to blocks evenly.
    max_num_tokens_padded = B * round_up(max_tokens_per_batch, block_size)

    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=device)
    assert max_num_tokens_padded % block_size == 0
    max_num_m_blocks = max_num_tokens_padded // block_size
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=device)

    ops.batched_moe_align_block_size(
        max_tokens_per_batch,
        block_size,
        expert_num_tokens,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )

    return sorted_ids, expert_ids, num_tokens_post_pad


# ---------------------------------------------------------------------------
# [K100/gfx928 patch] CUDA-graph-capturable replacement for
# _torch_moe_align_block_size (the definition above is shadowed on purpose).
#
# The original pure-torch version cannot be captured into a CUDA graph:
#
#   flat = flat_all[(flat_all >= 0) & (flat_all < num_experts)]
#   torch.AcceleratorError: operation not permitted when stream is capturing
#   (hipErrorStreamCaptureUnsupported)
#
# Boolean-mask indexing produces an output whose *size* depends on the data,
# so it needs a host synchronisation, which graph capture forbids.  This
# variant keeps every tensor shape data-independent:
#
#   * invalid / -1 padding entries become the sentinel `num_experts` instead
#     of being filtered out (bincount then yields a fixed num_experts + 1
#     buckets and the sentinel bucket is dropped by a static slice);
#   * padding rows are scattered into one extra trailing scratch slot and
#     sliced away, so no boolean mask is ever used for a write;
#   * max_num_tokens_padded is computed only from the static raw_numel.
#
# Semantics are identical to the previous implementation: `-1` padding rows
# (inserted by the engine when it pads a batch) still do not participate in
# the counting/ranking, and `num_tokens_post_padded` still counts only valid
# assignments.
# ---------------------------------------------------------------------------
def _torch_moe_align_block_size(  # noqa: F811 - intentional shadowing
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    pad_sorted_ids: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = topk_ids.device
    flat_all = topk_ids.flatten().to(torch.int64)
    raw_numel = flat_all.numel()

    valid = (flat_all >= 0) & (flat_all < num_experts)
    flat = torch.where(
        valid, flat_all, torch.full_like(flat_all, num_experts)
    )

    # Upstream's tight, data-independent bound.  The previous variant used
    # `numel + num_experts * (block_size - 1)`, whose value depends on how
    # many entries survived filtering, i.e. on the data -> not capturable.
    # `numel <= raw_numel` makes the tight bound sufficient: the padded total
    # is at most numel * block_size <= raw_numel * block_size.
    max_num_tokens_padded = raw_numel + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if raw_numel < num_experts:
        max_num_tokens_padded = min(
            raw_numel * block_size, max_num_tokens_padded
        )

    # torch.bincount has no CUDA kernel in this ROCm build: it falls back to
    # the CPU implementation, which copies device<->host and therefore aborts
    # CUDA graph capture ("Cannot copy between CPU and CUDA tensors during CUDA
    # graph capture unless the CPU tensor is pinned").  Integer scatter_add is
    # exact (no float rounding, so order does not matter) and stays on device.
    counts_full = torch.zeros(
        num_experts + 1, dtype=torch.int64, device=dev
    ).scatter_add_(0, flat, torch.ones_like(flat))
    counts = counts_full[:num_experts]
    seg = (counts + (block_size - 1)) // block_size * block_size
    npp = seg.sum().to(torch.int32).reshape(1)
    seg_start = torch.cumsum(seg, 0) - seg

    order = torch.argsort(flat, stable=True)
    sorted_flat = flat[order]
    is_pad = sorted_flat >= num_experts
    flat_c = sorted_flat.clamp(max=num_experts - 1)
    raw_start = torch.cumsum(counts, 0) - counts
    rank_in_expert = torch.arange(raw_numel, device=dev) - raw_start[flat_c]
    dest = torch.where(
        is_pad,
        torch.full_like(sorted_flat, max_num_tokens_padded),
        seg_start[flat_c] + rank_in_expert,
    )

    scratch = torch.full(
        (max_num_tokens_padded + 1,), raw_numel, dtype=torch.int32, device=dev
    )
    scratch[dest] = torch.where(
        is_pad, torch.full_like(order, -1), order
    ).to(torch.int32)
    sorted_ids = scratch[:max_num_tokens_padded]

    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    blocks_per_expert = seg // block_size
    cum_blocks = torch.cumsum(blocks_per_expert, 0)
    block_idx = torch.arange(max_num_m_blocks, device=dev)
    expert_ids = torch.searchsorted(cum_blocks, block_idx, right=True).to(torch.int32)

    return sorted_ids, expert_ids, npp
