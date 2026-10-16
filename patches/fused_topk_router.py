# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for Hygon K100-AI (gfx928) compatibility -- see patches/NOTICE.md
from collections.abc import Callable

import os

import torch
import triton
import triton.language as tl

import vllm._custom_ops as ops
import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter


def _get_padding_mask(num_tokens: int) -> torch.Tensor | None:
    if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
        is_padding = get_forward_context().is_padding
        return is_padding[:num_tokens] if is_padding is not None else None
    return None


# [patch] gfx928/K100-AI: ops.topk_softmax / ops.topk_sigmoid 是预编译 C++ 核,
# 没有 gfx928 target, torch.empty 输出缓冲整片不填 -> 预填充期读到全零(全民
# 投 expert 0, 输出质量崩), decode 期读到浮点位模式野值 -> 下游 VMFault。
# 纯 torch 重写, 语义与 CUDA 核一致(fp32 打分, topk, 可选归一, pad 行清零)。
_TOPK_PATCH_HERALD = {"printed": False}


def _topk_herald() -> None:
    if not _TOPK_PATCH_HERALD["printed"] and not torch.compiler.is_compiling():
        print("[patch] torch topk router active (gfx928)", flush=True)
        _TOPK_PATCH_HERALD["printed"] = True


def _torch_topk_fill(
    scores: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    renormalize: bool,
    is_padding: torch.Tensor | None,
) -> None:
    k = topk_indices.shape[1]
    w, ids = torch.topk(scores, k, dim=-1)
    if renormalize:
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    if is_padding is not None:
        w = w.masked_fill(is_padding.unsqueeze(-1), 0.0)
    topk_weights.copy_(w)
    topk_indices.copy_(ids.to(topk_indices.dtype))
    token_expert_indices.copy_(ids.to(token_expert_indices.dtype))


# ---------------------------------------------------------------------------
# [K100/gfx928 perf] Fused top-k router.
#
# The pure-torch fallback above costs ~74 us per call at M=1 in CUDA-graph
# replay (cast + softmax + topk + sum + div + 3 copies = ~6 kernel nodes, and
# a node costs ~13 us on this stack).  One Triton kernel does the whole thing:
# cast, softmax/sigmoid, top-k, renormalise and the three stores.  Measured
# 74.4 -> 31.3 us at M=1, and numerically identical to the torch path to
# <=3e-7 relative (torch's own fp32 rounding), i.e. far below bf16 resolution.
#
# Set K100_ROUTER_TRITON=0 to fall back to the torch implementation.
# ---------------------------------------------------------------------------
_ROUTER_USE_TRITON = os.environ.get("K100_ROUTER_TRITON", "1") != "0"
_ROUTER_HERALD = {"printed": False}


@triton.jit
def _k100_router_kernel(
    g_ptr, w_ptr, i_ptr, te_ptr, pad_ptr,
    M, E: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr,
    USE_SIGMOID: tl.constexpr, RENORM: tl.constexpr, HAS_PAD: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < M
    cols = tl.arange(0, E)
    x = tl.load(g_ptr + rows[:, None] * E + cols[None, :],
                mask=rmask[:, None], other=float("-inf")).to(tl.float32)
    if USE_SIGMOID:
        s = 1.0 / (1.0 + tl.exp(-x))
    else:
        mx = tl.max(x, axis=1, keep_dims=True)
        e = tl.exp(x - mx)
        s = e / tl.sum(e, axis=1, keep_dims=True)

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    cur = s
    for j in tl.static_range(K):
        val = tl.max(cur, axis=1)
        idx = tl.argmax(cur, axis=1)
        tl.store(i_ptr + rows * K + j, idx.to(i_ptr.dtype.element_ty), mask=rmask)
        tl.store(te_ptr + rows * K + j, idx.to(te_ptr.dtype.element_ty), mask=rmask)
        tl.store(w_ptr + rows * K + j, val, mask=rmask)
        acc += val
        cur = tl.where(cols[None, :] == idx[:, None], float("-inf"), cur)

    if RENORM:
        inv = 1.0 / (acc + 1e-20)
    else:
        inv = acc * 0.0 + 1.0
    if HAS_PAD:
        pad = tl.load(pad_ptr + rows, mask=rmask, other=0)
        inv = tl.where(pad, 0.0, inv)
    for j in tl.static_range(K):
        v = tl.load(w_ptr + rows * K + j, mask=rmask, other=0.0)
        tl.store(w_ptr + rows * K + j, v * inv, mask=rmask)


def _k100_router_triton(
    scoring: str,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    is_padding: torch.Tensor | None,
) -> bool:
    """Returns False when the fast path is not applicable."""
    M, E = gating_output.shape
    if not _ROUTER_USE_TRITON or M == 0 or E != triton.next_power_of_2(E):
        return False
    if gating_output.stride(-1) != 1:
        return False
    if not _ROUTER_HERALD["printed"] and not torch.compiler.is_compiling():
        print(f"[patch] fused triton router active (gfx928, E={E})", flush=True)
        _ROUTER_HERALD["printed"] = True
    k = topk_indices.shape[1]
    blk = max(1, min(16, triton.next_power_of_2(M)))
    pad = is_padding if is_padding is not None else gating_output
    _k100_router_kernel[(triton.cdiv(M, blk),)](
        gating_output, topk_weights, topk_indices, token_expert_indices, pad,
        M, E=E, K=k, BLOCK_M=blk,
        USE_SIGMOID=(scoring == "sigmoid"),
        RENORM=renormalize,
        HAS_PAD=(is_padding is not None),
        num_warps=4,
    )
    return True


def vllm_topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, ...]:
    _topk_herald()
    _pad = _get_padding_mask(topk_indices.shape[0])
    if not _k100_router_triton(
        "softmax", topk_weights, topk_indices, token_expert_indices,
        gating_output, renormalize, _pad,
    ):
        scores = torch.softmax(gating_output.float(), dim=-1)
        _torch_topk_fill(
            scores,
            topk_weights,
            topk_indices,
            token_expert_indices,
            renormalize,
            _pad,
        )
    return topk_weights, topk_indices


def vllm_topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, ...]:
    _topk_herald()
    _pad = _get_padding_mask(topk_indices.shape[0])
    if not _k100_router_triton(
        "sigmoid", topk_weights, topk_indices, token_expert_indices,
        gating_output, renormalize, _pad,
    ):
        scores = torch.sigmoid(gating_output.float())
        _torch_topk_fill(
            scores,
            topk_weights,
            topk_indices,
            token_expert_indices,
            renormalize,
            _pad,
        )

    return topk_weights, topk_indices


def dispatch_topk_softmax_func(
    use_rocm_aiter: bool = False,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    if use_rocm_aiter:
        return rocm_aiter_ops.topk_softmax
    return vllm_topk_softmax


def dispatch_topk_sigmoid_func(
    use_rocm_aiter: bool = False,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    if use_rocm_aiter:
        return rocm_aiter_ops.topk_sigmoid
    return vllm_topk_sigmoid


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: torch.dtype | None = None,
    scoring_func: str = "softmax",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    M, _ = hidden_states.size()

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(
        M,
        topk,
        dtype=torch.int32 if indices_type is None else indices_type,
        device=hidden_states.device,
    )
    token_expert_indices = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )

    if scoring_func == "softmax":
        topk_func = dispatch_topk_softmax_func(
            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()
        )
        topk_weights, topk_ids = topk_func(
            topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
        )

        return topk_weights, topk_ids, token_expert_indices
    elif scoring_func == "sigmoid":
        topk_func = dispatch_topk_sigmoid_func(
            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()
        )
        topk_weights, topk_ids = topk_func(
            topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
        )

        return topk_weights, topk_ids, token_expert_indices
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")


class FusedTopKRouter(BaseRouter):
    """Default router using standard fused top-k routing."""

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        scoring_func: str = "softmax",
        renormalize: bool = True,
        eplb_state: EplbLayerState | None = None,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
        )
        self.renormalize = renormalize
        self.scoring_func = scoring_func

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return get_routing_method_type(
            scoring_func=self.scoring_func,
            top_k=self.top_k,
            renormalize=self.renormalize,
            num_expert_group=None,
            has_e_score_bias=False,
        )

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing using standard fused top-k."""
        topk_weights, topk_ids, token_expert_indices = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            indices_type=indices_type,
            scoring_func=self.scoring_func,
        )

        return topk_weights, topk_ids
