# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for Hygon K100-AI (gfx928) compatibility -- see patches/NOTICE.md

from collections.abc import Callable

import torch
from compressed_tensors.quantization import QuantizationStrategy

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    init_int8_linear_kernel,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

logger = init_logger(__name__)


# [patch] gfx928/K100-AI: TritonInt8ScaledMMLinearKernel 在本卡静默吐全零
# (单卡数值复现: OUT 全 0, REF 正常 -> logits 近均匀 -> 输出乱码)。
# 回退: 装载后把 int8 权重按 scale 反量化成 bf16, 前向走普通 bf16 GEMM。
# 稠密层(attn qkv/o、router gate、shared expert)体量小, 显存税可接受;
# MoE 专家 GEMM 走 fused_moe_kernel(数值已验证), 不在此路径。
class _TorchInt8DequantKernel:
    """int8 权重 -> bf16 反量化常驻, apply 走 torch bf16 linear。"""

    def __init__(self, layer_param_names):
        del layer_param_names

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w = layer.weight.data  # int8 [N, K]
        ws = layer.weight_scale.data.float()
        if ws.dim() == 1:  # per-tensor
            if ws.numel() != 1:
                raise NotImplementedError(
                    "[patch] multi-partition per-tensor int8 scale 未适配"
                )
            ws = ws.reshape(1, 1)
        wdq = (w.float() * ws).to(torch.bfloat16)
        layer.weight = torch.nn.Parameter(wdq, requires_grad=False)
        if not torch.compiler.is_compiling():
            print(
                f"[patch] int8->bf16 dequant linear: {tuple(wdq.shape)}",
                flush=True,
            )

    def apply_weights(self, layer, x, bias):
        y = torch.nn.functional.linear(
            x.to(torch.bfloat16), layer.weight.data
        )
        if bias is not None:
            y = y + bias
        return y


class CompressedTensorsW8A8Int8(CompressedTensorsScheme):
    def __init__(
        self, strategy: str, is_static_input_scheme: bool, input_symmetric: bool
    ):
        self.strategy = strategy
        self.is_static_input_scheme = is_static_input_scheme
        self.input_symmetric = input_symmetric

    @classmethod
    def get_min_capability(cls) -> int:
        # turing and up
        return 75

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        layer.logical_widths = output_partition_sizes

        # [patch] gfx928: 绕过 TritonInt8ScaledMMLinearKernel(全零), 用反量化回退
        self.kernel = _TorchInt8DequantKernel(
            layer_param_names=[
                "weight",
                "weight_scale",
                "input_scale",
                "input_zero_point",
                "azp_adj",
            ],
        )

        # WEIGHT
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes), input_size_per_partition, dtype=torch.int8
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        if self.strategy == QuantizationStrategy.CHANNEL:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
        else:
            assert self.strategy == QuantizationStrategy.TENSOR
            weight_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE
        input_zero_point = None
        input_scale = None
        if self.is_static_input_scheme:
            input_scale = BasevLLMParameter(
                data=torch.empty(1, dtype=torch.float32), weight_loader=weight_loader
            )
            if not self.input_symmetric:
                # Note: compressed-tensors stores the zp using the same dtype
                # as the weights
                # AZP loaded as int8 but used as int32
                input_zero_point = BasevLLMParameter(
                    data=torch.empty(1, dtype=torch.int8), weight_loader=weight_loader
                )

        layer.register_parameter("input_zero_point", input_zero_point)
        layer.register_parameter("input_scale", input_scale)
        if not hasattr(layer, "azp_adj"):
            layer.register_parameter("azp_adj", None)

    # Checkpoints are serialized in compressed-tensors format, which is
    # different from the format the kernel may want. Handle repacking here.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)
