# patches/ 目录的许可与改动声明

本目录下的 8 个文件**都是 [vLLM](https://github.com/vllm-project/vllm) 源码的修改版**（不是本项目的原创代码）。它们保留着上游的许可头：

```
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

因此这 8 个文件**继续适用 Apache-2.0**（完整文本见同目录 `LICENSE-APACHE-2.0.txt`），
而不是本仓库根目录那份 0BSD。

按 Apache-2.0 第 4(b) 条的要求，下面声明**我们改了这些文件**，并逐文件说明改了什么。
每个文件的头部也内嵌了一行改动提示。

## 改动清单

| 文件 | 覆盖/对应的上游模块 | 改了什么 | 为什么 |
|---|---|---|---|
| `_custom_ops.py` | `vllm/_custom_ops.py` | 用 `torch.library` 覆盖若干 `_C` 算子的 CUDA 分派（`silu_and_mul`、`moe_sum`、`top_k_per_row_decode` 等），改为 Triton / 纯 torch 实现 | 这批预编译算子在 gfx928 上不报错但输出空（全零或哨兵值），导致整模型乱码 |
| `moe_align_block_size.py` | `.../fused_moe/moe_align_block_size.py` | 用纯 torch 重写（bincount→scatter_add_、argsort、cumsum、searchsorted），过滤负值填充行；padding 尺寸改为静态上界 | 原实现填表失败；重写后同时满足 cudagraph 捕获的静态形状要求 |
| `fused_moe.py` | `.../fused_moe/fused_moe.py` | 发射内核前对 `expert_ids` 做双边 clamp | 防止野值索引越界（越界块输出被 mask 丢弃，数值无影响） |
| `fused_topk_router.py` | `.../fused_moe/router/fused_topk_router.py` | `topk_softmax` / `topk_sigmoid` 改为 fp32 纯 torch 实现（softmax/sigmoid→topk→可选归一→pad 行清零） | 上游调用的预编译 topk 核在 gfx928 上不填输出，prefill 期全民投专家 0、decode 期野值越界 |
| `ple_layer.py` | `vllm/models/qwen4_exp/amd/ple_layer.py` | ①加载器支持 int8 分片 + 逐行 scale（配对缓冲、fp32 反量化）②N-gram 表以 int8 常驻显存，gather 后再反量化，不在加载期构造 bf16 大表 | 镜像里的加载器只认 BF16 布局（起不来）；bf16 常驻表吃掉 23.84 GiB/卡，改 int8 后让出约 12 GiB 给 KV 池 |
| `model.py` | `vllm/models/qwen4_exp/amd/model.py` | 配合 PLE int8 的加载路径调整 | 同上 |
| `warmup.py` | `vllm/v1/worker/gpu/warmup.py` | 增加 `VLLM_SKIP_WARMUP=1` 开关 | 上游 warmup 在本卡的集合通信上自旋挂死（70 分钟不结束） |
| `ct_w8a8_int8.py` | `.../compressed_tensors/schemes/compressed_tensors_w8a8_int8.py` | W8A8 INT8 方案路径调整 | 配合稠密层反量化回 bf16 的绕行方案 |

## 细分差异

改动相对上游原文件的 unified diff 放在仓库的 `diffs/` 目录（**5 份**，覆盖有原文件可比对的那几个；另外 3 个是整段替换型改动，见 `diffs/README.md` 的说明）。

## 提醒

- 这些补丁是针对**特定镜像版本**（`vllm0.29.0-...-dtk26.04-...qwen3.8flashnext`）写的，直接套到其它版本的 vLLM 上大概率不适用。
- 补丁里保留了上游代码结构，仅做兼容性修复；不含任何私有信息。
