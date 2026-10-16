# diffs/ 说明

这里放的是**补丁相对官方镜像内原文件的 unified diff**，方便审阅"到底改了什么"。

覆盖 **5 个**有原文件可比对的补丁：

| diff | 对应的补丁 |
|---|---|
| `fused_moe.py.diff` | `patches/fused_moe.py` |
| `moe_align_block_size.py.diff` | `patches/moe_align_block_size.py` |
| `ple_layer.py.diff` | `patches/ple_layer.py` |
| `model.py.diff` | `patches/model.py` |
| `warmup.py.diff` | `patches/warmup.py` |

另外 **3 个补丁没有 diff**：`_custom_ops.py`、`fused_topk_router.py`、`ct_w8a8_int8.py`——
它们是"整段替换型"改动（前者是新增的分派覆盖层，后两者近乎重写），与官方原文件逐行 diff 没有审阅价值。
要看改了什么，直接对照 `patches/NOTICE.md` 里的逐文件说明。

> 取原文件的方式：`docker run --rm --entrypoint cat <镜像> /usr/local/lib/python3.10/dist-packages/vllm/<路径>`，
> 再 `diff -u 原文件 补丁文件` 即可复现这些 diff。
