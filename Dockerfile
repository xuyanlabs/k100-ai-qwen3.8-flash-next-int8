# =============================================================
# 由官方镜像 + 本项目补丁构建一个"已打补丁"的镜像
#
# 用法:
#   docker build -t k100-qwen38-int8:patched .
#   IMG=k100-qwen38-int8:patched bash /data2/serve_cg_oai.sh
#
# 说明:
#   - 基础镜像体积约 42.9 GB，本步骤只新增不到 1 MB 的一层；
#     推到镜像仓库时也只会推这一层（基础层若已存在则复用）。
#   - 不想自己构建也行：直接跑 scripts/serve_cg_oai.sh，
#     它会把 patches/ 以只读方式挂载进官方镜像，效果等价。
# =============================================================
ARG BASE_IMAGE=harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.29.0-ubuntu22.04-dtk26.04-py3.10-20260831-qwen3.8flashnext
FROM ${BASE_IMAGE}

ARG VLLM_ROOT=/usr/local/lib/python3.10/dist-packages/vllm

# 1) 整类失效的 _C 预编译核 → 用 torch.library 覆盖分派
COPY patches/_custom_ops.py            ${VLLM_ROOT}/_custom_ops.py

# 2) Qwen4Exp(AMD) 模型侧：PLE 表 int8 常驻 + 模型改动
COPY patches/model.py                  ${VLLM_ROOT}/models/qwen4_exp/amd/model.py
COPY patches/ple_layer.py              ${VLLM_ROOT}/models/qwen4_exp/amd/ple_layer.py

# 3) MoE：对齐填表纯 torch 重写 / 越界 clamp
COPY patches/moe_align_block_size.py   ${VLLM_ROOT}/model_executor/layers/fused_moe/moe_align_block_size.py
COPY patches/fused_moe.py              ${VLLM_ROOT}/model_executor/layers/fused_moe/fused_moe.py

# 4) 路由：预编译 topk 核空转 → 纯 torch 重写
COPY patches/fused_topk_router.py      ${VLLM_ROOT}/model_executor/layers/fused_moe/router/fused_topk_router.py

# 5) int8 量化方案与 warmup
COPY patches/ct_w8a8_int8.py           ${VLLM_ROOT}/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py
COPY patches/warmup.py                 ${VLLM_ROOT}/v1/worker/gpu/warmup.py

# 6) MoE 调优配置（起服脚本用 VLLM_TUNED_CONFIG_FOLDER=/moe_configs 指向它）
COPY "configs/E=512,N=160,device_name=K100_AI.json" /moe_configs/

# 基础镜像自带的入口保持不变；起服请用手册里的 serve 脚本 /
# 或自己写 docker run（需挂 /opt/hyhal 与 /dev/kfd 等，见 README）。
