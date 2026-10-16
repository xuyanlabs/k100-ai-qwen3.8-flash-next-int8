#!/usr/bin/env bash
# =============================================================
# serve_cg_oai.sh — Qwen3.8-Flash-Next INT8 起服（TP4 + MTP + OpenAI 兼容面）
#
# 相对"官方镜像裸跑"的全部差异都在这份脚本 + patches/ 里：
#   1) --max-model-len 262144（原生上限）
#   2) MTP 投机解码：--speculative-config {"method":"mtp","num_speculative_tokens":1}
#   3) OpenAI 工具调用：--enable-auto-tool-choice --tool-call-parser qwen3_xml
#   4) 思维链分离：--reasoning-parser qwen3
#   5) 前缀缓存 PC=1（会同时把 mamba_cache_mode 切到 align）
#   6) 挂载 patches/ 覆盖镜像内失效的预编译核（见 README / 优化历程）
#
# 用法:
#   bash serve_cg_oai.sh                    # 用默认值起服
#   GPUS=0,1,2,3 MODEL_DIR=/data/models/qwen38-int8-w8a8 bash serve_cg_oai.sh
#   PREFLIGHT_ONLY=1 bash serve_cg_oai.sh   # 只做环境自检，不动任何容器
#
# 可调开关（都有默认值，按需覆盖）:
#   DOCKER_SOCK  docker socket（默认 unix:///var/run/docker.sock）
#   IMG          镜像名（默认官方镜像；换成自己 build 的 patched 镜像也可以）
#   DATA_DIR     数据盘根（默认 /data2）
#   MODEL_DIR    权重目录（默认 $DATA_DIR/qwen38-int8-w8a8）
#   GPUS         用哪几张卡（默认 4,5,6,7）
#   PORT         服务端口（默认 8388）
#   MAXLEN       上下文长度（默认 262144）
#   NSPEC        MTP 投机步数（默认 1）
#   CG_MODE      cudagraph 模式（默认 FULL_AND_PIECEWISE）
#   TOOL         工具解析器（默认 qwen3_xml；off 关闭）
#   REASON       思维链解析器（默认 qwen3；off 关闭）
#   PC           前缀缓存（默认 1；0 关闭，同时 mamba 走 classic）
# =============================================================
set -uo pipefail

SOCK="${DOCKER_SOCK:-unix:///var/run/docker.sock}"
IMG="${IMG:-harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.29.0-ubuntu22.04-dtk26.04-py3.10-20260831-qwen3.8flashnext}"
DATA_DIR="${DATA_DIR:-/data2}"
MODEL_DIR="${MODEL_DIR:-$DATA_DIR/qwen38-int8-w8a8}"
GPUS="${GPUS:-4,5,6,7}"
PORT="${PORT:-8388}"
MAXLEN="${MAXLEN:-262144}"
NSPEC="${NSPEC:-1}"
CG_MODE="${CG_MODE:-FULL_AND_PIECEWISE}"
TOOL="${TOOL:-qwen3_xml}"
REASON="${REASON:-qwen3}"
PC="${PC:-1}"
NAME="${NAME:-qwen38-tp4}"

PATCH_DIR="$DATA_DIR/patches"
MOE_DIR="$DATA_DIR/moe_configs"
CACHE_DIR="$DATA_DIR/torch_compile_cache"
VLLM_ROOT=/usr/local/lib/python3.10/dist-packages/vllm

[ "$TOOL" = "off" ] && TOOL=""
[ "$REASON" = "off" ] && REASON=""
if [ "$PC" = "1" ]; then PC_FLAG="--enable-prefix-caching"; else PC_FLAG="--no-enable-prefix-caching"; fi
CG_JSON="{\"cudagraph_mode\": \"$CG_MODE\"}"
SPEC_JSON="{\"method\": \"mtp\", \"num_speculative_tokens\": $NSPEC}"
D() { docker -H "$SOCK" "$@"; }

echo "########## $(date) ##########"
echo "镜像   : $IMG"
echo "权重   : $MODEL_DIR"
echo "配置   : MAXLEN=$MAXLEN  MTP nspec=$NSPEC  cudagraph=$CG_MODE  tool=${TOOL:-off}  reason=${REASON:-off}  prefix-cache=$PC"
echo "显卡   : $GPUS    端口: $PORT    容器名: $NAME"

echo; echo "=== 0. 起服前自检 ==="
fail=0
if docker -H "$SOCK" info >/dev/null 2>&1; then
  echo "  [OK] docker socket: $SOCK"
else
  echo "  [!!] 连不上 docker（$SOCK）。若你的 docker 在别处，用 DOCKER_SOCK=... 指定"; fail=1
fi
if D image inspect "$IMG" >/dev/null 2>&1; then
  echo "  [OK] 镜像已存在"
else
  echo "  [!!] 本地没有镜像：$IMG"
  echo "       请先 docker pull 该镜像，或用本仓库 Dockerfile 构建后 IMG=你的tag 起服"; fail=1
fi
if [ -f "$MODEL_DIR/config.json" ]; then
  echo "  [OK] 权重目录就位（$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)）"
else
  echo "  [!!] 权重目录不完整：$MODEL_DIR（缺 config.json）"; fail=1
fi
miss=""
for f in _custom_ops.py fused_topk_router.py fused_moe.py moe_align_block_size.py \
         ple_layer.py model.py warmup.py ct_w8a8_int8.py; do
  [ -f "$PATCH_DIR/$f" ] || miss="$miss $f"
done
if [ -z "$miss" ]; then
  echo "  [OK] 补丁齐全（$PATCH_DIR，8 个）"
else
  echo "  [!!] 缺补丁：$miss  （把仓库 patches/ 拷到 $PATCH_DIR）"; fail=1
fi
if [ -f "$MOE_DIR/E=512,N=160,device_name=K100_AI.json" ]; then
  echo "  [OK] MoE 调优配置就位"
else
  echo "  [!!] 缺 $MOE_DIR/E=512,N=160,device_name=K100_AI.json"; fail=1
fi
if [ -d /opt/hyhal ]; then
  echo "  [OK] /opt/hyhal 存在"
else
  echo "  [!!] 宿主缺 /opt/hyhal（由 DTK/驱动安装提供），容器内需要它"; fail=1
fi
if [ "$fail" = "1" ]; then
  echo; echo "自检未通过，已停在起服之前（什么容器都没动）。"; exit 1
fi
echo "  自检通过。"
if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
  echo "PREFLIGHT_ONLY=1，仅自检，未启动。"; exit 0
fi

echo; echo "=== 1. 停旧容器 / 卡位与端口 ==="
D rm -f "$NAME" 2>/dev/null
sleep 3
hy-smi 2>/dev/null | grep -E '^[0-9]' | awk '{print "  卡"NR-1": VRAM "$6}'
if ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
  echo "  [!!] 端口 $PORT 仍被占用，退出"; exit 1
fi
mkdir -p "$CACHE_DIR"

echo; echo "=== 2. 起容器（TP4 + MTP + OpenAI 面） ==="
D run -d --name "$NAME" \
  --network=host --shm-size=256g --privileged \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$MODEL_DIR":/models/qwen38-int8:ro \
  -v "$PATCH_DIR/ple_layer.py":$VLLM_ROOT/models/qwen4_exp/amd/ple_layer.py:ro \
  -v "$PATCH_DIR/fused_moe.py":$VLLM_ROOT/model_executor/layers/fused_moe/fused_moe.py:ro \
  -v "$PATCH_DIR/moe_align_block_size.py":$VLLM_ROOT/model_executor/layers/fused_moe/moe_align_block_size.py:ro \
  -v "$PATCH_DIR/warmup.py":$VLLM_ROOT/v1/worker/gpu/warmup.py:ro \
  -v "$PATCH_DIR/fused_topk_router.py":$VLLM_ROOT/model_executor/layers/fused_moe/router/fused_topk_router.py:ro \
  -v "$PATCH_DIR/ct_w8a8_int8.py":$VLLM_ROOT/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py:ro \
  -v "$PATCH_DIR/_custom_ops.py":$VLLM_ROOT/_custom_ops.py:ro \
  -v "$PATCH_DIR/model.py":$VLLM_ROOT/models/qwen4_exp/amd/model.py:ro \
  -e VLLM_TORCH_RESHAPE_CACHE=1 \
  -e VLLM_SKIP_WARMUP=1 \
  -e PYTHONFAULTHANDLER=1 \
  -e VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0 \
  -e K100_CG_JSON="$CG_JSON" \
  -e K100_SPEC_JSON="$SPEC_JSON" \
  -e K100_MAXLEN="$MAXLEN" \
  -e K100_TOOL="$TOOL" \
  -e K100_REASON="$REASON" \
  -e K100_PC_FLAG="$PC_FLAG" \
  -v "$CACHE_DIR":/root/.cache/vllm \
  -v "$MOE_DIR":/moe_configs:ro \
  -e VLLM_TUNED_CONFIG_FOLDER=/moe_configs \
  -e VLLM_NUMA_BIND=1 \
  -e HIP_VISIBLE_DEVICES="$GPUS" \
  "$IMG" bash -c 'source /opt/dtk/env.sh; source /opt/hyhal/env.sh; \
    echo "INNER maxlen=$K100_MAXLEN spec=$K100_SPEC_JSON tool=$K100_TOOL reason=$K100_REASON cg=$K100_CG_JSON pc=$K100_PC_FLAG"; \
    exec vllm serve /models/qwen38-int8 \
      --served-model-name qwen3.8-flash-next \
      --tensor-parallel-size 4 \
      --max-model-len "$K100_MAXLEN" \
      --max-num-batched-tokens 4096 \
      --max-num-seqs 64 \
      --speculative-config "$K100_SPEC_JSON" \
      ${K100_TOOL:+--enable-auto-tool-choice --tool-call-parser $K100_TOOL} \
      ${K100_REASON:+--reasoning-parser $K100_REASON} \
      --gpu-memory-utilization 0.97 \
      "$K100_PC_FLAG" \
      --compilation-config "$K100_CG_JSON" \
      --host 0.0.0.0 --port '"$PORT"''
echo "容器 $NAME 已起（卡 $GPUS）"

echo; echo "=== 3. 等就绪（首次含图编译，最长 40 分钟）==="
code=""
for i in $(seq 1 240); do
  code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null)
  if [ "$code" = "200" ]; then
    echo "READY！（约 $((i*10)) 秒）"
    break
  fi
  st=$(D inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null)
  if [ "$st" = "exited" ] || [ "$st" = "dead" ]; then
    echo "!! 容器退出了，最后 60 行日志："
    D logs --tail 60 "$NAME" 2>&1
    exit 1
  fi
  [ $((i % 12)) -eq 0 ] && echo "  ...$((i*10))s 还在加载/编译"
  sleep 10
done
[ "${code:-}" = "200" ] || { echo "!! 40 分钟没就绪"; exit 1; }

echo; echo "=== 4. 就绪自检 ==="
curl -s -m 10 http://127.0.0.1:$PORT/v1/models | head -c 300; echo
D logs "$NAME" 2>&1 | grep -E 'GPU KV cache size|Maximum concurrency|Available KV cache memory' | tail -3
echo
echo ">>> 起服成功。端口 $PORT，模型名 qwen3.8-flash-next"
echo ">>> 注意：第一个请求是冷启动，测速请跑第二遍（见 README 已知限制）"
