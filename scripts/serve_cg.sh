#!/usr/bin/env bash
# =============================================================
# serve_qwen38_tp4.sh — Qwen3.8-Flash-Next INT8 TP4 起服（卡 4-7）
# docker socket 默认用系统标准位置，可用 DOCKER_SOCK 覆盖
# 用法：bash serve_qwen38_tp4.sh
# =============================================================
set -uo pipefail
SOCK="${DOCKER_SOCK:-unix:///var/run/docker.sock}"
IMG="harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.29.0-ubuntu22.04-dtk26.04-py3.10-20260831-qwen3.8flashnext"
GPUS="${GPUS:-4,5,6,7}"
TP="${TP:-$(echo "$GPUS" | awk -F, '{print NF}')}"
PORT="${PORT:-8388}"
D() { docker -H "$SOCK" "$@"; }

echo "########## $(date) ##########"
echo "=== 0. 卡位与端口自检 ==="
hy-smi 2>/dev/null | grep -E '^[0-9]' | awk '{print "  卡"NR-1": VRAM "$6}'
ss -tlnp 2>/dev/null | grep ":$PORT " && { echo "!! 端口 $PORT 被占用，换端口再来"; exit 1; }

echo; echo "=== 1. 起 TP4 容器 ==="
D rm -f qwen38-tp4 2>/dev/null
D run -d --name qwen38-tp4 \
  --network=host --shm-size=256g --privileged \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --group-add video \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /data2/qwen38-int8-w8a8:/models/qwen38-int8:ro \
  -v /data2/patches/ple_layer.py:/usr/local/lib/python3.10/dist-packages/vllm/models/qwen4_exp/amd/ple_layer.py:ro \
  -v /data2/patches/fused_moe.py:/usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe.py:ro \
  -v /data2/patches/moe_align_block_size.py:/usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/fused_moe/moe_align_block_size.py:ro \
  -v /data2/patches/warmup.py:/usr/local/lib/python3.10/dist-packages/vllm/v1/worker/gpu/warmup.py:ro \
  -v /data2/patches/fused_topk_router.py:/usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:ro \
  -v /data2/patches/ct_w8a8_int8.py:/usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py:ro \
  -v /data2/patches/_custom_ops.py:/usr/local/lib/python3.10/dist-packages/vllm/_custom_ops.py:ro \
  -e VLLM_TORCH_RESHAPE_CACHE=1 \
  -e VLLM_SKIP_WARMUP=1 \
  -e PYTHONFAULTHANDLER=1 \
  -e VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0 \
  ${GDN_DECODE_DEBUG:+-e GDN_DECODE_DEBUG=1} \
  ${EAGER:+-e EAGER=1} \
  ${GDN_DECODE_DEBUG:+-v /data2/patches/qwen_gdn_linear_attn.py:/usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:ro} \
  ${TORCH_INDEX_DEBUG:+-e TORCH_INDEX_DEBUG=1} \
  ${TORCH_INDEX_DEBUG:+-v /data2/patches/sitecustomize.py:/usr/lib/python3.10/sitecustomize.py:ro} \
  ${QSADBG:+-e QSADBG=1} \
  ${QSA_TORCH_TOPK:+-e QSA_TORCH_TOPK=1} \
  ${MOEDBG:+-e MOEDBG=1} \
  ${HIDPROBE:+-e HIDPROBE=1} \
  ${EMBDBG:+-e EMBDBG=1} \
  ${MOENAN:+-e MOENAN=1} \
  ${MOENAN:+-v /data2/patches/qwen3_next.py:/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/qwen3_next.py:ro} \
  -v /data2/patches/model.py:/usr/local/lib/python3.10/dist-packages/vllm/models/qwen4_exp/amd/model.py:ro \
  ${QSADBG:+-v /data2/patches/qsa_ops.py:/usr/local/lib/python3.10/dist-packages/vllm/models/qwen4_exp/amd/ops/qsa.py:ro} \
  ${QSADBG:+-v /data2/patches/indexer_qsa.py:/usr/local/lib/python3.10/dist-packages/vllm/models/qwen4_exp/amd/indexer_qsa.py:ro} \
  ${PLE_OFF:+-v /data2/patches/config_pleoff.json:/models/qwen38-int8/config.json:ro} \
  -v /data2/torch_compile_cache:/root/.cache/vllm \
  -v /data2/moe_configs:/moe_configs:ro \
  -e VLLM_TUNED_CONFIG_FOLDER=/moe_configs \
  -e VLLM_NUMA_BIND=1 \
  -e K100_TP="$TP" \
  -e HIP_VISIBLE_DEVICES="$GPUS" \
  "$IMG" bash -c 'source /opt/dtk/env.sh; source /opt/hyhal/env.sh; \
    exec vllm serve /models/qwen38-int8 \
      --served-model-name qwen3.8-flash-next \
      --tensor-parallel-size "$K100_TP" \
      --max-model-len 131072 \
      --max-num-batched-tokens 4096 \
      --max-num-seqs 64 \
      ${EAGER:+--enforce-eager} \
      --gpu-memory-utilization 0.97 \
      --no-enable-prefix-caching \
      --compilation-config '"'"'{"cudagraph_mode": "FULL_AND_PIECEWISE"}'"'"' \
      --host 0.0.0.0 --port '"$PORT"''
echo "容器 qwen38-tp4 已起（卡 4,5,6,7）"

echo; echo "=== 2. 等就绪（125B INT8 加载+编译，耐心）==="
for i in $(seq 1 240); do
  code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null)
  if [ "$code" = "200" ]; then
    echo "READY！（约 $((i*10)) 秒）"
    break
  fi
  st=$(D inspect -f '{{.State.Status}}' qwen38-tp4 2>/dev/null)
  if [ "$st" = "exited" ] || [ "$st" = "dead" ]; then
    echo "!! 容器退出了，最后 40 行日志："
    D logs --tail 40 qwen38-tp4 2>&1
    exit 1
  fi
  [ $((i % 12)) -eq 0 ] && echo "  ...$((i*10))s 还在加载（正常，125B 模型）"
  sleep 10
done
[ "${code:-}" = "200" ] || { echo "!! 40 分钟没就绪"; exit 1; }

echo; echo "=== 3. 就绪自检 ==="
curl -s -m 10 http://127.0.0.1:$PORT/v1/models | head -c 400; echo
echo; echo ">>> TP4 起服成功。端口 $PORT，模型名 qwen3.8-flash-next"
