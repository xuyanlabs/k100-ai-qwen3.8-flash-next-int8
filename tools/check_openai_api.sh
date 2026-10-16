#!/bin/bash
# OpenAI 兼容面联测：max_model_len / tool calling / reasoning_content / 速度
echo "=== [1] /v1/models ==="
curl -s -m 10 http://127.0.0.1:8388/v1/models | python3 -c "import sys,json;d=json.load(sys.stdin);print(json.dumps(d,ensure_ascii=False)[:600])"
echo
echo "=== [2] tool calling (OpenAI tools 参数) ==="
python3 - <<'PY'
import json, urllib.request
payload = {
    "model": "qwen3.8-flash-next",
    "messages": [{"role": "user", "content": "北京现在天气怎么样？"}],
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询某城市当前天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名"}},
                "required": ["city"],
            },
        },
    }],
    "tool_choice": "auto",
    "max_tokens": 512,
    "temperature": 0.0,
}
req = urllib.request.Request(
    "http://127.0.0.1:8388/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    ch = d["choices"][0]
    m = ch["message"]
    print("finish_reason:", ch.get("finish_reason"))
    print("tool_calls:", json.dumps(m.get("tool_calls"), ensure_ascii=False))
    print("reasoning_content:", (m.get("reasoning_content") or "")[:200])
    print("content:", (m.get("content") or "")[:200])
except Exception as e:
    print("EXC", type(e).__name__, str(e)[:400])
PY
echo
echo "=== [3] 流式 + reasoning_content 分片 ==="
python3 - <<'PY'
import json, urllib.request
payload = {
    "model": "qwen3.8-flash-next",
    "messages": [{"role": "user", "content": "9+4 等于几？只回答数字。"}],
    "max_tokens": 256, "temperature": 0.0, "stream": True,
    "stream_options": {"include_usage": True},
}
req = urllib.request.Request(
    "http://127.0.0.1:8388/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
seen_r = seen_c = 0
usage = {}
txt = ""
try:
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            o = json.loads(data)
            if o.get("usage"):
                usage = o["usage"]
            for c in o.get("choices", []):
                dl = c.get("delta", {})
                if dl.get("reasoning_content"):
                    seen_r += 1
                if dl.get("content"):
                    seen_c += 1
                    txt += dl["content"]
finally:
    pass
print(f"reasoning 分片数={seen_r}  content 分片数={seen_c}")
print("最终 content:", repr(txt[-80:]))
print("usage:", usage)
PY
echo
echo "=== [4] 长上下文上限探测 (262144) ==="
python3 - <<'PY'
import json, urllib.request
# 用重复文本堆一个 ~20k token 的 prompt，确认 262144 下不报 max_model_len 错
filler = "这是一段用于填充上下文的测试文本。" * 1400
payload = {
    "model": "qwen3.8-flash-next",
    "messages": [{"role": "user", "content": filler + "\n\n请只回答：OK"}],
    "max_tokens": 16, "temperature": 0.0,
}
req = urllib.request.Request(
    "http://127.0.0.1:8388/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    print("prompt_tokens:", d.get("usage", {}).get("prompt_tokens"))
    print("answer:", repr(d["choices"][0]["message"].get("content")))
except Exception as e:
    print("EXC", type(e).__name__, str(e)[:400])
PY
echo
echo "=== [5] 速度（MTP 应仍是 ~31 tok/s）==="
bash /data2/bench_single_stream.sh 2>&1 | tail -2
echo OAI_TEST_DONE
