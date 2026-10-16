#!/bin/bash
echo "=== [1] 正确口径速度（reasoning + content 都算）==="
python3 - <<'PY'
import json, time, urllib.request

def run(tag, q, mt):
    payload = {
        "model": "qwen3.8-flash-next",
        "messages": [{"role": "user", "content": q}],
        "max_tokens": mt, "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        "http://127.0.0.1:8388/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    first = None
    n = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            s = raw.strip()
            if not s.startswith(b"data:"):
                continue
            body = s[5:].strip()
            if body == b"[DONE]":
                break
            o = json.loads(body)
            if o.get("usage"):
                usage = o["usage"]
            for c in o.get("choices", []):
                dl = c.get("delta", {})
                if dl.get("content") or dl.get("reasoning"):
                    if first is None:
                        first = time.time() - t0
                    n += 1
    tot = time.time() - t0
    ct = usage.get("completion_tokens", n)
    dec = tot - (first or 0)
    print(f"  [{tag}] TTFT={first:.2f}s 产出={ct} tok 分片={n} 解码={dec:.1f}s "
          f"单流={ct/dec if dec > 0 else 0:.2f} tok/s")

run("短问答", "3+5 等于几？只回答数字。", 256)
run("长生成", "写一段 300 字左右的短文，介绍杭州的西湖，要求条理清晰。", 400)
PY
echo
echo "=== [2] 流式工具调用 ==="
python3 - <<'PY'
import json, urllib.request
payload = {
    "model": "qwen3.8-flash-next",
    "messages": [{"role": "user", "content": "北京现在天气怎么样？"}],
    "tools": [{"type": "function", "function": {
        "name": "get_weather", "description": "查询城市天气",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}],
    "tool_choice": "auto", "max_tokens": 512, "temperature": 0.0, "stream": True,
}
req = urllib.request.Request(
    "http://127.0.0.1:8388/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
tc = {}
finish = None
reason = ""
content = ""
with urllib.request.urlopen(req, timeout=600) as r:
    for raw in r:
        s = raw.strip()
        if not s.startswith(b"data:"):
            continue
        body = s[5:].strip()
        if body == b"[DONE]":
            break
        o = json.loads(body)
        for c in o.get("choices", []):
            dl = c.get("delta", {})
            if c.get("finish_reason"):
                finish = c["finish_reason"]
            if dl.get("reasoning"):
                reason += dl["reasoning"]
            if dl.get("content"):
                content += dl["content"]
            for t in dl.get("tool_calls") or []:
                i = t.get("index", 0)
                e = tc.setdefault(i, {"name": "", "args": ""})
                if t.get("function", {}).get("name"):
                    e["name"] += t["function"]["name"]
                if t.get("function", {}).get("arguments"):
                    e["args"] += t["function"]["arguments"]
print("finish_reason:", finish)
print("tool_calls:", json.dumps(tc, ensure_ascii=False))
print("reasoning 前 80:", reason[:80])
print("content:", repr(content[:120]))
PY
echo
echo "=== [3] content 是否干净（不带 <think> 泄漏）==="
python3 - <<'PY'
import json, urllib.request
payload = {"model": "qwen3.8-flash-next",
           "messages": [{"role": "user", "content": "水是生命之源，人体大约有"}],
           "max_tokens": 400, "temperature": 0.0,
           "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request("http://127.0.0.1:8388/v1/chat/completions",
                            data=json.dumps(payload).encode(),
                            headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as r:
    d = json.loads(r.read())
m = d["choices"][0]["message"]
print("content:", repr((m.get("content") or "")[:200]))
print("reasoning:", repr((m.get("reasoning") or "")[:80]))
print("think 泄漏:", "<think>" in (m.get("content") or ""))
PY
echo OAI_TEST3_DONE
