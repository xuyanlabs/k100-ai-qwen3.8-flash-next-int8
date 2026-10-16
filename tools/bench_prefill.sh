#!/bin/bash
echo "=== prefill 速率判定：单请求大 prompt vs 20 请求小 prompt ==="
python3 - <<'PY'
import json, time, urllib.request

BASE = "http://127.0.0.1:8388"

def ttft(prompt_tokens_target, tag):
    # 1 个中文字符 ≈ 1 token 左右，用重复文本逼近目标长度
    unit = "这是一段用于压测预填充的填充文本，长度适中。"
    filler = unit * max(1, prompt_tokens_target // 10)
    body = json.dumps({
        "model": "qwen3.8-flash-next",
        "messages": [{"role": "user", "content": filler + "\n只回答：OK"}],
        "max_tokens": 8, "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    usage = {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            s = line.strip()
            if not s.startswith(b"data:"):
                continue
            b = s[5:].strip()
            if b == b"[DONE]":
                break
            o = json.loads(b)
            if o.get("usage"):
                usage = o["usage"]
            for c in o.get("choices", []):
                dl = c.get("delta", {})
                if (dl.get("content") or dl.get("reasoning")) and first is None:
                    first = time.time() - t0
    pt = usage.get("prompt_tokens", 0)
    print(f"  [{tag}] prompt={pt} tok  TTFT={first:.2f}s  =>  prefill {pt/first if first else 0:.0f} tok/s")
    return pt, first

for n in (1500, 3000, 6000):
    ttft(n, f"单请求 ~{n} tok")
PY
echo PREFILL_PROBE_DONE
