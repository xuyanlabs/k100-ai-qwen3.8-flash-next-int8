#!/bin/bash
python3 - <<'PY'
import json
import time
import urllib.request


def post(payload, timeout=1800):
    req = urllib.request.Request(
        "http://127.0.0.1:8388/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    n = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                if ch.get("delta", {}).get("content"):
                    if ttft is None:
                        ttft = time.time() - t0
                    n += 1
    t1 = time.time()
    ct = usage.get("completion_tokens", n)
    dec = t1 - t0 - (ttft or 0)
    return ttft, ct, dec, t1 - t0


for label, q, mt in [
    ("短问答", "3+5 等于几？只回答数字。", 256),
    ("长生成", "写一段 300 字左右的短文，介绍杭州的西湖，要求条理清晰。", 400),
]:
    ttft, ct, dec, tot = post({
        "model": "qwen3.8-flash-next",
        "messages": [{"role": "user", "content": q}],
        "max_tokens": mt, "temperature": 0.0,
        "stream": True, "stream_options": {"include_usage": True},
    })
    tps = ct / dec if dec > 0 else 0
    print(f"[{label}] TTFT={ttft:.2f}s  tokens={ct}  解码耗时={dec:.1f}s  "
          f"总耗时={tot:.1f}s  单流解码={tps:.2f} tok/s")
PY
