#!/bin/bash
# 修复后质量验收: 原始补全 (无思维链) + 一个 chat
python3 - <<'PY'
import json
import time
import urllib.request


def post(path, payload, timeout=900):
    req = urllib.request.Request(
        "http://127.0.0.1:8388" + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d, time.time() - t0


print("=== A) 原始补全 (贪心, 无需思维链) ===")
cases = [
    ("The capital of France is", 16),
    ("2, 4, 6, 8,", 16),
    ("1, 2, 3, 4, 5,", 16),
    ("水是生命之源，人体大约有", 24),
    ("def fibonacci(n):", 40),
    ("床前明月光，", 16),
]
for p, mt in cases:
    try:
        d, dt = post("/v1/completions", {
            "model": "qwen3.8-flash-next", "prompt": p,
            "max_tokens": mt, "temperature": 0.0,
        })
        c = d["choices"][0]
        print(f"  {p!r}")
        print(f"    -> {c['text']!r}   [{dt:.1f}s]")
    except Exception as e:
        print(f"  {p!r} EXC {type(e).__name__}: {str(e)[:200]}")

print()
print("=== B) chat (给足 token 让思考跑完) ===")
for q in ["3+5 等于几？只回答数字。", "中国的首都是哪座城市？只回答城市名。"]:
    try:
        d, dt = post("/v1/chat/completions", {
            "model": "qwen3.8-flash-next",
            "messages": [{"role": "user", "content": q}],
            "max_tokens": 256, "temperature": 0.0,
        })
        ch = d["choices"][0]
        txt = ch["message"]["content"]
        print(f"  Q: {q}")
        print(f"    finish={ch.get('finish_reason')} usage={d.get('usage',{}).get('completion_tokens')} [{dt:.1f}s]")
        print(f"    A: {txt[-300:]!r}")
    except Exception as e:
        print(f"  Q: {q} EXC {type(e).__name__}: {str(e)[:200]}")
PY
