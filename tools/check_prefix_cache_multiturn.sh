#!/bin/bash
# harness 真实形态模拟：固定 system+工具段，历史逐轮增长，看每轮 TTFT
python3 - <<'PY'
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8388"
UNIT = "海光K100推理服务压测填充文本，内容固定，用于模拟工具定义与系统说明。"  # 32 字
SYSTEM = ("【固定 system + 工具定义 开始】\n" + UNIT * 150 + "\n【结束】\n")


def chat(messages, tag):
    body = json.dumps({
        "model": "qwen3.8-flash-next",
        "messages": messages,
        "max_tokens": 48,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    usage = {}
    out = ""
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
                if dl.get("content") or dl.get("reasoning"):
                    if first is None:
                        first = time.time() - t0
                    out += dl.get("content") or ""
    print(f"  [{tag}] prompt={usage.get('prompt_tokens')} tok  "
          f"TTFT={first:.2f}s  出字={out.strip()[:20]!r}")
    return out.strip()


msgs = [{"role": "system", "content": SYSTEM}]
for turn in range(1, 5):
    msgs.append({"role": "user", "content": f"第{turn}轮：请只回答 T{turn}"})
    ans = chat(msgs, f"第{turn}轮")
    msgs.append({"role": "assistant", "content": ans or "T"})
    # 模拟一轮工具返回（harness 每轮都会塞进上下文）
    msgs.append({"role": "user",
                 "content": "<tool_response>\n" + UNIT * (40 * turn) + "\n</tool_response>"})
PY
echo PC_HARNESS_DONE
