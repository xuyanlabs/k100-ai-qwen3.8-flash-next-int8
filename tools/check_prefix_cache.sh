#!/bin/bash
# 前缀缓存 A/B：同一个长 prompt 连问两次 + 换后缀再问一次
# 关闭时应三次都一样慢；开启后第 2/3 次 TTFT 应显著下降
python3 - <<'PY'
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8388"


def metrics():
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=8) as r:
            return r.read().decode(errors="ignore")
    except Exception:
        return ""


def pc_counters():
    out = {}
    for line in metrics().splitlines():
        if line.startswith("vllm:prefix_cache") and not line.startswith("#"):
            key = line.split("{")[0]
            out[key] = line.split()[-1]
    return out


UNIT = "海光K100的推理服务压测填充文本，内容固定以便复用前缀缓存。"  # 30 字
PREFIX = ("【固定前缀开始】系统说明：以下是一段用于验证前缀缓存的固定文本。\n"
          + UNIT * 260
          + "\n【固定前缀结束】\n")


def ask(suffix, tag):
    body = json.dumps({
        "model": "qwen3.8-flash-next",
        "messages": [{"role": "user", "content": PREFIX + suffix}],
        "max_tokens": 24,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    txt = ""
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
                if dl.get("content") or dl.get("reasoning"):
                    if first is None:
                        first = time.time() - t0
                    txt += dl.get("content") or ""
    pt = usage.get("prompt_tokens", 0)
    print(f"  [{tag}] prompt={pt} tok  TTFT={first:.2f}s  "
          f"输出={txt[-24:]!r}")
    return first, txt


print("=== 前缀缓存测试（固定前缀 %d 字）===" % len(PREFIX))
print("  计数起点:", pc_counters())
a1, o1 = ask("请只回答：AA", "第1次 同prompt")
a2, o2 = ask("请只回答：AA", "第2次 同prompt")
a3, o3 = ask("请只回答：BB", "第3次 换后缀")
print("  计数终点:", pc_counters())
same = (o1 == o2)
print(f"  第1/2 次输出一致: {same}")
PY
echo PC_TEST_DONE
