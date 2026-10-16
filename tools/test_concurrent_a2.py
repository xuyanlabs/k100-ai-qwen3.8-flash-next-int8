#!/usr/bin/env python3
# test_concurrent_a2.py — A 测试：1/4/8/20 并发下的单流速率 / 聚合吞吐 / TTFT / KV / MTP 接受率
# 与旧版差异：思维链字段在 0.29 里叫 reasoning（不是 reasoning_content），必须一起计入
# 用法: python3 test_concurrent_a2.py [端口] [max_tokens] [并发列表]
import json
import re
import statistics
import sys
import threading
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8388
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 400
LEVELS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 4, 8, 20]
BASE = f"http://127.0.0.1:{PORT}"
MODEL = "qwen3.8-flash-next"
PROMPT = "请详细介绍浙江省舟山市的渔业发展历史与现状，分章节展开，不少于500字。"


def http_json(path, timeout=10):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return json.loads(r.read())


def metrics():
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=8) as r:
            return r.read().decode(errors="ignore")
    except Exception:
        return ""


def spec_counters(txt):
    out = {}
    for key in ("drafts_total", "accepted_tokens_total", "draft_tokens_total"):
        m = re.search(
            r"^vllm:spec_decode_num_%s\{[^}]*\}\s+([0-9.eE+]+)\s*$" % key,
            txt, re.M)
        out[key] = float(m.group(1)) if m else None
    return out


def kv_peak():
    """返回 (KV 峰值百分比, 采样次数)"""
    peak = [0.0]
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            txt = metrics()
            for line in txt.splitlines():
                if line.startswith("vllm:kv_cache_usage_perc"):
                    try:
                        peak[0] = max(peak[0], float(line.split()[-1]))
                    except Exception:
                        pass
            stop.wait(0.5)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    return peak, stop


def one_request(result):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    nchunk = 0
    usage = {}
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
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
                    dl = ch.get("delta", {})
                    if dl.get("content") or dl.get("reasoning"):
                        if ttft is None:
                            ttft = time.time() - t0
                        nchunk += 1
        t1 = time.time()
        ct = usage.get("completion_tokens", nchunk)
        rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        result.update({
            "ok": True, "ttft": ttft, "total": t1 - t0, "completion": ct,
            "prompt": usage.get("prompt_tokens", 0), "reasoning": rt,
        })
    except Exception as e:
        result.update({"ok": False, "err": f"{type(e).__name__}: {str(e)[:140]}"})


def run_level(conc):
    m0 = spec_counters(metrics())
    peak, stop = kv_peak()
    results = [{} for _ in range(conc)]
    threads = [threading.Thread(target=one_request, args=(results[i],))
               for i in range(conc)]
    t0 = time.time()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.time() - t0
    time.sleep(1.0)
    stop.set()
    m1 = spec_counters(metrics())

    ok = [r for r in results if r.get("ok")]
    fail = [r for r in results if not r.get("ok")]

    print(f"\n[C={conc}] 成功 {len(ok)}/{conc}" + (f"  失败 {len(fail)}" if fail else ""))
    if fail:
        print(f"  首个错误: {fail[0].get('err')}")
    if not ok:
        return

    tot_out = sum(r["completion"] for r in ok)
    tot_in = sum(r["prompt"] for r in ok)
    ttfts = [r["ttft"] for r in ok if r["ttft"]]
    # 单流解码速率：排除各自 TTFT
    tps = [r["completion"] / max(r["total"] - (r["ttft"] or 0), 0.01) for r in ok]
    last_done = max(r["total"] for r in ok)
    first_tok = min((r["ttft"] or 0) for r in ok)
    decode_window = max(last_done - first_tok, 0.01)

    print(f"  墙钟 {wall:.1f}s   输入 {tot_in} tok   输出 {tot_out} tok")
    print(f"  聚合吞吐(含prefill) {tot_out/wall:.1f} tok/s")
    print(f"  聚合吞吐(解码窗)   {tot_out/decode_window:.1f} tok/s")
    print(f"  单流解码 tok/s: min {min(tps):.1f} / 中位 {statistics.median(tps):.1f} "
          f"/ max {max(tps):.1f}")
    if ttfts:
        print(f"  TTFT: min {min(ttfts):.2f}s / 中位 {statistics.median(ttfts):.2f}s "
              f"/ max {max(ttfts):.2f}s")
    print(f"  KV 峰值占用 {peak[0]*100:.1f}%")
    d = (m1["drafts_total"] or 0) - (m0["drafts_total"] or 0)
    a = (m1["accepted_tokens_total"] or 0) - (m0["accepted_tokens_total"] or 0)
    if d > 0:
        print(f"  MTP: drafts {d:.0f}  接受 {a:.0f}  接受率 {a/d*100:.1f}%")
    else:
        print("  MTP: (无 spec 计数变化)")


def main():
    for c in LEVELS:
        run_level(c)
        time.sleep(4)
    print("\n=== A 测试完成 ===")


if __name__ == "__main__":
    main()
