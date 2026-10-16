#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_model_download.py — 按 ModelScope 官方清单逐文件校验模型权重

为什么需要它：
    ModelScope 的 API 会给出每个文件的 sha256。但"只校验文件大小"是**查不出**
    下载损坏的：本项目就踩过这个坑——续传时把数据写错了偏移，文件尺寸完全正确、
    连 safetensors 头都能解析，只有内容坏了（症状是某个 id 区间的中文 token
    被喂进错误的词嵌入，英文却完全正常）。

    170 GiB 这种体量**该用断点续传就用**（不用才不现实），关键是在续传之后
    **逐文件比对官方 sha256**：尺寸只说明"下够了字节"，不说明"下对了内容"。
    校验不通过的文件，重下即可（重下时不要拿那份坏文件当续传基点）。

用法：
    python3 verify_model_download.py --dir /data2/qwen38-int8-w8a8
    python3 verify_model_download.py --dir /data2/qwen38-int8-w8a8 --size-only   # 快速只比大小
    python3 verify_model_download.py --dir /path --repo hygon/另一个模型

目标机不能访问 modelscope.cn 怎么办：
    在有网的机器上导出清单——
        python3 verify_model_download.py --dir /tmp --save-manifest ms_files.json
    （导出时不会真的去校验 /tmp，只要目录存在即可；也可直接 --dump-only）
    把 ms_files.json 拷过去，再离线校验——
        python3 verify_model_download.py --dir /data2/qwen38-int8-w8a8 --manifest ms_files.json

退出码：0 = 全部一致；1 = 有缺失/多余/不一致；2 = 取清单失败。
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DEFAULT_REPO = "hygon/Qwen3.8-Flash-Next-Channel-INT8-w8a8"

# 控制台编码兜底：某些环境（如 GBK 控制台）直接打印中文会抛 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def fetch_manifest(repo, revision="master", timeout=60):
    url = ("https://modelscope.cn/api/v1/models/%s/repo/files"
           "?Revision=%s&Recursive=True" % (repo, revision))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    files = (d.get("Data") or {}).get("Files") or []
    return [f for f in files if f.get("Type") == "blob"]


def sha256_of(path, chunk=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return "%.1f%s" % (n, u)
        n /= 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="权重所在目录")
    ap.add_argument("--repo", default=DEFAULT_REPO, help="ModelScope 仓库 id")
    ap.add_argument("--revision", default="master")
    ap.add_argument("--jobs", type=int, default=4, help="并行校验线程数")
    ap.add_argument("--size-only", action="store_true", help="只比大小（快，但查不出内容损坏）")
    ap.add_argument("--manifest", help="改用本地清单文件（离线校验）")
    ap.add_argument("--save-manifest", help="把拉到的清单存成文件，供离线机器使用")
    ap.add_argument("--dump-only", action="store_true",
                    help="只拉/存清单，不做校验（配合 --save-manifest）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        print("目录不存在: %s" % args.dir)
        return 2

    if args.manifest:
        print("清单: 本地文件 %s" % args.manifest)
        try:
            with open(args.manifest, "r", encoding="utf-8") as fh:
                files = json.load(fh)
        except Exception as e:
            print("读清单失败: %s: %s" % (type(e).__name__, e))
            return 2
    else:
        print("仓库: %s" % args.repo)
        try:
            files = fetch_manifest(args.repo, args.revision)
        except Exception as e:
            print("取清单失败: %s: %s" % (type(e).__name__, e))
            print("     若目标机连不上 modelscope.cn，请用 --manifest 指定在别处导出的清单")
            return 2
        if args.save_manifest:
            with open(args.save_manifest, "w", encoding="utf-8") as fh:
                json.dump(files, fh, ensure_ascii=False, indent=1)
            print("清单已保存: %s" % args.save_manifest)

    if not files:
        print("清单为空（仓库 id 或 revision 是否正确？）")
        return 2

    if args.dump_only:
        print("--dump-only：只导出清单，未做校验。共 %d 个文件" % len(files))
        return 0

    total = sum(int(f.get("Size") or 0) for f in files)
    print("清单: %d 个文件，共 %s" % (len(files), human(total)))
    print("目录: %s" % os.path.abspath(args.dir))
    print("模式: %s" % ("只比大小" if args.size_only else "逐文件 sha256"))
    print("-" * 68)

    missing, size_bad, sha_bad = [], [], []
    need_hash = []

    for f in files:
        rel = f["Path"]
        want_size = int(f.get("Size") or 0)
        p = os.path.join(args.dir, rel)
        if not os.path.isfile(p):
            missing.append(rel)
            continue
        got = os.path.getsize(p)
        if got != want_size:
            size_bad.append((rel, want_size, got))
            continue
        if not args.size_only:
            need_hash.append((rel, p, (f.get("Sha256") or "").lower()))

    # 目录里多出来的文件（提示用，不算失败）
    have = set()
    for root, _dirs, names in os.walk(args.dir):
        for n in names:
            have.add(os.path.relpath(os.path.join(root, n), args.dir).replace(os.sep, "/"))
    expect = set(f["Path"] for f in files)
    extra = sorted(have - expect)

    if need_hash:
        n_done = [0]
        t0 = time.time()

        def check(item):
            rel, p, want = item
            got = sha256_of(p) if want else ""
            n_done[0] += 1
            if not args.quiet and (n_done[0] % 10 == 0 or n_done[0] == len(need_hash)):
                el = time.time() - t0
                rate = n_done[0] / el if el > 0 else 0
                left = (len(need_hash) - n_done[0]) / rate if rate > 0 else 0
                sys.stdout.write("\r  已校验 %d/%d  剩余约 %.0fs      "
                                 % (n_done[0], len(need_hash), left))
                sys.stdout.flush()
            if want and got != want:
                return (rel, want, got)
            return None

        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            for bad in ex.map(check, need_hash):
                if bad:
                    sha_bad.append(bad)
        if not args.quiet:
            print()

    print("-" * 68)
    ok = True
    if missing:
        ok = False
        print("[缺失] %d 个文件：" % len(missing))
        for r in missing[:20]:
            print("    " + r)
        if len(missing) > 20:
            print("    ...（还有 %d 个）" % (len(missing) - 20))
    if size_bad:
        ok = False
        print("[大小不符] %d 个：" % len(size_bad))
        for rel, w, g in size_bad[:20]:
            print("    %s  期望 %s 实际 %s" % (rel, human(w), human(g)))
    if sha_bad:
        ok = False
        print("[sha256 不符] %d 个（内容损坏，必须重下这几个文件）：" % len(sha_bad))
        for rel, w, g in sha_bad[:20]:
            print("    %s\n        want %s\n        got  %s" % (rel, w, g))
        print("    处理办法：把上面这几个文件重下（重下时不要拿现有坏文件当续传基点），")
        print("              下完再跑一次本脚本确认。其余文件不用动。")
    if extra:
        print("[目录里的多余文件] %d 个（一般无害，可自行确认）：" % len(extra))
        for r in extra[:10]:
            print("    " + r)

    if ok:
        print("[结论] %d 个文件全部通过 %s"
              % (len(files), "大小校验" if args.size_only else "大小 + sha256 校验"))
        return 0
    print("[结论] 校验未通过（见上）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
