#!/usr/bin/env python3
# validate_align.py — 旧/新 _torch_moe_align_block_size 等价性对拍 (CPU)
import torch


def round_up(x, m):
    return ((x + m - 1) // m) * m


def cdiv(x, m):
    return (x + m - 1) // m


def old(topk_ids, block_size, num_experts, pad_sorted_ids):
    dev = topk_ids.device
    flat_all = topk_ids.flatten().to(torch.int64)
    raw_numel = flat_all.numel()
    flat = flat_all[(flat_all >= 0) & (flat_all < num_experts)]
    numel = flat.numel()
    mntp = raw_numel + num_experts * (block_size - 1)
    if pad_sorted_ids:
        mntp = round_up(mntp, block_size)
    if raw_numel < num_experts:
        mntp = min(raw_numel * block_size, mntp)
    need = numel + num_experts * (block_size - 1)
    if need > mntp:
        mntp = need
    counts = torch.bincount(flat, minlength=num_experts)
    seg = (counts + (block_size - 1)) // block_size * block_size
    npp = seg.sum().to(torch.int32).reshape(1)
    seg_start = torch.cumsum(seg, 0) - seg
    order = torch.argsort(flat, stable=True)
    sorted_flat = flat[order]
    raw_start = torch.cumsum(counts, 0) - counts
    rank = torch.arange(numel, device=dev) - raw_start[sorted_flat]
    dest = seg_start[sorted_flat] + rank
    sorted_ids = torch.full((mntp,), raw_numel, dtype=torch.int32, device=dev)
    if numel > 0:
        sorted_ids[dest] = order.to(torch.int32)
    mmb = cdiv(mntp, block_size)
    bpe = seg // block_size
    cum = torch.cumsum(bpe, 0)
    expert_ids = torch.searchsorted(cum, torch.arange(mmb, device=dev),
                                    right=True).to(torch.int32)
    return sorted_ids, expert_ids, npp


def new(topk_ids, block_size, num_experts, pad_sorted_ids):
    dev = topk_ids.device
    flat_all = topk_ids.flatten().to(torch.int64)
    raw_numel = flat_all.numel()
    valid = (flat_all >= 0) & (flat_all < num_experts)
    flat = torch.where(valid, flat_all, torch.full_like(flat_all, num_experts))
    mntp = raw_numel + num_experts * (block_size - 1)
    if pad_sorted_ids:
        mntp = round_up(mntp, block_size)
    if raw_numel < num_experts:
        mntp = min(raw_numel * block_size, mntp)
    counts_full = torch.zeros(
        num_experts + 1, dtype=torch.int64, device=dev
    ).scatter_add_(0, flat, torch.ones_like(flat))
    counts = counts_full[:num_experts]
    seg = (counts + (block_size - 1)) // block_size * block_size
    npp = seg.sum().to(torch.int32).reshape(1)
    seg_start = torch.cumsum(seg, 0) - seg
    order = torch.argsort(flat, stable=True)
    sorted_flat = flat[order]
    is_pad = sorted_flat >= num_experts
    flat_c = sorted_flat.clamp(max=num_experts - 1)
    raw_start = torch.cumsum(counts, 0) - counts
    rank = torch.arange(raw_numel, device=dev) - raw_start[flat_c]
    dest = torch.where(is_pad, torch.full_like(sorted_flat, mntp),
                       seg_start[flat_c] + rank)
    scratch = torch.full((mntp + 1,), raw_numel, dtype=torch.int32, device=dev)
    scratch[dest] = torch.where(is_pad, torch.full_like(order, -1), order).to(torch.int32)
    sorted_ids = scratch[:mntp]
    mmb = cdiv(mntp, block_size)
    bpe = seg // block_size
    cum = torch.cumsum(bpe, 0)
    expert_ids = torch.searchsorted(cum, torch.arange(mmb, device=dev),
                                    right=True).to(torch.int32)
    return sorted_ids, expert_ids, npp


torch.manual_seed(0)
fails = 0
cases = 0
for M, topk in [(1, 10), (1, 1), (4, 10), (10, 10), (64, 10), (300, 10),
                (512, 10), (1024, 10), (2048, 8)]:
    for E in (512, 64):
        for B in (16, 32, 64):
            for pad in (False, True):
                for mode in ("clean", "tailpad1", "tailpad3"):
                    cases += 1
                    ids = torch.randint(0, E, (M, topk))
                    # 引擎的填充永远是尾部整行 topk_ids = -1 (CUDA-graph 补齐)
                    if mode == "tailpad1" and M >= 2:
                        ids[-1, :] = -1
                    elif mode == "tailpad3" and M >= 4:
                        ids[-3:, :] = -1
                    a = old(ids.clone(), B, E, pad)
                    b = new(ids.clone(), B, E, pad)
                    npp = int(a[2])
                    nblk = cdiv(npp, B) if npp else 0
                    ok = (
                        int(a[2]) == int(b[2])
                        and b[0].numel() >= npp
                        and b[1].numel() >= nblk
                        and torch.equal(a[0][:npp], b[0][:npp])
                        and torch.equal(a[1][:nblk], b[1][:nblk])
                    )
                    if ok:
                        tail = b[0][npp:]
                        raw = int(ids.flatten().numel())
                        if tail.numel() and not bool((tail == raw).all()):
                            ok = False
                    if not ok:
                        fails += 1
                        if fails <= 5:
                            print(f"MISMATCH M={M} topk={topk} E={E} B={B} "
                                  f"pad={pad} mode={mode}")
                            print("   old sorted head:", a[0][:24].tolist())
                            print("   new sorted head:", b[0][:24].tolist())
                            print("   old expert head:", a[1][:12].tolist())
                            print("   new expert head:", b[1][:12].tolist())
                            print("   npp:", int(a[2]), int(b[2]),
                                  "len:", a[0].numel(), b[0].numel())
print(f"cases={cases} mismatches={fails}")
print("VALIDATE_DONE", flush=True)
