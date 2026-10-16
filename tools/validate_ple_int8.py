#!/usr/bin/env python3
# validate_ple_int8.py — 验证 int8 存储路径与旧 bf16 路径逐位一致
import torch

from vllm.models.qwen4_exp.amd import ple_layer as P
from vllm.models.qwen4_exp.common.ple import copy_ple_embedding_shard_


class SI:
    def __init__(self, s, e):
        self.org_vocab_start_index = s
        self.org_vocab_end_index = e


class FakeEmb(torch.nn.Module):
    def __init__(self, local_rows, dim, s, e, tp=2):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.zeros(local_rows, dim, dtype=torch.bfloat16)
        )
        self.shard_indices = SI(s, e)
        self.num_embeddings_per_partition = local_rows
        self.embedding_dim = dim
        self.tp_size = tp


ORG, ROWS_PER_RANK, DIM, SHARDS = 1000, 512, 8, 4
allok = True
for rank in (0, 1):
    s = rank * ROWS_PER_RANK
    e = s + ROWS_PER_RANK
    emb = FakeEmb(ROWS_PER_RANK, DIM, s, e)
    ref = emb.weight.data.clone()
    for si in range(SHARDS):
        shard_size = (ORG + SHARDS - 1) // SHARDS
        cstart = si * shard_size
        n = max(0, min(shard_size, ORG - cstart))
        if n == 0:
            continue
        q = torch.randint(-128, 128, (n, DIM), dtype=torch.int8)
        sc = torch.rand(n, 1) * 0.004 + 1e-4
        # 旧路径: 加载时反量化成 bf16 再摆放
        dq = (q.float() * sc.float()).to(torch.bfloat16)
        copy_ple_embedding_shard_(
            ref, dq, checkpoint_start=cstart, tp_start=s, tp_end=e
        )
        # 新路径: int8 + 每行 scale 摆放
        P._k100_store_int8_shard(emb, q, sc, checkpoint_start=cstart)
    recon = (emb.k100_q.float() * emb.k100_scale.unsqueeze(-1)).to(torch.bfloat16)
    same = torch.equal(recon, ref)
    allok &= same
    print(f"  rank {rank}: int8 重建 vs bf16 参考 逐位一致 = {same}")
    print(f"    rows={recon.shape[0]} dim={recon.shape[1]} "
          f"非零行={int((recon.abs().sum(-1) > 0).sum())}")

print()
print("=== gather 语义抽查 (masking + 反量化) ===")
emb = FakeEmb(ROWS_PER_RANK, DIM, 0, ROWS_PER_RANK, tp=1)
emb.k100_q = torch.randint(-128, 128, (ROWS_PER_RANK, DIM), dtype=torch.int8)
emb.k100_scale = torch.ones(ROWS_PER_RANK, dtype=torch.float32)
emb.k100_out_dtype = torch.bfloat16
ids = torch.tensor([[0, 0, 5, 7, 100, 300, 511, 511]], dtype=torch.int64)
got = P._k100_int8_ngram_embedding(emb, ids)
ref_g = (emb.k100_q[ids].float() * emb.k100_scale[ids].unsqueeze(-1)).to(torch.bfloat16)
print("  形状:", tuple(got.shape), " 与手工参比一致:", bool(torch.equal(got, ref_g)))
print("  PELEINT8 =", P._PLE_INT8)
print("VALIDATE_PLE_INT8_DONE" if allok else "VALIDATE_PLE_INT8_FAIL")
