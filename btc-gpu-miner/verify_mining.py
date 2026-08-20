#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_mining.py
================
Self-test for the mining cryptography in bitcoin_miner.py.

It verifies the maths against GROUND TRUTH pulled from the real Bitcoin
blockchain (blockstream.info API):

  A. target/difficulty maths
  B. block-header assembly + double-SHA-256 against a real mined block
  C. merkle-root (coinbase + branch) computation against a real block
  D. the vectorised PyTorch hasher against hashlib on a synthetic job
  E. the miner's share-finding path (compare against target + re-verify)

Run:  python3 verify_mining.py
"""

import json
import sys
import time
import urllib.request

try:
    import torch
except ImportError:
    torch = None

import bitcoin_miner as bm          # noqa: E402


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "verify-mining"})
    with urllib.request.urlopen(req, timeout=30) as r:
        text = r.read().decode()
    try:
        return json.loads(text)
    except ValueError:
        return text.strip()   # some endpoints return a bare value (e.g. block hash)


def test_a_target_math():
    assert bm.target_from_diff(1) == bm.DIFF1_TARGET, "diff-1 target"
    t524 = bm.target_from_diff(524288)
    assert t524 < bm.DIFF1_TARGET
    assert t524 == bm.DIFF1_TARGET // 524288
    tw = bm.target_words(t524)
    assert sum(w << (224 - 32 * i) for i, w in enumerate(tw)) == t524, "words round-trip"
    assert tw[0] == 0, "top word of diff-524288 target is zero"
    print("  [A] target maths OK")


def test_b_real_block_header():
    height = 900000
    bh = get_json(f"https://blockstream.info/api/block-height/{height}")
    blk = get_json(f"https://blockstream.info/api/block/{bh}")
    # rebuild header the way a Stratum miner would, from pool-style fields
    header = (blk["version"].to_bytes(4, "little")
              + bytes.fromhex(blk["previousblockhash"])[::-1]   # internal order
              + bytes.fromhex(blk["merkle_root"])[::-1]
              + blk["timestamp"].to_bytes(4, "little")
              + int(blk["bits"]).to_bytes(4, "little")
              + blk["nonce"].to_bytes(4, "little"))
    digest = bm.sha256d(header)
    assert digest[::-1].hex() == blk["id"], \
        f"header hash mismatch: {digest[::-1].hex()} vs {blk['hash']}"
    print(f"  [B] header of block {height} ({bh[:16]}...) verified")


def test_c_real_merkle():
    height = 100000          # small block (~4 txs)
    bh = get_json(f"https://blockstream.info/api/block-height/{height}")
    blk = get_json(f"https://blockstream.info/api/block/{bh}")
    txids = get_json(f"https://blockstream.info/api/block/{bh}/txids")
    assert len(txids) >= 2

    # -- full tree: internal byte order, paired in block order (consensus rule)
    level = [bytes.fromhex(t)[::-1] for t in txids]
    path = 0
    branch = []
    while len(level) > 1:
        sibling_idx = path ^ 1
        sibling = level[sibling_idx] if sibling_idx < len(level) else level[-1]
        branch.append(sibling[::-1].hex())   # displayed order, as a pool sends it
        nxt = []
        for i in range(0, len(level), 2):
            a = level[i]
            b = level[i + 1] if i + 1 < len(level) else level[i]
            nxt.append(bm.sha256d(a + b))
        path //= 2
        level = nxt
    assert level[0][::-1].hex() == blk["merkle_root"], \
        f"merkle tree mismatch: {level[0][::-1].hex()} vs {blk['merkle_root']}"

    # -- pool path: coinbase leaf + branch (what the miner actually does)
    root2 = bm.merkle_root_with_branch(bytes.fromhex(txids[0]), branch)
    assert root2.hex() == blk["merkle_root"], "merkle_root_with_branch mismatch"
    print(f"  [C] merkle root of block {height} (branch of {len(branch)} hashes) verified")


def _synthetic_job_dict(extranonce1="aabbccddeeff"):
    from mock_pool import MockJob
    j = MockJob("verifyjob")
    return {
        "job_id": j.job_id, "prevhash": j.prevhash,
        "coinb1": j.coinb1, "coinb2": j.coinb2,
        "merkle_branch": j.branch, "version": j.version,
        "nbits": j.nbits, "ntime": j.ntime,
        "extranonce1": extranonce1, "extranonce2_size": 6,
        "share_diff": 0.00001,
    }


def test_d_torch_vs_hashlib():
    if torch is None:
        print("  [D] SKIPPED (no torch)")
        return
    job = _synthetic_job_dict()
    en2 = "010203040506"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sha = bm.TorchSha256(device)
    pre = bm.precompute_job(job, en2, device)
    nonces = [0, 1, 2, 3, 7, 65535, 1000000, 0xFFFFFFFE, 0xFFFFFFFF]
    n = torch.tensor(nonces, dtype=torch.int64, device=device)
    d2 = sha.hash_headers(n, pre)
    for i, nv in enumerate(nonces):
        words = [int(d2[i, k].item()) & 0xFFFFFFFF for k in range(8)]
        digest_be = b"".join(w.to_bytes(4, "big") for w in words)
        display = digest_be[::-1].hex()
        ref = bm.header_sha256d(job, en2, job["ntime"], f"{nv:08x}")[::-1].hex()
        assert display == ref, f"torch mismatch at nonce {nv}: {display} vs {ref}"
    print(f"  [D] torch hasher matches hashlib on {len(nonces)} nonces")


def test_e_share_finding():
    if torch is None:
        print("  [E] SKIPPED (no torch)")
        return
    job = _synthetic_job_dict()
    en2 = "1234567890ab"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sha = bm.TorchSha256(device)
    pre = bm.precompute_job(job, en2, device)
    target = bm.target_from_diff(job["share_diff"])
    tw = bm.target_words(target)

    found = None
    batch = 65536
    for base in range(0, 4_000_000, batch):
        nonces = torch.arange(base, base + batch, dtype=torch.int64, device=device)
        d2 = sha.hash_headers(nonces, pre)
        mask = bm.compare_le(d2, tw)
        if mask.any():
            i = int(mask.nonzero(as_tuple=False).flatten()[0].item())
            found = base + i
            words = [int(d2[i, k].item()) & 0xFFFFFFFF for k in range(8)]
            display = (b"".join(w.to_bytes(4, "big") for w in words)[::-1]).hex()
            break
    assert found is not None, "no share found in 4M nonces at diff 0.00001"
    # re-verify the recorded share independently
    ref = bm.header_sha256d(job, en2, job["ntime"], f"{found:08x}")[::-1].hex()
    assert ref == display, "recorded share hash does not match reference"
    assert int(ref, 16) <= target, "share does not meet target"
    print(f"  [E] share found at nonce {found} and independently re-verified")


def main():
    print("verify_mining.py - checking mining maths against ground truth\n")
    test_a_target_math()
    test_b_real_block_header()
    test_c_real_merkle()
    test_d_torch_vs_hashlib()
    test_e_share_finding()
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
