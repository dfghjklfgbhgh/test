#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bitcoin_miner.py
================
A SHA-256d ("Bitcoin mining") worker written in Python + PyTorch.

It is designed to work OFFLINE with the GitHub-synced workflow:

    pool_connector.py --writes-->  jobs.txt  --(user pull)-->  THIS MINER
    THIS MINER       --writes-->  shares.txt --(user push)-->  pool_connector.py

* Reads jobs from `jobs.txt` (JSON array of job dicts produced by pool_connector.py).
* Mines the newest job by brute-forcing the 32-bit block-header nonce.
* Uses PyTorch (CUDA GPU when available, CPU fallback otherwise).
* Writes every valid share (header hash <= pool share target) into `shares.txt`.

IMPORTANT HONEST NOTE
---------------------
Real-world Bitcoin mining is done with specialised ASIC hardware; a GPU
mining in Python is many orders of magnitude too slow to ever find a share
at a real pool's difficulty.  This project is a working, end-to-end
educational implementation of the actual Stratum mining math (coinbase,
merkle root, block header, double-SHA-256, target comparison), and it is
fully testable against the bundled mock pool at low difficulty.

Usage
-----
    python3 bitcoin_miner.py --jobs jobs.txt --shares shares.txt

    python3 bitcoin_miner.py --jobs jobs.txt --diff 0.00001   # test difficulty
    python3 bitcoin_miner.py --jobs jobs.txt --max-nonces 500000 --stop-on-share
"""

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

# --------------------------------------------------------------------------
# SHA-256 constants
# --------------------------------------------------------------------------
K = [
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
]
H0 = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
]

MASK32 = 0xFFFFFFFF
# difficulty-1 share target (Bitcoin's famous 0x00000000FFFF0000...000)
DIFF1_TARGET = 0x00000000FFFF0000000000000000000000000000000000000000000000000000


def target_from_diff(diff: float) -> int:
    """Pool share target (as a 256-bit integer) for a given difficulty."""
    diff = float(diff)
    if diff <= 0:
        raise ValueError(f"difficulty must be > 0, got {diff}")
    return int(DIFF1_TARGET // diff)


def target_words(target: int):
    """256-bit target -> 8 big-endian word ints."""
    return [(target >> (224 - 32 * i)) & MASK32 for i in range(8)]


# --------------------------------------------------------------------------
# Pure-Python / hashlib reference math (authoritative)
# --------------------------------------------------------------------------
def sha256d(data: bytes) -> bytes:
    """Bitcoin's double SHA-256."""
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def build_coinbase(job: dict, extranonce2_hex: str) -> bytes:
    """coinbase tx = coinb1 || extranonce1 || extranonce2 || coinb2"""
    return (bytes.fromhex(job["coinb1"])
            + bytes.fromhex(job["extranonce1"])
            + bytes.fromhex(extranonce2_hex)
            + bytes.fromhex(job["coinb2"]))


def merkle_root_with_branch(leaf_digest: bytes, merkle_branch) -> bytes:
    """Combine the coinbase hash with the pool's merkle branch.

    Convention (verified against real mined blocks): the pool sends branch
    hashes in *displayed* byte order; internally they are byte-reversed.
    The coinbase is always the leftmost leaf (tx index 0), so it is the
    left child at every level:  parent = SHA256d(node || sibling).
    Returns the merkle root in displayed order.
    """
    h = leaf_digest[::-1]                 # internal byte order
    for bh in merkle_branch:
        b = bytes.fromhex(bh)[::-1]       # branch sibling, internal order
        h = sha256d(h + b)                # coinbase is the left child
    return h[::-1]                        # back to displayed order


def build_header_bytes(job: dict, extranonce2_hex: str, ntime_hex: str, nonce_hex: str) -> bytes:
    """Assemble the 80-byte block header from Stratum job fields."""
    merkle_display = merkle_root_with_branch(
        sha256d(build_coinbase(job, extranonce2_hex)), job["merkle_branch"])
    return (
        int(job["version"], 16).to_bytes(4, "little")   # version (LE)
        + bytes.fromhex(job["prevhash"])                # prevhash: already internal order
        + merkle_display[::-1]                          # merkle root: internal order
        + int(ntime_hex, 16).to_bytes(4, "little")      # time (LE)
        + int(job["nbits"], 16).to_bytes(4, "little")   # bits (LE)
        + int(nonce_hex, 16).to_bytes(4, "little")      # nonce (LE)
    )


def header_sha256d(job: dict, extranonce2_hex: str, ntime_hex: str, nonce_hex: str) -> bytes:
    """Full reference share hash (displayed-order digest)."""
    return sha256d(build_header_bytes(job, extranonce2_hex, ntime_hex, nonce_hex))


# --------------------------------------------------------------------------
# Scalar SHA-256 (used for the job-dependent precompute step)
# --------------------------------------------------------------------------
def _rotr(x: int, n: int) -> int:
    return ((x >> n) | (x << (32 - n))) & MASK32


def sha256_compress_scalar(state, w16):
    """state: list of 8 uint32; w16: list of 16 uint32 message words."""
    W = list(w16) + [0] * 48
    for t in range(16, 64):
        s0 = _rotr(W[t - 15], 7) ^ _rotr(W[t - 15], 18) ^ (W[t - 15] >> 3)
        s1 = _rotr(W[t - 2], 17) ^ _rotr(W[t - 2], 19) ^ (W[t - 2] >> 10)
        W[t] = (W[t - 16] + s0 + W[t - 7] + s1) & MASK32
    a, b, c, d, e, f, g, h = state
    for t in range(64):
        S1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
        ch = (e & f) ^ (~e & g)
        t1 = (h + S1 + ch + K[t] + W[t]) & MASK32
        S0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        t2 = (S0 + maj) & MASK32
        h, g, f, e, d, c, b, a = g, f, e, (d + t1) & MASK32, c, b, a, (t1 + t2) & MASK32
    return [(x + y) & MASK32 for x, y in zip(state, (a, b, c, d, e, f, g, h))]


def _i32(v):
    """Build an int32 torch scalar from an unsigned 32-bit value."""
    v = int(v) & MASK32
    return v if v < 2 ** 31 else v - 2 ** 32


# --------------------------------------------------------------------------
# Vectorised SHA-256 on torch tensors (int32, all arithmetic mod 2**32)
# --------------------------------------------------------------------------
class TorchSha256(object):
    """Batch SHA-256d tailored to the 80-byte Bitcoin header.

    Exploits the structure of the header:

        block1 = header[0:64]   (version|prevhash|merkle[0:28])   -- constant
        block2 = header[64:80] + padding (merkle[28:32], time, bits,
                 nonce, 0x80..., length = 640)

    So the first block is hashed once per (job, extranonce2); each candidate
    only hashes block2 + the second SHA-256 pass.  All tensors are int32;
    every operation is mod 2**32 (adds wrap, right shifts are masked).
    """

    def __init__(self, device):
        self.device = device
        self.Kt = torch.tensor([_i32(v) for v in K], dtype=torch.int32, device=device)
        self.H0t = torch.tensor([_i32(v) for v in H0], dtype=torch.int32, device=device)
        self._masks = {n: (1 << (32 - n)) - 1 for n in range(1, 32)}

    def _rotr(self, x, n):
        return ((x >> n) & self._masks[n]) | (x << (32 - n))

    def _stack_words(self, words, B):
        """words: list of (B,) int64 tensors or python ints -> (B,16) int32."""
        cols = []
        for w in words:
            if isinstance(w, torch.Tensor):
                cols.append(w)
            else:
                cols.append(torch.full((B,), int(w) & MASK32, dtype=torch.int64,
                                       device=self.device))
        return torch.stack(cols, dim=1).to(torch.int32)

    def expand(self, w16):
        """(B,16) -> (B,64) message schedule."""
        W = [w16[:, t] for t in range(16)]
        for t in range(16, 64):
            x = W[t - 15]
            s0 = self._rotr(x, 7) ^ self._rotr(x, 18) ^ ((x >> 3) & self._masks[3])
            y = W[t - 2]
            s1 = self._rotr(y, 17) ^ self._rotr(y, 19) ^ ((y >> 10) & self._masks[10])
            W.append(W[t - 16] + s0 + W[t - 7] + s1)
        return torch.stack(W, dim=1)

    def compress(self, state, W):
        """state (B,8) int32, W (B,64) int32 -> (B,8) int32."""
        a, b, c, d, e, f, g, h = [state[:, i] for i in range(8)]
        for t in range(64):
            S1 = self._rotr(e, 6) ^ self._rotr(e, 11) ^ self._rotr(e, 25)
            ch = (e & f) ^ ((~e) & g)
            t1 = h + S1 + ch + self.Kt[t] + W[:, t]
            S0 = self._rotr(a, 2) ^ self._rotr(a, 13) ^ self._rotr(a, 22)
            maj = (a & b) ^ (a & c) ^ (b & c)
            t2 = S0 + maj
            h, g, f, e, d, c, b, a = (g, f, e, d + t1, c, b, a, t1 + t2)
        return torch.stack([a, b, c, d, e, f, g, h], dim=1) + state

    def hash_headers(self, nonces, pre):
        """nonces: (B,) int64 nonce values -> digest2 (B,8) int32 (big-endian words).

        pre: dict from precompute_job() for the current (job, extranonce2).
        """
        B = nonces.shape[0]
        n = nonces
        w3 = (((n & 0xFF) << 24) | (((n >> 8) & 0xFF) << 16)
              | (((n >> 16) & 0xFF) << 8) | ((n >> 24) & 0xFF))
        w16 = self._stack_words(
            [pre["w0"], pre["w1"], pre["w2"], w3,
             0x80000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x280], B)
        W = self.expand(w16)
        state1 = pre["state1"].unsqueeze(0).expand(B, 8)
        d1 = self.compress(state1, W)

        w16b = self._stack_words(
            [d1[:, 0], d1[:, 1], d1[:, 2], d1[:, 3],
             d1[:, 4], d1[:, 5], d1[:, 6], d1[:, 7],
             0x80000000, 0, 0, 0, 0, 0, 0, 0x100], B)
        Wb = self.expand(w16b)
        state0 = self.H0t.unsqueeze(0).expand(B, 8)
        return self.compress(state0, Wb)


def precompute_job(job: dict, extranonce2_hex: str, device):
    """Everything that does not depend on the nonce, for (job, extranonce2)."""
    merkle_display = merkle_root_with_branch(
        sha256d(build_coinbase(job, extranonce2_hex)), job["merkle_branch"])
    merkle_internal = merkle_display[::-1]

    block1 = (int(job["version"], 16).to_bytes(4, "little")
              + bytes.fromhex(job["prevhash"])
              + merkle_internal[:28])
    w1 = [int.from_bytes(block1[i:i + 4], "big") for i in range(0, 64, 4)]
    state1 = sha256_compress_scalar(H0, w1)

    w0 = int.from_bytes(merkle_internal[28:32], "big")
    w1c = int.from_bytes(int(job["ntime"], 16).to_bytes(4, "little"), "big")
    w2c = int.from_bytes(int(job["nbits"], 16).to_bytes(4, "little"), "big")

    return {
        "w0": w0, "w1": w1c, "w2": w2c,
        "state1": torch.tensor([_i32(v) for v in state1], dtype=torch.int32, device=device),
    }


def _bswap64(t):
    """Byte-swap each 32-bit word of an (B,8) int64 tensor."""
    return (((t & 0xFF) << 24) | (((t >> 8) & 0xFF) << 16)
            | (((t >> 16) & 0xFF) << 8) | ((t >> 24) & 0xFF))


def compare_le(digest, tw):
    """digest (B,8) int32 = words of the *internal* digest.

    The block hash that must be <= target is the *displayed* hash
    (byte-reversed digest).  As a big-endian integer its words are
    byteswap(digest[7]), byteswap(digest[6]), ... byteswap(digest[0]).
    Returns (B,) bool: displayed-hash-int <= target.
    """
    x = digest.to(torch.int64) & 0xFFFFFFFF
    w0 = _bswap64(x[:, 7])
    lt = w0 < tw[0]
    eq = w0 == tw[0]
    for i in range(1, 8):
        wi = _bswap64(x[:, 7 - i])
        lt = lt | (eq & (wi < tw[i]))
        eq = eq & (wi == tw[i])
    return lt | eq


# --------------------------------------------------------------------------
# Share store (JSON array, dedup by share_id)
# --------------------------------------------------------------------------
class SharesStore(object):
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()

    def load(self):
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def add(self, new_shares):
        with self.lock:
            existing = {s["share_id"] for s in self.load()}
            fresh = [s for s in new_shares if s["share_id"] not in existing]
            if not fresh:
                return 0
            merged = self.load() + fresh
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(merged, f, indent=2)
            os.replace(tmp, self.path)
            return len(fresh)


# --------------------------------------------------------------------------
# Job loading
# --------------------------------------------------------------------------
def load_jobs(path):
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    return [j for j in data if isinstance(j, dict) and "job_id" in j]


def newest_job(jobs):
    return sorted(jobs, key=lambda j: j.get("received_at", 0))[-1] if jobs else None


# --------------------------------------------------------------------------
# Miner
# --------------------------------------------------------------------------
class Miner(object):
    def __init__(self, args):
        self.args = args
        if torch is None:
            sys.exit("PyTorch not installed.  Run:  pip install torch")
        use_cuda = torch.cuda.is_available() and not args.cpu
        self.device = torch.device("cuda" if use_cuda else "cpu")
        if use_cuda:
            print(f"[miner] using CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[miner] NOTE: using CPU (no CUDA / --cpu). Slow but works for testing.")
        self.sha = TorchSha256(self.device)
        self.shares = SharesStore(args.shares)
        self.batch = args.batch
        if self.batch > 2 ** 16 and self.device.type == "cpu":
            self.batch = 2 ** 16
        self.cur_en2 = None
        self.best = None      # (display_int, display_hex, nonce, meets_target)

    # ---------------------------------------------------------------- job
    def load_job(self):
        try:
            jobs = load_jobs(self.args.jobs)
        except Exception as e:
            print(f"[miner] cannot read {self.args.jobs}: {e}")
            return None
        job = newest_job(jobs)
        if job is None:
            return None
        if self.args.diff:
            job = dict(job)
            job["share_diff"] = float(self.args.diff)
            job["share_target_hex"] = f"{target_from_diff(self.args.diff):064x}"
        return job

    def _resume(self, job):
        try:
            with open(self.args.progress) as f:
                p = json.load(f)
            if p.get("job_id") == job["job_id"] and p.get("ntime") == job["ntime"]:
                return p["extranonce2"], int(p["nonce"])
        except Exception:
            pass
        return None, 0

    def _save_progress(self, job, extranonce2, nonce):
        try:
            tmp = self.args.progress + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"job_id": job["job_id"], "ntime": job["ntime"],
                           "extranonce2": extranonce2, "nonce": nonce}, f)
            os.replace(tmp, self.args.progress)
        except Exception:
            pass

    def _track_best(self, d2, nonces, job, target):
        """Track the lowest displayed hash seen (lexicographic min of the
        byte-swapped digest words == min of the displayed 256-bit hash)."""
        x = d2.to(torch.int64) & 0xFFFFFFFF
        B = d2.shape[0]
        sentinel = (1 << 63) - 1
        tied = torch.ones(B, dtype=torch.bool, device=d2.device)
        best = torch.zeros((), dtype=torch.int64, device=d2.device)
        for i in range(8):
            wi = _bswap64(x[:, 7 - i])
            masked = torch.where(tied, wi, torch.full_like(wi, sentinel))
            m = masked.min()
            pick = (tied & (wi == m)).nonzero(as_tuple=False)
            if pick.numel():
                best = pick[0, 0]
            tied = tied & (wi == m)
        b = int(best)
        n = int(nonces[b])
        words = [int(d2[b, k].item()) & MASK32 for k in range(8)]
        display = (b"".join(w.to_bytes(4, "big") for w in words)[::-1]).hex()
        cur = int(display, 16)
        if self.best is None or cur < self.best[0]:
            self.best = (cur, display, n, cur <= target)
            self._write_best(job, target)

    def _write_best(self, job, target):
        if self.best is None:
            return
        _, display, n, meets = self.best
        data = {
            "job_id": job["job_id"],
            "extranonce2": self.cur_en2 or "",
            "ntime": job["ntime"],
            "nonce": f"{n:08x}",
            "hash": display,
            "target_hex": f"{target:064x}",
            "meets_target": bool(meets),
        }
        try:
            tmp = self.args.best_hash + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.args.best_hash)
        except Exception:
            pass

    def _new_extranonce2(self, job):
        size = int(job.get("extranonce2_size", 6)) * 2
        return "".join(random.choice("0123456789abcdef") for _ in range(size))

    # ---------------------------------------------------------------- run
    def run(self):
        args = self.args
        job = self.load_job()
        if job is None:
            sys.exit(f"[miner] no jobs found in {args.jobs} - run pool_connector.py first")

        target = int(job.get("share_target_hex") or "0", 16)
        if not target:
            target = target_from_diff(1)
        tw = target_words(target)

        extranonce2, nonce = self._resume(job)
        if extranonce2 is None:
            extranonce2 = self._new_extranonce2(job)
            nonce = 0
        self.cur_en2 = extranonce2
        pre = precompute_job(job, extranonce2, self.device)
        print(f"[miner] job {job['job_id']}  extranonce2={extranonce2}  "
              f"diff={job.get('share_diff')}  target=0x{target:064x}")

        start = time.time()
        window_hashes = 0
        last_report = time.time()
        last_check = time.time()
        jobs_mtime = self._jobs_mtime()
        total = 0
        try:
            while True:
                B = min(self.batch, 2 ** 32 - nonce)
                if B <= 0:                       # nonce space exhausted -> roll extranonce2
                    nonce = 0
                    extranonce2 = self._new_extranonce2(job)
                    self.cur_en2 = extranonce2
                    pre = precompute_job(job, extranonce2, self.device)
                    print(f"[miner] nonce space exhausted, new extranonce2={extranonce2}")
                    B = self.batch

                nonces = torch.arange(nonce, nonce + B, dtype=torch.int64, device=self.device)
                d2 = self.sha.hash_headers(nonces, pre)
                self._track_best(d2, nonces, job, target)
                mask = compare_le(d2, tw)

                if mask.any():
                    found = []
                    for i in mask.nonzero(as_tuple=False).flatten().tolist():
                        n = nonce + i
                        words = [int(d2[i, k].item()) & MASK32 for k in range(8)]
                        digest_be = b"".join(w.to_bytes(4, "big") for w in words)
                        display = digest_be[::-1].hex()
                        found.append({
                            "share_id": hashlib.sha256(
                                f"{job['job_id']}:{extranonce2}:{job['ntime']}:{n:08x}".encode()
                            ).hexdigest()[:16],
                            "job_id": job["job_id"],
                            "extranonce1": job.get("extranonce1", ""),
                            "extranonce2": extranonce2,
                            "ntime": job["ntime"],
                            "nonce": f"{n:08x}",
                            "hash": display,
                            "difficulty": float(job.get("share_diff", 0) or 0),
                            "found_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        })
                    added = self.shares.add(found)
                    for s in found:
                        print(f"[miner] SHARE 0x{s['hash'][:40]}... nonce={s['nonce']} "
                              f"(+{added} new in shares.txt)")
                    if args.stop_on_share:
                        self._save_progress(job, extranonce2, nonce + B)
                        self._write_best(job, target)
                        print("[miner] --stop-on-share: exiting")
                        return 0

                nonce += B
                total += B
                window_hashes += B
                now = time.time()

                if now - last_report >= 5:
                    rate = window_hashes / (now - last_report) if window_hashes else 0
                    print(f"[miner] {total / 1e6:9.1f} M hashes total | "
                          f"{rate / 1e6:8.2f} MH/s | nonce=0x{nonce:08x}")
                    last_report = now
                    window_hashes = 0
                    self._save_progress(job, extranonce2, nonce)
                    self._write_best(job, target)

                if args.max_nonces and total >= args.max_nonces:
                    self._save_progress(job, extranonce2, nonce)
                    self._write_best(job, target)
                    print(f"[miner] reached --max-nonces {total}")
                    return 0

                if now - last_check >= args.check_interval:
                    last_check = now
                    mtime = self._jobs_mtime()
                    if mtime != jobs_mtime:
                        jobs_mtime = mtime
                        new_job = self.load_job()
                        if new_job and new_job["job_id"] != job["job_id"]:
                            self._save_progress(job, extranonce2, nonce)
                            self._write_best(job, target)
                            job = new_job
                            extranonce2 = self._new_extranonce2(job)
                            self.cur_en2 = extranonce2
                            nonce = 0
                            pre = precompute_job(job, extranonce2, self.device)
                            target = int(job.get("share_target_hex") or "0", 16)
                            if not target:
                                target = target_from_diff(1)
                            tw = target_words(target)
                            print(f"[miner] switched to new job {job['job_id']}")
        except KeyboardInterrupt:
            print("\n[miner] interrupted - progress saved")
            self._save_progress(job, extranonce2, nonce)
            self._write_best(job, target)
            return 0

    def _jobs_mtime(self):
        try:
            return os.path.getmtime(self.args.jobs)
        except OSError:
            return 0


def main():
    ap = argparse.ArgumentParser(description="PyTorch SHA-256d Stratum-job miner")
    ap.add_argument("--jobs", default="jobs.txt", help="jobs file (from pool_connector)")
    ap.add_argument("--shares", default="shares.txt", help="shares output file")
    ap.add_argument("--progress", default="progress.json", help="resume state file")
    ap.add_argument("--batch", type=int, default=2 ** 20, help="nonces per kernel launch")
    ap.add_argument("--diff", type=float, default=0, help="override pool share difficulty")
    ap.add_argument("--max-nonces", type=int, default=0, help="stop after N nonces (0=never)")
    ap.add_argument("--stop-on-share", action="store_true", help="exit after first share")
    ap.add_argument("--check-interval", type=float, default=20, help="jobs.txt re-check period")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    ap.add_argument("--best-hash", default="best_hash.json",
                    help="file to write the best hash found (for diagnostics)")
    args = ap.parse_args()
    sys.exit(Miner(args).run() or 0)


if __name__ == "__main__":
    main()
