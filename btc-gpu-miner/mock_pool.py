#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mock_pool.py
============
A local Stratum v1 test pool used to exercise the full pipeline without a
real pool.  Critically, it VALIDATES every submitted share independently
(from the coinbase through the merkle root to the block-header double
SHA-256), so if the miner hashes correctly the pool says ACCEPTED.

    python3 mock_pool.py [--port 17014] [--diff 0.00001]
"""

import argparse
import hashlib
import json
import random
import socket
import threading
import time
from collections import Counter

DIFF1_TARGET = 0x00000000FFFF0000000000000000000000000000000000000000000000000000


def sha256d(b: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()


class MockJob(object):
    """A fake-but-structurally-valid Stratum job (coinbase with a 12-byte
    extranonce slot, merkle branch, version/bits/time)."""

    def __init__(self, job_id, height=None):
        height = height or random.randint(890000, 920000)
        self.job_id = job_id
        self.version = "20000000"
        self.prevhash = bytes(random.getrandbits(8) for _ in range(32)).hex()
        self.nbits = "1d00ffff"
        self.ntime = f"{int(time.time()):08x}"
        self.branch = [bytes(random.getrandbits(8) for _ in range(32)).hex()
                       for _ in range(3)]

        script_head = (bytes([0x03, height & 0xFF, (height >> 8) & 0xFF,
                              (height >> 16) & 0xFF])
                       + b"mock pool coinbase")
        reserved = 12  # extranonce1(6) + extranonce2(6)
        script_len = len(script_head) + reserved
        self.coinb1 = (b"\x01\x00\x00\x00"          # tx version
                       + b"\x01"                    # 1 input
                       + b"\x00" * 32               # prevout hash
                       + b"\xff\xff\xff\xff"        # prevout index
                       + bytes([script_len])
                       + script_head).hex()
        self.coinb2 = (b"\xff\xff\xff\xff"          # sequence
                       + b"\x00"                    # 0 outputs
                       + b"\x00\x00\x00\x00").hex() # locktime

    def block_hash(self, extranonce1, extranonce2, ntime, nonce):
        """Full reference share hash (internal-order digest)."""
        coinbase = (bytes.fromhex(self.coinb1) + bytes.fromhex(extranonce1)
                    + bytes.fromhex(extranonce2) + bytes.fromhex(self.coinb2))
        leaf = sha256d(coinbase)[::-1]    # internal byte order
        mroot = leaf
        for bh in self.branch:            # branch hashes are displayed order
            mroot = sha256d(mroot + bytes.fromhex(bh)[::-1])
        mroot = mroot[::-1]               # back to displayed order
        header = (int(self.version, 16).to_bytes(4, "little")
                  + bytes.fromhex(self.prevhash)
                  + mroot[::-1]
                  + int(ntime, 16).to_bytes(4, "little")
                  + int(self.nbits, 16).to_bytes(4, "little")
                  + int(nonce, 16).to_bytes(4, "little"))
        return sha256d(header)


class MockPool(threading.Thread):
    def __init__(self, host="127.0.0.1", port=17014, share_diff=0.00001,
                 job_interval=0):
        super().__init__(daemon=True)
        self.host, self.port = host, port
        self.share_diff = float(share_diff)
        self.job_interval = job_interval
        self.jobs = {}
        self.target = int(DIFF1_TARGET // self.share_diff)
        self.accepted = 0
        self.rejected = 0
        self.reasons = Counter()
        self.submitted = []
        self.stop_evt = threading.Event()

    def run(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(8)
        srv.settimeout(1)
        print(f"[mockpool] listening on {self.host}:{self.port} "
              f"diff={self.share_diff} target=0x{self.target:064x} ...")
        while not self.stop_evt.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            threading.Thread(target=self._handle_conn, args=(conn,),
                             daemon=True).start()
        try:
            srv.close()
        except Exception:
            pass

    def stop_pool(self):
        self.stop_evt.set()

    # ----------------------------------------------------------------
    def _new_job(self):
        job = MockJob(f"mock{len(self.jobs) + 1:04x}")
        self.jobs[job.job_id] = job
        return job

    @staticmethod
    def _notify_params(job):
        return [job.job_id, job.prevhash, job.coinb1, job.coinb2, job.branch,
                job.version, job.nbits, job.ntime, True]

    def _handle_conn(self, conn):
        conn.settimeout(1)
        extranonce1 = bytes(random.getrandbits(8) for _ in range(6)).hex()
        buf = b""

        def send(obj):
            conn.sendall((json.dumps(obj) + "\n").encode())

        try:
            while not self.stop_evt.is_set():
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    chunk = b""
                if not chunk:
                    continue
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except ValueError:
                        continue
                    method, params, rid = msg.get("method"), msg.get("params", []), msg.get("id")
                    if method == "mining.subscribe":
                        send({"id": rid, "error": None, "result": [
                            [["mining.notify", "mock"],
                             ["mining.set_difficulty", "mock"]],
                            extranonce1, 6]})
                        send({"id": None, "method": "mining.set_difficulty",
                              "params": [self.share_diff]})
                        send({"id": None, "method": "mining.notify",
                              "params": self._notify_params(self._new_job())})
                    elif method == "mining.authorize":
                        send({"id": rid, "error": None, "result": True})
                    elif method == "mining.submit":
                        _, job_id, en2, ntime, nonce = params[:5]
                        job = self.jobs.get(job_id)
                        if job is None:
                            self.rejected += 1
                            self.reasons["unknown job"] += 1
                            send({"id": rid, "error": None, "result": False})
                            continue
                        h = job.block_hash(extranonce1, en2, ntime, nonce)
                        ok = int.from_bytes(h, "little") <= self.target
                        if ok:
                            self.accepted += 1
                            self.submitted.append({
                                "job_id": job_id, "extranonce2": en2,
                                "ntime": ntime, "nonce": nonce,
                                "hash": h[::-1].hex(), "diff": self.share_diff})
                            print(f"[mockpool] ACCEPTED share nonce={nonce} "
                                  f"hash=0x{h[::-1].hex()[:32]}...")
                        else:
                            self.rejected += 1
                            self.reasons["below target"] += 1
                            print(f"[mockpool] rejected share nonce={nonce}")
                        send({"id": rid, "error": None, "result": ok})
                    elif method == "mining.ping":
                        send({"id": rid, "result": []})
                    else:
                        print(f"[mockpool] unknown message: {msg}")
        except Exception as e:
            print(f"[mockpool] conn error: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description="local Stratum test pool")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=17014)
    ap.add_argument("--diff", type=float, default=0.00001)
    ap.add_argument("--job-interval", type=float, default=0,
                    help="seconds between new jobs (0 = only on connect)")
    args = ap.parse_args()
    pool = MockPool(args.host, args.port, args.diff, args.job_interval)
    try:
        pool.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pool.stop_pool()


if __name__ == "__main__":
    main()
