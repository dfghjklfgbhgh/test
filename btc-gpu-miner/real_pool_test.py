#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
real_pool_test.py
=================
Test share acceptance against the REAL pool (no mock).

Flow:
  1. connect to the real Stratum pool, subscribe + authorize,
     try mining.suggest_difficulty (most pools ignore it),
  2. wait for a real mining.notify job,
  3. write the job to a local jobs.txt and run bitcoin_miner.py on it
     (a bounded nonce budget, so it finishes in seconds/minutes),
  4. submit the shares the miner found -- or, if none were found (the
     expected case at a real pool's difficulty), submit the BEST hash the
     miner saw -- and print the pool's verbatim response.

The pool's response is the whole point: if it says "low difficulty share"
the pipeline is correct and only the hashrate is too low; if it accepts,
even better.

Usage:
    python3 real_pool_test.py --worker krxYRPV4WQ.0x --max-nonces 3000000
"""

import argparse
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time

DIFF1_TARGET = 0x00000000FFFF0000000000000000000000000000000000000000000000000000


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


class RealPool(object):
    """Minimal Stratum v1 client for the test (inline handshake + reader
    thread; submit responses resolved through a pending queue)."""

    def __init__(self, host, port, worker, password):
        self.host, self.port = host, port
        self.worker, self.password = worker, password
        self.sock = None
        self._buf = b""
        self.extranonce1 = None
        self.en2size = None
        self.diff = None
        self.jobs = {}
        self.newest_job = None
        self.diffs_seen = []
        self.responses = []
        self._next = 0
        self._lock = threading.Lock()
        self._pending = {}
        self._stop = threading.Event()

    # ------------------------------------------------------------ helpers
    def send(self, obj):
        with self._lock:
            self.sock.sendall((json.dumps(obj) + "\n").encode())

    def _readline(self, timeout=15):
        self.sock.settimeout(timeout)
        while b"\n" not in self._buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("closed")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line.strip()

    def _inline(self, msg, timeout=20):
        with self._lock:
            self._next += 1
            rid = self._next
        self.send({**msg, "id": rid})
        while True:
            line = self._readline(timeout)
            if not line:
                continue
            try:
                resp = json.loads(line)
            except ValueError:
                continue
            if isinstance(resp, dict) and resp.get("id") == rid and "method" not in resp:
                if resp.get("error") is not None:
                    raise RuntimeError(f"pool error: {resp['error']}")
                return resp.get("result")
            self._dispatch(resp)

    def submit(self, job_id, en2, ntime, nonce, timeout=40):
        with self._lock:
            self._next += 1
            rid = self._next
            q = queue.Queue()
            self._pending[rid] = q
        self.send({"id": rid, "method": "mining.submit",
                   "params": [self.worker, job_id, en2, ntime, nonce]})
        try:
            return q.get(timeout=timeout)
        except queue.Empty:
            return {"timeout": True}

    # ------------------------------------------------------------ connect
    def connect(self, suggest=1e-5):
        log(f"connecting to {self.host}:{self.port} ...")
        self.sock = socket.create_connection((self.host, self.port), timeout=15)
        r = self._inline({"method": "mining.subscribe",
                          "params": ["real-pool-test/1.0"]})
        self.extranonce1 = r[1]
        self.en2size = r[2]
        log(f"subscribed  extranonce1={self.extranonce1}  extranonce2_size={self.en2size}")

        r = self._inline({"method": "mining.authorize",
                          "params": [self.worker, self.password]})
        log(f"authorize -> {r!r}")
        if r is not True:
            raise RuntimeError("authorize failed - check worker/password")

        if suggest:
            self.send({"id": 99, "method": "mining.suggest_difficulty",
                       "params": [suggest]})
            log(f"sent mining.suggest_difficulty [{suggest}] (pools usually ignore it)")
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        quiet = 0
        while not self._stop.is_set():
            try:
                line = self._readline(10)
            except socket.timeout:
                quiet += 1
                if quiet >= 3:          # ~30 s of silence -> keepalive ping
                    try:
                        self.send({"method": "mining.ping", "params": []})
                        log("sent keepalive ping")
                    except Exception:
                        break
                    quiet = 0
                continue
            except Exception:
                break                  # connection closed
            if not line:
                continue
            quiet = 0
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            try:
                self._dispatch(msg)
            except Exception as e:
                log(f"dispatch error: {e}")

    def _dispatch(self, msg):
        # response to a submit
        if isinstance(msg, dict) and msg.get("id") is not None and "method" not in msg:
            q = self._pending.pop(msg["id"], None)
            if q is not None:
                q.put(msg)
            return
        method, params = msg.get("method"), msg.get("params", [])
        if method == "mining.set_difficulty":
            d = float(params[0])
            self.diff = d
            self.diffs_seen.append(d)
            log(f"set_difficulty -> {d}")
        elif method == "mining.notify":
            (job_id, prevhash, coinb1, coinb2, branch,
             version, nbits, ntime, clean) = params[:9]
            rec = {
                "received_at": time.time(),
                "job_id": job_id,
                "prevhash": prevhash,
                "coinb1": coinb1,
                "coinb2": coinb2,
                "merkle_branch": branch,
                "version": version,
                "nbits": nbits,
                "ntime": ntime,
                "clean_jobs": bool(clean),
                "extranonce1": self.extranonce1,
                "extranonce2_size": self.en2size,
                "share_diff": self.diff,
                "share_target_hex": f"{int(DIFF1_TARGET // (self.diff or 1)):064x}",
                "pool": f"{self.host}:{self.port}",
                "worker": self.worker,
            }
            self.jobs[job_id] = rec
            self.newest_job = rec
            log(f"job {job_id}  diff={self.diff}  ntime={ntime}  clean={clean}")
        elif method == "mining.ping":
            self.send({"id": msg.get("id"), "result": []})
        else:
            log(f"(other) {msg}")

    def stop(self):
        self._stop.set()
        try:
            self.sock.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="real-pool share acceptance test")
    ap.add_argument("--host", default="btc.kryptex.network")
    ap.add_argument("--port", type=int, default=7014)
    ap.add_argument("--worker", default="krxYRPV4WQ.0x")
    ap.add_argument("--password", default="x")
    ap.add_argument("--suggest", type=float, default=1e-5,
                    help="mining.suggest_difficulty value (0 = don't send)")
    ap.add_argument("--max-nonces", type=int, default=4_000_000,
                    help="mining budget handed to bitcoin_miner.py")
    ap.add_argument("--workdir", default=".realpool_test")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    jobs_path = os.path.join(args.workdir, "jobs.txt")
    shares_path = os.path.join(args.workdir, "shares.txt")
    prog_path = os.path.join(args.workdir, "progress.json")
    best_path = os.path.join(args.workdir, "best_hash.json")
    for p in (shares_path, prog_path, best_path):
        if os.path.exists(p):
            os.remove(p)

    pool = RealPool(args.host, args.port, args.worker, args.password)
    pool.connect(args.suggest)
    time.sleep(2)
    log(f"difficulty after suggest: {pool.diff}  (diffs seen: {pool.diffs_seen})")

    deadline = time.time() + 60
    while pool.newest_job is None and time.time() < deadline:
        time.sleep(0.5)
    job = pool.newest_job
    if job is None:
        log("no job received - pool may need more time; exiting")
        pool.stop()
        sys.exit(1)

    with open(jobs_path, "w") as f:
        json.dump([job], f, indent=2)
    target = int(job["share_target_hex"], 16)
    log(f"using real job {job['job_id']}  pool diff={job['share_diff']}  "
        f"target=0x{target:064x}")

    # ---- mine with the real miner --------------------------------------
    budget = args.max_nonces
    cmd = [sys.executable, "bitcoin_miner.py",
           "--jobs", jobs_path, "--shares", shares_path,
           "--progress", prog_path, "--best-hash", best_path,
           "--max-nonces", str(budget), "--cpu"]
    log(f"running miner with {budget:,} nonce budget (~{budget / 1.6e5:.0f}s at CPU speed) ...")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    dt = time.time() - t0
    log(f"miner finished in {dt:.1f}s (rc={proc.returncode})")
    for line in proc.stdout.strip().splitlines()[-8:]:
        log("  miner:", line)

    # ---- collect what to submit -----------------------------------------
    to_submit = []
    if os.path.exists(shares_path):
        with open(shares_path) as f:
            shares = json.load(f)
        if shares:
            log(f"miner found {len(shares)} share(s) - submitting those")
            to_submit = shares
    if not to_submit and os.path.exists(best_path):
        with open(best_path) as f:
            best = json.load(f)
        log(f"no qualifying share at pool difficulty; submitting BEST hash: "
            f"{best['hash'][:24]}... nonce={best['nonce']} "
            f"meets_target={best['meets_target']}")
        to_submit = [best]

    if not to_submit:
        log("nothing to submit (no shares, no best-hash file)")
        pool.stop()
        return

    # ---- submit to the REAL pool ----------------------------------------
    log(f"submitting {len(to_submit)} share(s) to {args.host}:{args.port} ...")
    for s in to_submit:
        job_id = s.get("job_id")
        job_rec = pool.jobs.get(job_id)
        age = (time.time() - job_rec["received_at"]) if job_rec else None
        log(f"  -> submit job={job_id} en2={s['extranonce2']} "
            f"ntime={s['ntime']} nonce={s['nonce']} (job age {age:.0f}s)"
            if age is not None else f"  -> submit job={job_id}")
        resp = pool.submit(job_id, s["extranonce2"], s["ntime"], s["nonce"])
        pool.responses.append({"share": s, "response": resp})
        log(f"  POOL RESPONSE: {json.dumps(resp)}")

    # ---- verdict ---------------------------------------------------------
    print("\n==================== VERDICT ====================")
    for r in pool.responses:
        resp = r["response"]
        s = r["share"]
        if resp.get("result") is True:
            print(f"  SHARE nonce={s['nonce']}: ACCEPTED by the real pool ✓")
            continue
        if resp.get("timeout"):
            print(f"  SHARE nonce={s['nonce']}: no response (timeout) - connection issue")
            continue
        err = resp.get("error")
        errcode = err[0] if isinstance(err, list) and err else None
        errmsg = (err[1] if isinstance(err, list) and len(err) > 1 else str(err))
        msg = str(errmsg).lower()
        if errcode == 21:
            reason = "job not found / stale (job rotated before submit)"
        elif errcode == 22:
            reason = "duplicate share"
        elif errcode == 23 or "low difficulty" in msg:
            reason = "BELOW POOL DIFFICULTY (hash does not meet share target)"
        elif errcode == -1 and "invalid share" in msg:
            reason = ("BELOW POOL DIFFICULTY (pool's generic response for a "
                      "well-formed share whose hash is below the share target)")
        elif errcode == 24:
            reason = "unauthorized worker"
        else:
            reason = f"other (see response)"
        print(f"  SHARE nonce={s['nonce']}: REJECTED by the real pool "
              f"(error {errcode}: {errmsg} -> {reason})")
    print("=================================================")
    print("difficulties the pool set this session:", pool.diffs_seen)
    if pool.diffs_seen and max(pool.diffs_seen) >= 1000:
        print("note: pool difficulty is >= 1000, i.e. the share target requires the first")
        print("      ~51 bits of the block hash to be zero -> 2^51 hashes per share.")
    pool.stop()


if __name__ == "__main__":
    main()
