#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pool_connector.py
=================
Stratum v1 pool connector with GitHub file sync (pure Python stdlib, no deps).

What it does
------------
1. Connects to a Bitcoin Stratum pool (e.g. stratum+tcp://btc.kryptex.network:7014).
2. Receives mining jobs (mining.notify) and pool difficulty (mining.set_difficulty),
   and saves them as a JSON list into `jobs.txt` in a GitHub repository
   (Contents API) -- that file is what your offline GPU miner reads.
3. Watches the GitHub `shares.txt` file.  The moment it is updated/rewritten
   (by you manually pushing the miner's shares), the connector:
       * downloads the shares,
       * submits each one to the pool (mining.submit),
       * rewrites shares.txt on GitHub as an empty array ("clears it"),
       * and waits for the next update.

Why GitHub in the middle?  So the miner can run on a computer with no
internet/pool access -- you sync jobs.txt and shares.txt by hand (or via a
sync tool) between the miner machine and this connector machine.

Usage
-----
    export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx        # your PAT
    python3 pool_connector.py --config config.json      # see config.example.json

    # or fully on the command line:
    python3 pool_connector.py --pool stratum+tcp://btc.kryptex.network:7014 \
        --worker krxYRPV4WQ.0x --owner dfghjklfgbhgh --repo test
"""

import argparse
import base64
import json
import os
import queue
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
DIFF1_TARGET = 0x00000000FFFF0000000000000000000000000000000000000000000000000000


def target_from_diff(diff: float) -> int:
    return int(DIFF1_TARGET // max(float(diff), 1e-12))


def log(*args):
    print(time.strftime("[%H:%M:%S]"), *args, flush=True)


class GitHubError(Exception):
    pass


class GitHubConflict(GitHubError):
    pass


# ---------------------------------------------------------------------------
# GitHub Contents-API client (urllib only)
# ---------------------------------------------------------------------------
class GitHubClient(object):
    def __init__(self, token, owner, repo, branch="main"):
        if not token:
            raise GitHubError("no GitHub token (set GITHUB_TOKEN env var)")
        self.base = f"https://api.github.com/repos/{owner}/{repo}/contents/"
        self.branch = branch
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "pool-connector",
        }
        self.etag_cache = {}

    def _req(self, method, path, body=None, headers=None):
        url = self.base + urllib.parse.quote(path)
        data = json.dumps(body).encode() if body is not None else None
        hdrs = dict(self.headers)
        hdrs.update(headers or {})
        req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
                return r.status, (json.loads(raw) if raw else None), dict(r.headers)
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                j = json.loads(raw) if raw else None
            except Exception:
                j = None
            return e.code, j, dict(e.headers)

    def get_file(self, path):
        """Returns {"content","sha"} or None (404) or "unchanged" (304)."""
        hdrs = {}
        etag = self.etag_cache.get(path)
        if etag:
            hdrs["If-None-Match"] = etag
        code, j, h = self._req("GET", path, headers=hdrs)
        if code == 404:
            return None
        if code == 304:
            return "unchanged"
        if code != 200:
            raise GitHubError(f"GET {path}: HTTP {code} {j}")
        self.etag_cache[path] = h.get("ETag", "")
        content = base64.b64decode(j["content"]).decode("utf-8", errors="replace")
        return {"content": content, "sha": j["sha"]}

    def put_file(self, path, content, sha=None, message="update file"):
        body = {
            "message": message,
            "content": base64.b64encode(content.encode()).decode(),
            "branch": self.branch,
        }
        if sha:
            body["sha"] = sha
        code, j, _ = self._req("PUT", path, body=body)
        if code == 409:
            raise GitHubConflict(f"PUT {path}: conflict (file changed meanwhile)")
        if code not in (200, 201):
            raise GitHubError(f"PUT {path}: HTTP {code} {j}")
        return j["content"]["sha"]

    def delete_file(self, path, sha, message="delete file"):
        body = {"message": message, "sha": sha, "branch": self.branch}
        code, j, _ = self._req("DELETE", path, body=body)
        if code not in (200, 204):
            raise GitHubError(f"DELETE {path}: HTTP {code} {j}")
        return True


# ---------------------------------------------------------------------------
# Stratum v1 client (one background reader thread, thread-safe send)
# ---------------------------------------------------------------------------
class StratumDisconnected(Exception):
    pass


class StratumClient(object):
    def __init__(self, host, port, worker, password,
                 on_job, on_diff, on_ready, on_status):
        self.host, self.port, self.worker, self.password = host, port, worker, password
        self.on_job, self.on_diff = on_job, on_diff
        self.on_ready, self.on_status = on_ready, on_status
        self.stop_evt = threading.Event()
        self.sock = None
        self.extranonce1 = None
        self.extranonce2_size = None
        self.connected = threading.Event()
        self._send_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending = {}          # id -> {"evt": Event, "result": None, "error": None}
        self._next_id = 0
        self._thread = None

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.stop_evt.set()
        try:
            if self.sock:
                self.sock.shutdown(socket.SHUT_RDWR)
                self.sock.close()
        except Exception:
            pass

    # -- low level ---------------------------------------------------------
    def send(self, obj):
        if not self.sock:
            raise StratumDisconnected("no socket")
        with self._send_lock:
            self.sock.sendall((json.dumps(obj) + "\n").encode())

    def _request(self, msg, timeout=15):
        """Send a request and wait for the matching response."""
        if not self.connected.is_set():
            raise StratumDisconnected("pool not connected")
        with self._pending_lock:
            self._next_id += 1
            rid = self._next_id
            entry = {"evt": threading.Event(), "result": None, "error": None}
            self._pending[rid] = entry
        msg = dict(msg)
        msg["id"] = rid
        self.send(msg)
        if not entry["evt"].wait(timeout):
            with self._pending_lock:
                self._pending.pop(rid, None)
            raise StratumDisconnected(f"timeout waiting for response to {msg['method']}")
        if entry["error"] is not None:
            raise StratumDisconnected(f"pool error: {entry['error']}")
        return entry["result"]

    def submit(self, job_id, extranonce2, ntime, nonce):
        """Submit a share; returns pool result (True/False)."""
        res = self._request({
            "method": "mining.submit",
            "params": [self.worker, job_id, extranonce2, ntime, nonce],
        })
        return res is True

    # -- main loop ---------------------------------------------------------
    def _run(self):
        while not self.stop_evt.is_set():
            try:
                self._connect_once()
            except Exception as e:
                self.on_status(f"connection error: {e}")
            self.connected.clear()
            if not self.stop_evt.is_set():
                time.sleep(5)  # reconnect backoff

    def _connect_once(self):
        self.on_status(f"connecting to {self.host}:{self.port} ...")
        s = socket.create_connection((self.host, self.port), timeout=15)
        s.settimeout(5)
        self.sock = s
        self.connected.clear()

        # -- subscribe
        res = self._request({"method": "mining.subscribe",
                             "params": ["python-pool-connector/1.0"]})
        self.extranonce1 = res[1]
        self.extranonce2_size = res[2]
        self.on_status(f"subscribed extranonce1={self.extranonce1} "
                       f"extranonce2_size={self.extranonce2_size}")

        # -- authorize
        res = self._request({"method": "mining.authorize",
                             "params": [self.worker, self.password]})
        if res is not True:
            self.on_status(f"authorize FAILED for worker {self.worker!r} - check the wallet string")
            self.sock.close()
            return
        self.on_status(f"authorized worker {self.worker!r}")
        self.connected.set()
        self.on_ready(self.extranonce1, self.extranonce2_size)

        # -- reader loop
        buf = b""
        quiet_loops = 0
        while not self.stop_evt.is_set():
            try:
                chunk = s.recv(65536)
            except socket.timeout:
                quiet_loops += 1
                if quiet_loops >= 24:          # ~2 minutes of silence
                    try:
                        self.send({"method": "mining.ping", "params": []})
                        self.on_status("sent keepalive ping")
                    except Exception:
                        break
                    quiet_loops = 0
                continue
            if not chunk:
                break
            quiet_loops = 0
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
                try:
                    self._handle(msg)
                except Exception as e:
                    self.on_status(f"handler error: {e}")
        self.on_status("connection closed")
        try:
            s.close()
        except Exception:
            pass

    def _handle(self, msg):
        if isinstance(msg, dict) and "id" in msg and msg["id"] is not None and "method" not in msg:
            # response to one of our requests
            with self._pending_lock:
                entry = self._pending.pop(msg["id"], None)
            if entry:
                entry["result"] = msg.get("result")
                entry["error"] = msg.get("error")
                entry["evt"].set()
            return
        method = msg.get("method")
        params = msg.get("params", [])
        if method == "mining.notify":
            self.on_job(params)
        elif method == "mining.set_difficulty":
            self.on_diff(float(params[0]))
        elif method == "mining.set_extranonce":
            self.on_status(f"set_extranonce: {params}")
            self.on_ready(params[0], int(params[1]))
        elif method == "mining.ping":
            self.send({"id": msg.get("id"), "result": []})
        else:
            self.on_status(f"unknown message: {msg}")


# ---------------------------------------------------------------------------
# Connector application
# ---------------------------------------------------------------------------
class ConnectorApp(object):
    def __init__(self, cfg):
        gh = cfg.get("github", {})
        token = (os.environ.get(gh.get("token_env", "GITHUB_TOKEN"))
                 or os.environ.get("GITHUB_TOKEN")
                 or gh.get("token") or "")
        self.github = GitHubClient(token,
                                   gh.get("owner") or os.environ.get("GITHUB_OWNER", ""),
                                   gh.get("repo") or os.environ.get("GITHUB_REPO", ""),
                                   gh.get("branch") or os.environ.get("GITHUB_BRANCH", "main"))
        self.jobs_path = gh.get("jobs_path", "jobs.txt")
        self.shares_path = gh.get("shares_path", "shares.txt")
        self.keep_jobs = int(cfg.get("keep_jobs", 5))
        self.poll_interval = float(cfg.get("poll_interval_sec", 10))
        self.min_push = float(cfg.get("min_job_push_interval_sec", 15))

        self.pool_url = cfg["pool_url"]
        self.worker = cfg.get("worker", "")
        self.password = cfg.get("password", "x")

        self.jobs = []                 # newest last
        self.jobs_lock = threading.Lock()
        self.diff = 1.0
        self.extranonce1 = ""
        self.extranonce2_size = 6
        self.last_push = 0.0
        self.last_share_sha = None
        self.submitted = set()
        self.submitted_lock = threading.Lock()
        self.push_queue = queue.Queue()
        self.stop_evt = threading.Event()

    # -- stratum callbacks (background reader thread) ----------------------
    def handle_job(self, params):
        (job_id, prevhash, coinb1, coinb2, merkle_branch,
         version, nbits, ntime, clean_jobs) = params[:9]
        record = {
            "received_at": time.time(),
            "job_id": job_id,
            "prevhash": prevhash,
            "coinb1": coinb1,
            "coinb2": coinb2,
            "merkle_branch": merkle_branch,
            "version": version,
            "nbits": nbits,
            "ntime": ntime,
            "clean_jobs": bool(clean_jobs),
            "extranonce1": self.extranonce1,
            "extranonce2_size": self.extranonce2_size,
            "share_diff": self.diff,
            "share_target_hex": f"{target_from_diff(self.diff):064x}",
            "pool": self.pool_url,
            "worker": self.worker,
        }
        with self.jobs_lock:
            self.jobs = [j for j in self.jobs if j["job_id"] != job_id] + [record]
            self.jobs = self.jobs[-self.keep_jobs:]
        log(f"job {job_id}  diff={self.diff}  ntime={ntime}  clean={clean_jobs}")
        self.push_queue.put(("jobs", None))

    def handle_diff(self, diff):
        self.diff = float(diff)
        with self.jobs_lock:
            for j in self.jobs:
                j["share_diff"] = self.diff
                j["share_target_hex"] = f"{target_from_diff(self.diff):064x}"
        log(f"difficulty -> {self.diff}")
        self.push_queue.put(("jobs", None))

    def handle_ready(self, extranonce1, size):
        self.extranonce1 = extranonce1
        self.extranonce2_size = int(size)
        log(f"extranonce1={extranonce1} extranonce2_size={size}")

    # -- GitHub push worker (own thread so the pool reader never blocks) ---
    def push_worker(self):
        while not self.stop_evt.is_set():
            try:
                kind, _ = self.push_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                if kind == "jobs":
                    self.push_jobs()
            except Exception as e:
                log(f"push error: {e}")

    def push_jobs(self):
        now = time.time()
        if now - self.last_push < self.min_push:
            return
        with self.jobs_lock:
            content = json.dumps(self.jobs, indent=2)
            n_jobs = len(self.jobs)
        res = self.github.get_file(self.jobs_path)
        if res == "unchanged":
            self.last_push = now
            return
        sha = res["sha"] if res else None
        if res and res["content"].strip() == content.strip():
            self.last_push = now
            return
        self.github.put_file(self.jobs_path, content, sha=sha,
                             message=f"update jobs ({n_jobs} job(s))")
        self.last_push = now
        log(f"pushed {n_jobs} job(s) -> {self.jobs_path}")

    # -- shares watcher (main thread) --------------------------------------
    def watch_shares(self):
        log(f"watching {self.shares_path} every {self.poll_interval}s ...")
        while not self.stop_evt.is_set():
            time.sleep(self.poll_interval)
            try:
                self._process_shares()
            except GitHubConflict:
                self.last_share_sha = None   # re-read next round
            except Exception as e:
                log(f"shares watcher error: {e}")

    def _process_shares(self):
        res = self.github.get_file(self.shares_path)
        if res is None or res == "unchanged":
            return
        if self.last_share_sha == res["sha"]:
            return
        self.last_share_sha = res["sha"]
        content = res["content"]
        try:
            shares = json.loads(content) if content.strip() else []
        except ValueError:
            log(f"shares.txt is not valid JSON ({len(content)} bytes) - ignoring")
            return
        if not isinstance(shares, list):
            return
        if not shares:
            return

        with self.submitted_lock:
            pending = [s for s in shares if s.get("share_id") not in self.submitted]
        if not pending:
            self._clear_shares(res)
            return
        log(f"{len(pending)} new share(s) found in {self.shares_path} - submitting")

        failed = False
        for s in pending:
            try:
                ok = self.stratum.submit(s["job_id"], s["extranonce2"], s["ntime"], s["nonce"])
            except StratumDisconnected as e:
                log(f"  submit failed (pool down): {e}")
                failed = True
                break
            except Exception as e:
                log(f"  submit error: {e}")
                failed = True
                break
            with self.submitted_lock:
                self.submitted.add(s["share_id"])
            if ok:
                log(f"  SHARE ACCEPTED  nonce={s.get('nonce')} job={s.get('job_id')}")
            else:
                log(f"  share rejected by pool  nonce={s.get('nonce')} job={s.get('job_id')}")
        if failed:
            log("  pool not reachable - will retry next poll")
            return
        self._clear_shares(res)
        log("shares submitted - shares.txt cleared")

    def _clear_shares(self, res):
        new_sha = self.github.put_file(self.shares_path, "[]",
                                       sha=res["sha"], message="shares submitted - cleared")
        self.last_share_sha = new_sha

    # -- main --------------------------------------------------------------
    def run(self):
        parsed = urllib.parse.urlparse(self.pool_url)
        host = parsed.hostname
        port = parsed.port or 3333

        self.stratum = StratumClient(
            host, port, self.worker, self.password,
            on_job=self.handle_job, on_diff=self.handle_diff,
            on_ready=self.handle_ready, on_status=log)

        push_thread = threading.Thread(target=self.push_worker, daemon=True)
        push_thread.start()
        self.stratum.start()

        log(f"pool: {self.pool_url}  worker: {self.worker!r}")
        log(f"github: {self.github.base}  jobs={self.jobs_path} shares={self.shares_path}")
        try:
            self.watch_shares()
        except KeyboardInterrupt:
            log("stopping ...")
        finally:
            self.stop_evt.set()
            self.stratum.stop()


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def load_config(args):
    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = json.load(f)
    elif os.path.exists("config.json"):
        with open("config.json") as f:
            cfg = json.load(f)

    cli = {
        "pool_url": args.pool, "worker": args.worker, "password": args.password,
        "poll_interval_sec": args.poll, "keep_jobs": args.keep_jobs,
    }
    for k, v in cli.items():
        if v:
            cfg[k] = v

    gh = cfg.setdefault("github", {})
    if args.owner:
        gh["owner"] = args.owner
    if args.repo:
        gh["repo"] = args.repo
    if args.branch:
        gh["branch"] = args.branch
    if args.jobs_path:
        gh["jobs_path"] = args.jobs_path
    if args.shares_path:
        gh["shares_path"] = args.shares_path
    if not gh.get("owner") or not gh.get("repo"):
        raise SystemExit("github owner/repo required (config.json or --owner/--repo)")
    if not cfg.get("pool_url"):
        raise SystemExit("pool_url required (config.json or --pool)")
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Stratum pool connector with GitHub sync")
    ap.add_argument("--config", default="", help="path to config.json")
    ap.add_argument("--pool", default="", help="stratum+tcp://host:port")
    ap.add_argument("--worker", default="", help="pool worker / wallet string")
    ap.add_argument("--password", default="", help="pool password (usually x)")
    ap.add_argument("--owner", default="", help="github owner")
    ap.add_argument("--repo", default="", help="github repo")
    ap.add_argument("--branch", default="", help="github branch (default main)")
    ap.add_argument("--jobs-path", default="", help="github path for jobs (default jobs.txt)")
    ap.add_argument("--shares-path", default="", help="github path for shares (default shares.txt)")
    ap.add_argument("--poll", type=float, default=0, help="shares poll interval (s)")
    ap.add_argument("--keep-jobs", type=int, default=0, help="max jobs kept in jobs.txt")
    args = ap.parse_args()

    cfg = load_config(args)
    ConnectorApp(cfg).run()


if __name__ == "__main__":
    main()
