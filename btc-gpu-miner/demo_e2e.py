#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_e2e.py
===========
Full end-to-end test of the GitHub-synced mining pipeline:

    mock pool  ->  pool_connector.py  ->  GitHub jobs_demo.txt
         ^                                    |
         |                                    v
         |                     (simulates your manual pull)
         |                                    |
         |                            bitcoin_miner.py
         |                                    |
         |                             shares_demo.txt (GitHub, manual push)
         |                                    |
         +---------  pool_connector submits & clears  <--+

It uses a LOCAL mock pool (mock_pool.py) so shares are easy to find, but the
GitHub half is REAL (your repo / token).  Demo files jobs_demo.txt and
shares_demo.txt are created in the repo and deleted at the end.

Run:  export GITHUB_TOKEN=ghp_... ; python3 demo_e2e.py
"""

import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import mock_pool as mp

POOL_PORT = 17014
DEMO_PREFIX = "btc-gpu-miner/demo_e2e"
JOBS_GITHUB = f"{DEMO_PREFIX}_jobs.txt"
SHARES_GITHUB = f"{DEMO_PREFIX}_shares.txt"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".demo")


# ---------------------------------------------------------------- github api
def gh_api(method, path, body=None, token=None):
    url = "https://api.github.com" + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "demo-e2e",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            j = json.loads(raw) if raw else None
        except Exception:
            j = None
        return e.code, j


def gh_get(owner, repo, path, token):
    code, j = gh_api("GET", f"/repos/{owner}/{repo}/contents/{path}", token=token)
    if code == 404:
        return None
    if code != 200:
        raise RuntimeError(f"GET {path}: HTTP {code} {j}")
    content = base64.b64decode(j["content"]).decode()
    return {"content": content, "sha": j["sha"]}


def gh_put(owner, repo, path, content, token, sha=None):
    body = {"message": "demo e2e", "content": base64.b64encode(content.encode()).decode(),
            "branch": "main"}
    if sha:
        body["sha"] = sha
    code, j = gh_api("PUT", f"/repos/{owner}/{repo}/contents/{path}", body=body, token=token)
    if code not in (200, 201):
        raise RuntimeError(f"PUT {path}: HTTP {code} {j}")
    return j["content"]["sha"]


def gh_del(owner, repo, path, token):
    got = gh_get(owner, repo, path, token)
    if got is None:
        return
    code, j = gh_api("DELETE", f"/repos/{owner}/{repo}/contents/{path}",
                     body={"message": "demo cleanup", "sha": got["sha"], "branch": "main"},
                     token=token)
    if code not in (200, 204):
        print(f"  cleanup DELETE {path}: HTTP {code}")


# ---------------------------------------------------------------- helpers
def wait_until(desc, timeout, fn):
    start = time.time()
    while time.time() - start < timeout:
        try:
            v = fn()
            if v:
                return v
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"timeout waiting for: {desc}")


def log(*a):
    print("[demo]", *a, flush=True)


def main():
    token = os.environ.get("GITHUB_TOKEN", "")
    owner = os.environ.get("GITHUB_OWNER", "dfghjklfgbhgh")
    repo = os.environ.get("GITHUB_REPO", "test")
    if not token:
        sys.exit("set GITHUB_TOKEN first")

    os.makedirs(LOCAL_DIR, exist_ok=True)
    local_jobs = os.path.join(LOCAL_DIR, "jobs.txt")
    local_shares = os.path.join(LOCAL_DIR, "shares.txt")
    local_prog = os.path.join(LOCAL_DIR, "progress.json")
    conn_cfg = os.path.join(LOCAL_DIR, "connector.json")
    conn_log = os.path.join(LOCAL_DIR, "connector.log")

    pool = mp.MockPool("127.0.0.1", POOL_PORT, share_diff=0.00001)
    pool.start()

    cfg = {
        "pool_url": f"stratum+tcp://127.0.0.1:{POOL_PORT}",
        "worker": "demo/1", "password": "x",
        "github": {"token_env": "GITHUB_TOKEN", "owner": owner, "repo": repo,
                   "branch": "main", "jobs_path": JOBS_GITHUB,
                   "shares_path": SHARES_GITHUB},
        "poll_interval_sec": 2, "keep_jobs": 5, "min_job_push_interval_sec": 1,
    }
    with open(conn_cfg, "w") as f:
        json.dump(cfg, f, indent=2)

    log(f"starting connector (pool=mock, github={owner}/{repo})")
    proc = subprocess.Popen(
        [sys.executable, "pool_connector.py", "--config", conn_cfg],
        stdout=open(conn_log, "w"), stderr=subprocess.STDOUT,
        env={**os.environ, "GITHUB_TOKEN": token})
    connector = None
    try:
        # 1. connector should receive a job from the mock pool and push it
        log("waiting for connector to push a job to GitHub ...")
        got = wait_until("jobs on github", 90, lambda: gh_get(owner, repo, JOBS_GITHUB, token))
        jobs = json.loads(got["content"])
        assert isinstance(jobs, list) and jobs, "jobs list empty"
        job = jobs[-1]
        log(f"job {job['job_id']} received, diff={job['share_diff']}")
        assert job["share_diff"] == 0.00001

        # 2. simulate the manual pull: write jobs to the miner machine
        with open(local_jobs, "w") as f:
            json.dump(jobs, f, indent=2)

        # 3. run the miner (CPU torch, easy difficulty)
        log("running miner ...")
        minr = subprocess.run(
            [sys.executable, "bitcoin_miner.py", "--jobs", local_jobs,
             "--shares", local_shares, "--progress", local_prog,
             "--max-nonces", "2000000", "--stop-on-share"],
            capture_output=True, text=True, timeout=180)
        log(minr.stdout.strip().splitlines()[-3:])
        assert minr.returncode == 0, f"miner failed:\n{minr.stdout}\n{minr.stderr}"
        shares = json.load(open(local_shares))
        assert shares, "miner found no shares"
        log(f"miner found {len(shares)} share(s), e.g. {shares[0]['nonce']}")

        # 4. simulate the manual push: upload shares to GitHub
        log("pushing shares.txt to GitHub ...")
        gh_put(owner, repo, SHARES_GITHUB, json.dumps(shares, indent=2), token)

        # 5. connector must submit them to the (mock) pool and clear the file
        log("waiting for connector to submit & clear ...")
        wait_until("shares cleared on github", 120, lambda: (
            (got := gh_get(owner, repo, SHARES_GITHUB, token)) is not None
            and got["content"].strip() == "[]"))
        wait_until("pool accepted >= 1 share", 60, lambda: pool.accepted >= 1)
        log(f"mock pool accepted {pool.accepted} share(s), rejected {pool.rejected}")
        if pool.rejected:
            log(f"reject reasons: {dict(pool.reasons)}")
        assert pool.accepted >= 1, "no shares accepted by pool!"

        # cross-check: the accepted share matches what the miner recorded
        mshare = shares[0]
        pshare = pool.submitted[0]
        assert mshare["nonce"] == pshare["nonce"] and mshare["job_id"] == pshare["job_id"]
        assert mshare["hash"] == pshare["hash"]
        log("cross-check OK: pool validated the exact share the miner found")

        log("\n=== DEMO PASSED: full pipeline works end-to-end ===")
        print("connector log tail:")
        print(open(conn_log).read().strip()[-1500:])
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
        pool.stop_pool()
        log("cleaning up demo files from GitHub ...")
        gh_del(owner, repo, JOBS_GITHUB, token)
        gh_del(owner, repo, SHARES_GITHUB, token)


if __name__ == "__main__":
    main()
