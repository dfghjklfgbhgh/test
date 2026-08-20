# BTC GPU Miner — Stratum pool connector + PyTorch miner (GitHub-synced)

A complete, working **educational** Bitcoin-mining pipeline split across two
machines, connected only through GitHub files:

```
        ┌──────────────────────────────┐          ┌──────────────────────────────┐
        │   CONNECTOR MACHINE          │          │   MINER MACHINE (GPU)        │
        │   (has internet)             │          │   (may be offline)           │
        │                              │          │                              │
        │  pool_connector.py           │          │  bitcoin_miner.py            │
        │   │ connects to Stratum pool │          │   │ reads jobs.txt           │
        │   │ receives mining jobs     │          │   │ mines nonces on GPU      │
        │   │ writes jobs.txt ─────────┼──────────┼─▶ │ (PyTorch / CUDA)          │
        │   │ watches shares.txt ◀─────┼──────────┼── │ writes shares.txt         │
        │   │ submits shares to pool   │          │   │                          │
        │   │ clears shares.txt        │          │   │   (you push/pull these   │
        │   └──────────────────────────┘          │   │    files manually)       │
        └──────────────────────────────┘          └──────────────────────────────┘
```

The GitHub repo is just a mailbox: **jobs.txt** carries work to the miner,
**shares.txt** carries results back.

> Repo layout: the code lives in the `btc-gpu-miner/` folder; the mailbox
> files **`jobs.txt`** and **`shares.txt`** live at the repository **root**
> (that is what the config points at).

---

## ⚠️ Read this first (honest engineering notes)

1. **This will not mine profitable Bitcoin.** Real Bitcoin mining is done by
   ASIC hardware at tera-hashes-per-second. A GPU running Python/PyTorch is
   orders of magnitude too slow, and the Kryptex pool assigns a share
   difficulty of **524288** (≈2²⁵¹ hashes per share) — effectively
   unreachable for this miner. Treat this as a **fully working simulation of
   the real Stratum mining math**, verified against real blocks (see
   `verify_mining.py`) and fully testable with the bundled mock pool.

2. **Your GitHub token was shared in this chat.** It is a real credential,
   even if created for testing. After you finish testing, **revoke it** at
   github.com → Settings → Developer settings → Personal access tokens, and
   create a new one with only `repo` scope. Never paste tokens into chats.

3. The connector was tested live against `stratum+tcp://btc.kryptex.network:7014`
   with your worker string `krxYRPV4WQ.0x` — subscribe and authorize both
   succeed, and real jobs were received and pushed to `jobs.txt` in the repo
   (the pool starts at difficulty 1 and ramps up as it estimates your hashrate;
   after a while it sends ~262144, which is far beyond any GPU/Python miner).

---

## Files

| file (in `btc-gpu-miner/`) | purpose |
|---|---|
| `pool_connector.py` | Stratum v1 client + GitHub Contents-API sync (pure stdlib). Pushes jobs to the repo-root GitHub `jobs.txt`; watches GitHub `shares.txt`, submits new shares to the pool, then clears the file. |
| `bitcoin_miner.py` | Reads `jobs.txt`, mines the block-header nonce with a vectorised SHA-256d kernel in PyTorch (CUDA or CPU), writes shares to `shares.txt`. |
| `mock_pool.py` | Local Stratum test pool that *independently validates* every share (recomputes coinbase → merkle → header → double-SHA-256). |
| `verify_mining.py` | Proves the crypto against **real mined Bitcoin blocks** (blockstream.info) plus torch-vs-hashlib equivalence. |
| `demo_e2e.py` | End-to-end pipeline test: mock pool → connector → GitHub → miner → GitHub → connector submit & clear. |
| `config.example.json` | Template for connector configuration. |

---

## 1. Pool connector (runs on the machine with internet)

Only the Python standard library is needed (Python ≥ 3.8).

```bash
cp config.example.json config.json    # edit if you like
export GITHUB_TOKEN=ghp_...           # your personal access token (repo scope)

python3 pool_connector.py --config config.json
```

Or fully from the command line:

```bash
python3 pool_connector.py \
  --pool stratum+tcp://btc.kryptex.network:7014 \
  --worker krxYRPV4WQ.0x --password x \
  --owner dfghjklfgbhgh --repo test
```

What it does, continuously:

1. Connects to the pool, `mining.subscribe` + `mining.authorize`.
2. On every `mining.notify` / `mining.set_difficulty`, rewrites
   `jobs.txt` in the GitHub repo (JSON array; newest job last).
3. Polls GitHub `shares.txt` every `poll_interval_sec` (default 10 s).
   When it changes, it downloads the shares, submits each with
   `mining.submit`, then **rewrites `shares.txt` as `[]`** (clears it) and
   waits for the next update.
4. Auto-reconnects with backoff if the pool connection drops.

### `config.json` fields

```jsonc
{
  "pool_url": "stratum+tcp://btc.kryptex.network:7014",
  "worker": "krxYRPV4WQ.0x",          // wallet string / worker name
  "password": "x",
  "github": {
    "token_env": "GITHUB_TOKEN",       // env var holding the PAT
    "owner": "dfghjklfgbhgh",
    "repo": "test",
    "branch": "main",
    "jobs_path": "jobs.txt",           // file in the repo for jobs
    "shares_path": "shares.txt"        // file in the repo for shares
  },
  "poll_interval_sec": 10,
  "keep_jobs": 5,
  "min_job_push_interval_sec": 15
}
```

---

## 2. Miner (runs on the GPU machine, can be fully offline)

Needs PyTorch with CUDA. Install on the GPU machine, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
```

Run (with `jobs.txt` in the folder — the file you pulled from GitHub):

```bash
python3 bitcoin_miner.py --jobs jobs.txt --shares shares.txt --progress progress.json
```

Flags:

| flag | meaning |
|---|---|
| `--jobs FILE` | jobs file (default `jobs.txt`) |
| `--shares FILE` | shares output file (default `shares.txt`) |
| `--batch N` | nonces per kernel launch (default 2²⁰; auto-capped on CPU) |
| `--diff N` | override the pool share difficulty (e.g. `0.00001` for tests) |
| `--max-nonces N` | stop after N nonces (0 = never) |
| `--stop-on-share` | exit as soon as one share is found |
| `--check-interval S` | how often to re-read `jobs.txt` for a new job |
| `--cpu` | force CPU even if CUDA is available |

Behaviour:

* picks the **newest** job in `jobs.txt`;
* scans the 32-bit nonce space in batches (resume via `progress.json`);
* when the nonce space is exhausted, rolls a new `extranonce2` and continues;
* every valid share (header double-SHA-256 ≤ pool target) is appended to
  `shares.txt` (JSON array, deduplicated by `share_id`);
* on the next job in `jobs.txt` it switches automatically.

### The manual sync (your part)

```text
connector writes jobs.txt ──▶ you download it ──▶ miner machine (jobs.txt)
miner writes shares.txt   ◀── you upload it   ◀── (shares.txt)
```

The connector watches GitHub `shares.txt`; the moment it changes it submits
and clears it, so you can keep pushing `shares.txt` whenever the miner
finds new shares.

---

## 3. Testing everything (no real pool needed)

```bash
# 1. prove the crypto against real Bitcoin blocks (needs internet)
python3 verify_mining.py

# 2. full pipeline test: local mock pool + REAL GitHub sync
export GITHUB_TOKEN=ghp_...
python3 demo_e2e.py
```

`demo_e2e.py` creates `demo_e2e_jobs.txt` / `demo_e2e_shares.txt` in the
repo, runs the whole loop (connector → GitHub → miner → GitHub → submit &
clear), checks the mock pool *independently validated* the share, then
deletes the demo files.

You can also run the mock pool by hand and point the connector at it:

```bash
python3 mock_pool.py --port 17014 --diff 0.00001
python3 pool_connector.py --pool stratum+tcp://127.0.0.1:17014 --worker demo/1 --owner YOU --repo YOURREPO
```

---

## How the mining maths works (summary)

* coinbase tx = `coinb1 ‖ extranonce1 ‖ extranonce2 ‖ coinb2`
* merkle root = SHA256d-chain of the coinbase hash with the pool's
  `merkle_branch` (displayed order, larger pair first)
* block header = version ‖ prevhash ‖ merkle ‖ time ‖ bits ‖ nonce
  (80 bytes, Stratum field conventions — verified against real blocks)
* block hash = SHA256d(header); a **share** is any nonce whose hash
  (big-endian) ≤ `diff1_target / share_difficulty`
* PyTorch kernel: header block 1 is constant per (job, extranonce2), so only
  block 2 + the second SHA-256 pass are computed per candidate, vectorised
  as int32 tensors (all arithmetic mod 2³²).

---

## Real-pool test results (Kryptex, `btc.kryptex.network:7014`)

Tested live with worker `krxYRPV4WQ.0x` (the exact pipeline, no mock):

* subscribe / authorize: **OK** (`extranonce1` + `extranonce2_size=6` issued)
* `mining.suggest_difficulty [0.00001]`: **ignored** — pool kept difficulty
* pool difficulty set: **524288** (share target `0x0000000000001fff...`,
  i.e. the first **51 bits** of the block hash must be zero → `2^51` hashes per share)
* a real job was mined (2.5–3 M nonces, ~20 s) and the **best share** was submitted

```
POOL RESPONSE: {"id": 3, "result": false, "error": [-1, "Invalid share", null]}
```

Control tests against the same pool to interpret that answer:

| submitted | pool response |
|---|---|
| well-formed share, fresh job, below target | `[-1, "Invalid share"]` |
| unknown job id | `[21, "Job not found"]` |
| malformed extranonce2 / bad nonce | *(silently dropped, no response)* |

**Verdict: shares are NOT accepted by the real pool** — because the hash
does not meet the pool's share target, not because the pipeline is broken:
* the pool parses our shares, matches them to a real job and validates them
  (garbage is dropped, unknown jobs get a distinct error, well-formed shares
  get a real hash-vs-target verdict);
* the identical code **does** get `ACCEPTED` at a feasible difficulty (mock pool);
* the gap is pure hashrate: our best hash after 3 M tries had the first
  21 bits zero; the pool needs 51 → ~2³⁰ ≈ 1 billion × more work, i.e.
  ~163,000 days per share at this CPU's 0.16 MH/s (≈52 days even at
  500 MH/s on a fast GPU).

Re-run the test any time:

```bash
python3 real_pool_test.py --worker krxYRPV4WQ.0x --max-nonces 2500000
```

## Troubleshooting

* **authorize failed** — check the worker string (Kryptex format is
  `wallet.worker` or `username/worker`, password `x`).
* **`jobs.txt` empty / no jobs** — the pool sends the first job right after
  `authorize`; wait a few seconds, then check the GitHub file.
* **shares rejected with "low difficulty share"** — expected: the pool sets a
  difficulty (e.g. 524288) far above what this miner can reach. Use
  `--diff` or the mock pool for testing.
* **GitHub 409 conflict** — the connector retries automatically; if you are
  pushing `shares.txt` at the exact same moment it clears it, just push again.
* **`share not found` / stale job** — the job was rotated before the share
  reached the pool (inherent to manual sync); the miner picks up new jobs
  from `jobs.txt` automatically.
