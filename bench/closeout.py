#!/usr/bin/env python3
"""closeout.py [JOB ...] -- run at every box-job boundary: one command instead of a checklist.

1. sync      box -> repo: run ledger, quality results, the queue file, and the logs of finished jobs (bench/box/logs/)
2. drift     box job scripts vs their bench/box/ mirror; the STABLE build's source tree and model hashes vs stable/stable.env
3. docs      bench/ledger2md.py (LEDGER / SCOREBOARD / BUILDS) and bench/docgen.py (every generated block)
4. to-do     finished jobs without a notes.md entry, plus the judgment updates a result can require (listed, not automated)
Exit 1 when a drift check fails, so it cannot scroll by unnoticed.
"""
import hashlib
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOX, BENCH = "root@i5.local", "/ai/bench"
MIRROR = ROOT / "bench" / "box"


def parse_sha256sum(out):
    return {m.group(2): m.group(1) for m in re.finditer(r"^([0-9a-f]+)\s+\*?(\S+)$", out, re.M)}


def hash_diff(local, box):
    return {"differs": sorted(n for n in local if n in box and local[n] != box[n]),
            "missing_on_box": sorted(set(local) - set(box)),
            "box_only": sorted(set(box) - set(local))}


def jobs_without_notes(rows, notes, since):
    """Ids of jobs finished (done / failed) on or after `since` that notes.md never names as a word."""
    return [r["id"] for r in rows if r["status"] in ("done", "failed") and r["ended"][:10] >= since
            and not re.search(rf"(?<![\w-]){re.escape(r['id'])}(?![\w-])", notes)]


def ssh(cmd, timeout=600):
    r = subprocess.run(["ssh", BOX, cmd], capture_output=True, text=True, timeout=timeout)
    return r.stdout


def rsync(src, dst):
    subprocess.run(["rsync", "-a", f"{BOX}:{src}", str(dst)], check=True)


def sync(jobs):
    rsync(f"{BENCH}/ledger.jsonl", MIRROR / "ledger.jsonl")
    rsync(f"{BENCH}/queue.tsv", MIRROR / "queue.tsv")
    rsync(f"{BENCH}/qual/results/", ROOT / "bench" / "qual" / "results")
    for j in jobs:
        subprocess.run(["rsync", "-a", f"{BOX}:{BENCH}/{j}.log", str(MIRROR / "logs" / f"{j}.log")])


def script_drift():
    """/ai/bench scripts vs the repo. The mirror is bench/box/ (box-only scripts) plus bench/ (scripts shared with the Mac, per
    research/MIGRATION-flatten-bench.md); bench/*.py also holds Mac-only tools, so only bench/box/ files count as missing."""
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    shared = {p.name: sha(p) for p in sorted((ROOT / "bench").glob("*.sh")) + sorted((ROOT / "bench").glob("*.py"))}
    boxdir = {p.name: sha(p) for p in sorted(MIRROR.glob("*.sh")) + sorted(MIRROR.glob("*.py"))}
    box = parse_sha256sum(ssh(f"cd {BENCH} && sha256sum *.sh *.py stable.env 2>/dev/null"))
    boxdir["stable.env"] = sha(ROOT / "stable" / "stable.env")  # the box's copy of THE STABLE definition (bench/box/stable.sh)
    d = hash_diff({**{n: h for n, h in shared.items() if n in box}, **boxdir}, box)
    return d


BOX_STABLE = r'''
import hashlib, json, os, subprocess, sys
env = json.loads(sys.argv[1]); out = {}
src = os.path.dirname(env["STABLE_BOX_BUILD"].rstrip("/"))
g = lambda *a: subprocess.run(["git", "-C", src, *a], capture_output=True, text=True).stdout.strip()
out["tree"] = g("rev-parse", "HEAD^{tree}"); out["dirty"] = bool(g("status", "--porcelain", "--untracked-files=no"))
cache_p = "/ai/bench/.sha256cache.json"
cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
for k in ("MODEL", "MTP_HEAD", "MTP_VOCAB"):
    p = os.path.realpath(os.path.join("/ai/models", env[k]))
    if not os.path.exists(p):
        out[k] = "missing"; continue
    st = os.stat(p); key = f"{p}:{st.st_size}:{int(st.st_mtime)}"
    if key not in cache:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        cache[key] = h.hexdigest()
    out[k] = cache[key]
json.dump(cache, open(cache_p, "w"))
print(json.dumps(out))
'''


def stable_drift(env):
    """Problems: the box's STABLE source tree or model files disagree with stable/stable.env."""
    arg = json.dumps({k: env[k] for k in ("STABLE_BOX_BUILD", "MODEL", "MTP_HEAD", "MTP_VOCAB")})
    r = subprocess.run(["ssh", BOX, "/ai/.venv/bin/python - " + shlex.quote(arg)], input=BOX_STABLE,
                       capture_output=True, text=True, timeout=900)
    try:
        got = json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return [f"could not read the box's STABLE state: {r.stderr.strip()[-300:]}"]
    errs = []
    if got["tree"] != env["LLAMA_CPP_STABLE_TREE"]:
        errs.append(f"box STABLE source tree {got['tree']} != stable.env {env['LLAMA_CPP_STABLE_TREE']}")
    if got["dirty"]:
        errs.append("box STABLE source tree has uncommitted changes")
    for k in ("MODEL", "MTP_HEAD", "MTP_VOCAB"):
        if got[k] != env[k + "_SHA256"]:
            errs.append(f"box {env[k]}: sha256 {got[k][:16]}... != stable.env {env[k + '_SHA256'][:16]}...")
    return errs


TODO = """Judgment updates this result may need (not automatable; skip what does not apply):
  - notes.md: a dated entry with the numbers and the verdict against the job's pre-registered rule
  - research/design-harmony-ledger.md: an outcomes row if the job tested a move
  - SPEC.md: the decision / owed-item rows it settles (sections 1.0 B and C)
  - a promotion: stable/stable.env + series.toml entries + a SERVED_STEPS entry (docgen enforces the last two)
  - research/RESUME.md (local): the NOW block"""


def main(argv):
    jobs = [a for a in argv if not a.startswith("-")]
    sys.path.insert(0, str(ROOT / "bench"))
    import docgen
    env = docgen.parse_env((ROOT / "stable" / "stable.env").read_text())
    print("== 1. sync")
    sync(jobs)
    rows = docgen.parse_queue((MIRROR / "queue.tsv").read_text())
    failed = False
    print("== 2. drift")
    d = script_drift()
    for k, names in d.items():
        if names and k != "missing_on_box":  # Mac-side tools and retired jobs live only in the repo
            failed = True
        if names:
            print(f"  scripts {k}: {', '.join(names)}")
    for e in stable_drift(env):
        failed = True
        print(f"  {e}")
    if not failed:
        print("  none")
    print("== 3. docs")
    subprocess.run([sys.executable, str(ROOT / "bench" / "ledger2md.py")], check=True)
    print("== 4. to-do")
    since = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    missing = jobs_without_notes(rows, (ROOT / "notes.md").read_text(), since)
    print(f"  finished since {since} without a notes.md entry: {', '.join(missing) or 'none'}")
    run = [r for r in rows if r["status"] in ("running", "pending")]
    print("  queue: " + " -> ".join(f"{r['id']}" + (" (running since " + r["started"][11:16] + ")" if r["status"] == "running" else "") for r in run))
    print(TODO)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
