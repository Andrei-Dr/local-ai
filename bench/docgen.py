#!/usr/bin/env python3
"""Regenerate every derived documentation block from its single source, so docs cannot drift:

  stable/stable.env              -> serving command + build steps (README.md, QUICKSTART.md, research/patches/SERIES.md)
  research/patches/series.toml   -> the patch status table (SERIES.md) and the per-patch guide (research/patches/README.md);
                                    checked 1:1 against research/patches/mainline-series/*.patch
  bench/box/ledger.jsonl         -> the served arc LEGACY -> STABLE (README.md), via ledger2md.served_arc

Each target file holds marker pairs; only the text between them is rewritten.
usage: docgen.py            write the blocks (run by bench/ledger2md.py and the pre-commit hook)
       docgen.py --check    exit 1 if any block is stale or the series check fails (writes nothing)
"""
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATCHES = ROOT / "research" / "patches"


class DocgenError(Exception):
    pass


def begin(name):
    return f"<!-- BEGIN GENERATED: {name} (bench/docgen.py; edit the source, not this block) -->"


def end(name):
    return f"<!-- END GENERATED: {name} -->"


def replace_block(text, name, body):
    b, e = begin(name), end(name)
    i, j = text.find(b), text.find(e)
    if i < 0 or j < i:
        raise DocgenError(f"markers for block '{name}' missing")
    return text[:i] + b + "\n" + body.strip("\n") + "\n" + text[j:]


def parse_env(text):
    """KEY="value" lines of a bash-sourceable env file (comments and blank lines ignored)."""
    env = {}
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = re.match(r'([A-Z0-9_]+)="([^"]*)"$', line.strip())
        if not m:
            raise DocgenError(f"stable.env line {n}: expected KEY=\"value\" alone on the line: {line.strip()}")
        env[m.group(1)] = m.group(2)
    return env


def serving_command(env, ctx):
    return (f"{env['SERVE_ENV']} llama-server -m {env['MODEL']} -md {env['MTP_HEAD']} -c {ctx} {env['SERVE_ARGS']} "
            f"{env['SERVE_CTX_' + ctx]} (+ LLAMA_MTP_VOCAB_FILE={env['MTP_VOCAB']})")


def check_series(patch_dir, entries, themes):
    """Errors for: a patch file without an entry, an entry without a patch file, duplicate entries, unknown themes."""
    files = {p.name for p in Path(patch_dir).glob("*.patch")}
    named = [e["file"] for e in entries]
    errs = [f"{f}: no entry in series.toml" for f in sorted(files - set(named))]
    errs += [f"{f}: listed in series.toml but no patch file" for f in sorted(set(named) - files)]
    errs += [f"{f}: listed more than once in series.toml" for f in sorted({f for f in named if named.count(f) > 1})]
    errs += [f"{e['file']}: unknown theme '{e.get('theme')}'" for e in entries if e.get("theme") not in themes]
    return errs


def check_served_steps(env, steps):
    """The served arc's last step must be the current STABLE (build, model): a promotion has to append its step."""
    last = steps[-1]
    if (last[1], last[2]) != (env.get("STABLE_BOX_BUILD"), env.get("MODEL")):
        return [f"stable/stable.env STABLE ({env.get('STABLE_BOX_BUILD')}, {env.get('MODEL')}) is not the last bench/ledger2md.py "
                f"SERVED_STEPS entry ({last[1]}, {last[2]}): append the promotion's step"]
    return []


def series_table(entries):
    o = ["| # | was | patch | status | switch / default | evidence | gate |", "|---|---|---|---|---|---|---|"]
    for e in entries:
        cells = [e["file"][:4], e["was"], e["title"], e["status"], e["switch"], e["evidence"], e["gate"]]
        o.append("| " + " | ".join(c.replace("\n", " ").replace("|", "\\|") for c in cells) + " |")
    return "\n".join(o)


def patch_guide(entries, themes):
    o = []
    for key, t in themes.items():
        es = [e for e in entries if e["theme"] == key]
        if not es:
            continue
        o += [f"### {t['heading']} ({es[0]['file'][:4]}-{es[-1]['file'][:4]})", t["intro"].strip(), ""]
        for e in es:
            sw = f" Switch: `{e['switch']}`." if e["switch"] and e["switch"] not in ("always", "") else ""
            o.append(f"- **{e['file'][:4]}** {e['plain']}{sw} Status: {e['status']}.")
        o.append("")
    return "\n".join(o)


def serving_block(env):
    ctx = env["SERVE_CTX_DEFAULT"]
    other = [k[len("SERVE_CTX_"):] for k in env if k.startswith("SERVE_CTX_") and k != "SERVE_CTX_DEFAULT" and not k.endswith(ctx)]
    o = [f"STABLE since {env['STABLE_SINCE']}: llama.cpp `{env['LLAMA_CPP_BASE']}` + patches {env['LLAMA_CPP_SERIES']}, model "
         f"`{env['MODEL']}`, draft head `{env['MTP_HEAD']}` with vocabulary `{env['MTP_VOCAB']}`. Source: `stable/stable.env`.",
         "", "```", f"LLAMA_MTP_VOCAB_FILE={env['MTP_VOCAB']} {serving_command(env, ctx).split(' (+ ')[0]}", "```"]
    for c in other:
        o.append(f"At `-c {c}`: `{env['SERVE_CTX_' + c]}` instead of `{env['SERVE_CTX_' + ctx]}`.")
    o.append("`stable/serve.sh` runs exactly this (`--ctx " + " | ".join([ctx] + other) + "`).")
    return "\n".join(o)


def build_block(env):
    return "\n".join(["```", f"git clone {env['LLAMA_CPP_REPO']} llama.cpp && cd llama.cpp",
                      f"git checkout {env['LLAMA_CPP_BASE']}",
                      "git am ../research/patches/mainline-series/*.patch",
                      f"cmake -B build {env['CMAKE_FLAGS']}", "cmake --build build -j --target llama-server", "```",
                      "`stable/build.sh` does exactly this and verifies the patched source tree against `LLAMA_CPP_STABLE_TREE`."])


def hashes_block(env):
    o = ["| file | sha256 |", "|---|---|"]
    for k in ("MODEL", "MTP_HEAD", "MTP_VOCAB"):
        o.append(f"| `{env[k]}` | `{env[k + '_SHA256']}` |")
    return "\n".join(o)


def parse_queue(text):
    """Rows of the box queue file (queue.sh: id, status, tries, started, ended, rc, cmd; tab-separated)."""
    keys = ("id", "status", "tries", "started", "ended", "rc", "cmd")
    return [dict(zip(keys, line.split("\t", 6))) for line in text.splitlines() if line.count("\t") >= 6]


def script_name(cmd):
    m = re.search(r"/ai/bench/([\w.-]+\.(?:sh|py))", cmd or "")
    return m.group(1) if m else None


def purpose(script_text):
    """The job script's first comment line after the shebang (its one-line statement of purpose)."""
    for line in script_text.splitlines()[1:]:
        if line.startswith("#") and line.strip("# ").strip():
            return line.lstrip("#").strip()
    return ""


def queue_board(rows, scripts, recent=8):
    def what(r):
        t = purpose(scripts.get(script_name(r["cmd"]) or "", ""))
        return (t if len(t) <= 150 else t[:147] + "...").replace("|", "\\|")
    o = ["| state | job | started | what it measures (first line of its script) |", "|---|---|---|---|"]
    for st in ("running", "pending"):
        for r in rows:
            if r["status"] == st:
                o.append(f"| {st} | `{r['id']}` | {r['started'][:16] if r['started'] != '-' else '-'} | {what(r)} |")
    fin = sorted((r for r in rows if r["status"] in ("done", "failed")), key=lambda r: r["ended"], reverse=True)[:recent]
    if fin:
        o += ["", "Last finished: " + "; ".join(
            f"`{r['id']}` {r['status']}" + (f" (rc {r['rc']})" if r["status"] == "failed" else "") + f" {r['ended'][:16]}" for r in fin)]
    return "\n".join(o)


LINK = re.compile(r"\]\(([^)#\s]+)")
REPO_PATH = re.compile(r"`((?:bench|research|stable|src|tests)/[\w./-]+\.\w+)`")


def broken_links(root, md_files, path_check=()):
    """Relative markdown links (every file) and backticked repo paths (files in path_check) that point at nothing."""
    errs = []
    for f in md_files:
        text = (Path(root) / f).read_text(errors="ignore")
        for target in LINK.findall(text):
            if re.match(r"[a-z]+:", target):
                continue
            if not (Path(root) / f).parent.joinpath(target).exists():
                errs.append(f"{f}: link to missing {target}")
        if f in path_check:
            for target in REPO_PATH.findall(text):
                if "*" not in target and not (Path(root) / target).exists():
                    errs.append(f"{f}: mentions missing {target}")
    return errs


def headline(rows):
    a, z = rows[0], rows[-1]
    short = [z[k] for k in ("code", "reason", "edit") if z.get(k)]
    writes = short + ([z["dec93"]] if z.get("dec93") else [])
    return (f"prompt reading ~{z['pf93']:.0f} tokens/s at 9.3k tokens ({a['name']} {a['pf93']:.0f}, {z['pf93'] / a['pf93']:.1f}x); "
            f"writing {min(writes):.0f}-{max(writes):.0f} tokens/s ({min(short):.0f}-{max(short):.0f} on short prompts, "
            f"{z['dec93']:.0f} after a 9.3k-token prompt)")


def arc():
    """(markdown, rows) of the served arc from the synced run ledger."""
    sys.path.insert(0, str(ROOT / "bench"))
    import ledger2md
    recs = ledger2md.load(ROOT / "bench" / "ledger_backfill.jsonl") + ledger2md.load(ROOT / "bench" / "box" / "ledger.jsonl")
    rows, kinds = ledger2md.served_arc_rows(recs)
    md = "\n".join(line.replace("## Served arc", "### Served arc") for line in ledger2md.served_arc_md(rows, kinds))
    return md.strip("\n"), rows


def model_files_block(env):
    o = ["| file | what | where from |", "|---|---|---|"]
    for k, what in (("MTP_HEAD", "draft (MTP) head for speculative decoding"), ("MODEL", "the STABLE model"),
                    ("MTP_VOCAB", "the draft head's vocabulary (patch 0022)")):
        o.append(f"| `{env[k]}` | {what} | {env[k + '_SOURCE']} |")
    return "\n".join(o)


# Kept current (path mentions checked); notes.md is an append-only log and snapshots are dated, so they are link-checked only.
LIVING = ("research/design-harmony-ledger.md", "research/patches/README.md", "research/patches/SERIES.md")


def git(*args):
    import subprocess
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout


def research_index():
    files = [f for f in git("ls-files", "research/*.md").split() if f != "research/README.md"]
    def entry(f):
        lines = (ROOT / f).read_text(errors="ignore").splitlines()
        title = next((x.lstrip("# ").strip() for x in lines if x.startswith("#")), Path(f).stem)
        born = (git("log", "--diff-filter=A", "--format=%cs", "--", f).split() or ["?"])[-1]
        return f"- [{Path(f).relative_to('research')}]({Path(f).relative_to('research')}) — {title} ({born})"
    living = [entry(f) for f in files if f in LIVING]
    snaps = sorted((entry(f) for f in files if f not in LIVING), key=lambda e: e.rsplit("(", 1)[-1], reverse=True)
    return "\n".join(["**Living documents** (kept current):", "", *living, "",
                      "**Dated snapshots** (true as of the date shown; never updated, newest first):", "", *snaps])


def box_queue():
    q = ROOT / "bench" / "box" / "queue.tsv"
    if not q.exists():
        return "(no synced queue: run bench/closeout.py)"
    rows = parse_queue(q.read_text())
    scripts = {}
    for r in rows:
        n = script_name(r["cmd"])
        if n and (ROOT / "bench" / "box" / n).exists():
            scripts[n] = (ROOT / "bench" / "box" / n).read_text(errors="ignore")
    latest = max((t for r in rows for t in (r["started"], r["ended"]) if t != "-"), default="?")[:16]
    return f"As of the last `bench/closeout.py` sync (latest queue event {latest}).\n\n" + queue_board(rows, scripts)


def render():
    """{path: {block name: body}} plus the series errors."""
    env = parse_env((ROOT / "stable" / "stable.env").read_text())
    data = tomllib.loads((PATCHES / "series.toml").read_text())
    entries = sorted(data["patch"], key=lambda e: e["file"])
    errs = check_series(PATCHES / "mainline-series", entries, data["themes"])
    sys.path.insert(0, str(ROOT / "bench"))
    import ledger2md
    errs += check_served_steps(env, ledger2md.SERVED_STEPS)
    serving = serving_block(env)
    arc_md, rows = arc()
    head = headline(rows)
    md_files = git("ls-files", "*.md").split()
    errs += broken_links(ROOT, md_files, path_check=[f for f in md_files if f in LIVING or ("/" not in f and f != "notes.md") or f.startswith("stable/")])
    blocks = {
        ROOT / "README.md": {"headline": head, "served-arc": arc_md, "serving": serving},
        ROOT / "QUICKSTART.md": {"headline": head, "serving": serving, "build": build_block(env), "model-files": model_files_block(env)},
        ROOT / "SPEC.md": {"box-queue": box_queue()},
        ROOT / "research" / "README.md": {"research-index": research_index()},
        PATCHES / "SERIES.md": {"serving": serving, "series-table": series_table(entries)},
        PATCHES / "README.md": {"headline": head, "patch-guide": patch_guide(entries, data["themes"])},
        ROOT / "stable" / "MODELS.md": {"model-hashes": hashes_block(env)},
    }
    return blocks, errs


def main(argv):
    check = "--check" in argv
    blocks, errs = render()
    if errs:
        print("docs sources disagree:\n  " + "\n  ".join(errs))
        return 1
    stale = []
    for path, bs in blocks.items():
        old = path.read_text()
        new = old
        for name, body in bs.items():
            new = replace_block(new, name, body)
        if new != old:
            stale.append(path.relative_to(ROOT))
            if not check:
                path.write_text(new)
    if stale:
        print(("stale: " if check else "regenerated: ") + ", ".join(map(str, stale)))
    return 1 if check and stale else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
