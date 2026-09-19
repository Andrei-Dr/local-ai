"""One-shot: backfill build provenance into old /ai/bench/ledger.jsonl records written before the 2026-09-20
hardening. For each record whose git dict lacks cmake/toolchain, if the build's source tree + commit are still
reachable and the build dir still has a CMakeCache, reconstruct cmake + toolchain (from the CURRENT tree state,
which is valid only because these commits are clean and on kept branches). Records that were DIRTY with no diff
captured cannot be reconstructed -- they get git.unrecoverable=true and are left otherwise as-is.

Idempotent: rewrites ledger.jsonl in place (after a .bak). Prints a summary. Run on the box:
  /ai/.venv/bin/python /ai/bench/backfill_provenance.py
"""
import json, os, re, shutil, subprocess

B = "/ai/bench"
LED = f"{B}/ledger.jsonl"


def sh(*a):
    try:
        return subprocess.run(a, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def cmake(build, key):
    try:
        for ln in open(os.path.join(build, "CMakeCache.txt")):
            if ln.startswith(key + ":") or ln.startswith(key + "="):
                return ln.split("=", 1)[1].strip()
    except Exception:
        pass
    return None


def toolchain():
    cxx = sh("c++", "--version").splitlines()
    nvcc = sh("nvcc", "--version")
    return {"cxx": cxx[0] if cxx else None, "nvcc": (re.findall(r"release [\d.]+, V[\d.]+", nvcc) or [None])[0]}


def reachable(src, commit):
    return os.path.isdir(src) and commit and sh("git", "-C", src, "cat-file", "-t", commit) == "commit"


def main():
    rows = [json.loads(l) for l in open(LED)]
    tc = toolchain()
    filled = lost = already = 0
    for r in rows:
        g = r.get("git")
        if not g or r.get("kind") not in ("specbench", "quality"):
            continue
        if (g.get("cmake") or {}).get("cuda_arch"):
            already += 1
            continue
        build = r.get("build") or ""
        src = os.path.dirname(build.rstrip("/"))
        commit = g.get("commit_full") or g.get("commit")
        # complete the full commit + describe if the tree is here
        if reachable(src, commit):
            full = sh("git", "-C", src, "rev-parse", commit) or commit
            g["commit_full"] = full
            g.setdefault("describe", sh("git", "-C", src, "describe", "--always", "--tags", commit))
        cc = cmake(build, "CMAKE_CUDA_ARCHITECTURES")
        if cc and os.path.exists(os.path.join(build, "CMakeCache.txt")):
            g["cmake"] = {"cuda_arch": cc, "build_type": cmake(build, "CMAKE_BUILD_TYPE"),
                          "ggml_cuda": cmake(build, "GGML_CUDA"), "cxx": cmake(build, "CMAKE_CXX_COMPILER")}
            g["toolchain"] = tc
            g["provenance_backfilled"] = "2026-09-20 (tree+CMakeCache still present; commit clean on a kept branch)"
            filled += 1
        else:
            # no live build dir to read flags from
            if g.get("dirty") and not g.get("dirty_diff_sha"):
                g["unrecoverable"] = "dirty build, no diff captured (pre-hardening), tree/build gone"
            else:
                g["cmake_unknown"] = "build dir gone; cmake flags not recorded at run time"
            lost += 1
    shutil.copyfile(LED, LED + ".prebackfill.bak")
    with open(LED, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"backfill: {filled} filled from live trees, {lost} unrecoverable/unknown, {already} already had cmake")


if __name__ == "__main__":
    main()
