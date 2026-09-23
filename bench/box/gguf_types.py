#!/usr/bin/env python3
"""gguf_types.py SRC [SRC ...] [--emit-quantize FILE] [--base TYPE] [--md]

Read the per-tensor QUANTIZATION RECIPE of a published dynamic GGUF — from a local file OR from an HTTP
URL with Range requests only (start 8 MiB, doubled until the header + tensor-info block parses, cap
512 MiB; tensor DATA is never fetched) — and, with --emit-quantize, write the llama-quantize override
lines that replay that recipe on our own source model:
    --tensor-type "^blk\\.7\\.ffn_gate_exps\\.weight$"=q2_k
Several SRCs = shards of one model: tensor lists merge; a duplicate tensor name across shards is an
error (exit 2), as is busting the 512 MiB cap or an empty/unparsable header.

GGUF v2/v3 little-endian is parsed by hand (stdlib only): magic, version, tensor_count, kv_count, every
KV skipped by its value type (scalars, strings, arrays incl. arrays of strings and nested arrays), then
tensor infos (name, n_dims, dims, ggml type id, offset). general.architecture / general.file_type are
kept. Unknown type ids become TYPE_<id> — never a crash — and are EXCLUDED from bpw totals with a note.

Report: per role (tensor name minus the leading blk.<n>. ) tensor count, weights, type histogram, mean
bits-per-weight (block-bytes/block-weights table below); a role whose type varies across layers also
gets a run-length line like  ffn_up_exps: L0-3 Q4_K, L4-35 Q2_K, L36-39 Q3_K.  Totals: weights,
estimated bytes, overall bpw, split into experts (name contains _exps) vs the rest.

Base type for overrides: --base, else the type covering the most WEIGHTS among QUANTIZED tensors
(not-F32/F16/BF16/I8/I16/I32/I64). Tensors whose type equals the base, plus F32/I*, emit nothing
(llama-quantize keeps those itself). Exit codes: 0 ok, 2 usage-ish (dup name, cap, no parse).
"""
import argparse, json, re, struct, sys, urllib.error, urllib.request

CAP = 512 * (1 << 20)
START = 8 * (1 << 20)
TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K",
         11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
         18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8",
         25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
         39: "MXFP4"}
BPW = {"F32": 32.0, "F16": 16.0, "BF16": 16.0, "Q8_0": 8.5, "Q6_K": 6.5625, "Q5_K": 5.5, "Q5_1": 6.0,
       "Q5_0": 5.5, "Q4_K": 4.5, "Q4_1": 5.0, "Q4_0": 4.5, "IQ4_NL": 4.5, "IQ4_XS": 4.25, "Q3_K": 3.4375,
       "IQ3_S": 3.4375, "IQ3_XXS": 3.0625, "Q2_K": 2.625, "IQ2_S": 2.5, "IQ2_XS": 2.3125, "IQ2_XXS": 2.0625,
       "IQ1_M": 1.75, "IQ1_S": 1.5625, "TQ2_0": 2.0625, "TQ1_0": 1.6875, "MXFP4": 4.25,
       "F64": 64.0, "F16 ": 16.0, "I8": 8.0, "I16": 16.0, "I32": 32.0, "I64": 64.0}
SKIP_FAMILY = {"F32", "F16", "BF16", "I8", "I16", "I32", "I64"}      # never base candidates
EMIT_SKIP = {"F32", "BF16", "I8", "I16", "I32", "I64"}               # no override lines at all


class ToolError(Exception):
    def __init__(self, msg, rc=2):
        Exception.__init__(self, msg)
        self.rc = rc


class Trunc(Exception):
    pass


def tname(tid):
    return TYPES.get(tid, "TYPE_%d" % tid)


def tbpw(name):
    return BPW.get(name)


# ---------------------------------------------------------------- GGUF header decode
# canonical GGUF value-type ids: 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32, 7 bool, 8 str,
# 9 array, 10 u64, 11 i64, 12 f64, 26 f16
_VSZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8, 26: 2}
_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 10: "<Q", 11: "<q", 12: "<d",
        26: "<e"}


class R:
    def __init__(self, buf):
        self.b, self.p = buf, 0

    def take(self, n):
        if self.p + n > len(self.b):
            raise Trunc(self.p)
        v = self.b[self.p:self.p + n]
        self.p += n
        return v

    def u8(self):
        return self.take(1)[0]

    def u32(self):
        return struct.unpack_from("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack_from("<Q", self.take(8))[0]

    def s(self):
        return self.take(self.u64()).decode("utf-8", "replace")

    def val(self, t, store=True):
        if not store:
            self.skip1(t)
            return None
        if t == 8:
            return self.s()
        if t == 9:
            et = self.u32()
            return [self.val(et, True) for _ in range(self.u64())]
        if t == 7:
            return bool(self.u8())
        if t in _FMT:
            return struct.unpack_from(_FMT[t], self.take(_VSZ[t]))[0]
        raise ToolError("gguf value type %d unsupported" % t)

    def skip1(self, t):
        if t == 8:
            self.s()
        elif t == 9:
            et, n = self.u32(), self.u64()
            if et in _VSZ:
                self.take(_VSZ[et] * n)                    # bulk skip of primitive arrays
            else:
                for _ in range(n):
                    self.skip1(et)
        elif t in _VSZ:
            self.take(_VSZ[t])
        else:
            raise ToolError("gguf value type %d unsupported" % t)


def decode_header(buf):
    r = R(buf)
    if r.take(4) != b"GGUF":
        raise ToolError("not a GGUF (bad magic)")
    ver = r.u32()
    if ver not in (2, 3):
        raise ToolError("GGUF version %d unsupported (need v2/v3)" % ver)
    n_tensors = r.u64()
    n_kv = r.u64()
    meta = {}
    for _ in range(n_kv):
        key = r.s()
        t = r.u32()
        keep = key in ("general.architecture", "general.file_type")
        v = r.val(t, store=keep)
        if keep:
            meta[key] = v
    infos = []
    dimfmt = "<I" if ver == 2 else "<Q"
    dw = 4 if ver == 2 else 8
    for _ in range(n_tensors):
        name = r.s()
        nd = r.u32()
        dims = [struct.unpack_from(dimfmt, r.take(dw))[0] for _ in range(nd)]
        tid = r.u32()
        off = r.u64()
        infos.append({"name": name, "dims": dims, "type": tname(tid), "offset": off})
    return meta, infos


# ---------------------------------------------------------------- sources
def _urlopen_range(url, lo, hi):
    req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (lo, hi)})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    except (urllib.error.URLError, OSError) as e:
        raise ToolError("fetch failed: %s" % e)


def windowed_decode(fetch, label, ranges_log):
    fetch = fetch or _urlopen_range
    n = START
    total = None
    while True:
        chunk = fetch(label, 0, (n - 1) if total is None else min(total - 1, n - 1))
        ranges_log.append((0, len(chunk) - 1, n))
        if total is None and len(chunk) < n:
            total = len(chunk)
        try:
            return decode_header(chunk)
        except Trunc:
            if total is not None:
                raise ToolError("%s: header truncated at %d bytes" % (label, total))
            n *= 2
            if n > CAP:
                raise ToolError("%s: header/info block exceeds the %d MiB cap" % (label, CAP >> 20))


def _local_window(path):
    def f(_, lo, hi):
        with open(path, "rb") as fh:
            fh.seek(lo)
            return fh.read(hi - lo + 1)
    return f


def analyze(srcs, opener=None):
    infos, metas, ranges = [], {}, []
    def fetch_for(s):
        if s.startswith("http://") or s.startswith("https://"):
            return opener or _urlopen_range
        return _local_window(s)
    for s in srcs:
        try:
            meta, lst = windowed_decode(fetch_for(s), s, ranges)
        except OSError as e:
            raise ToolError("cannot read %s: %s" % (s, e))
        seen = {t["name"] for t in infos}
        dupes = [t["name"] for t in lst if t["name"] in seen]
        if dupes:
            raise ToolError("duplicate tensor names across shards: %s" % dupes[:5])
        for k, v in meta.items():
            metas.setdefault(k, v)
        infos += lst
    if not infos:
        raise ToolError("no tensors parsed")
    return {"meta": metas, "tensors": infos, "ranges": ranges}


# ---------------------------------------------------------------- stats + report
BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


def role_of(name):
    m = BLK_RE.match(name)
    return m.group(2) if m else name


def layer_of(name):
    m = BLK_RE.match(name)
    return int(m.group(1)) if m else None


def weights_of(t):
    w = 1
    for d in t["dims"]:
        w *= d
    return w


def role_stats(tensors):
    by = {}
    for t in tensors:
        r = by.setdefault(role_of(t["name"]), {"n": 0, "weights": 0, "hist": {}, "layers": [],
                                               "bytes": 0, "excluded": 0})
        w = weights_of(t)
        b = BPW.get(t["type"])
        r["n"] += 1
        r["weights"] += w
        r["hist"][t["type"]] = r["hist"].get(t["type"], 0) + 1
        r["layers"].append((layer_of(t["name"]), t["type"], w))
        if b is None:
            r["excluded"] += 1
        else:
            r["bytes"] += w * b / 8.0
    for r in by.values():
        r["bpw"] = round(r["bytes"] * 8 / r["weights"], 2) if r["weights"] and not r["excluded"] else None
        r["layers"].sort(key=lambda x: (-1 if x[0] is None else x[0]))
    return by


def runlength(entries):
    """entries [(layer,type,w)] sorted by layer -> 'L0-3 Q4_K, L4 Q2_K' for layer-bearing members."""
    lay = [(l, t) for l, t, _ in entries if l is not None]
    if len({t for _, t in lay}) < 2:
        return None
    out, i = [], 0
    while i < len(lay):
        j = i
        while j + 1 < len(lay) and lay[j + 1][1] == lay[i][1] and lay[j + 1][0] == lay[j][0] + 1:
            j += 1
        out.append(("L%d" % lay[i][0]) if i == j else ("L%d-%d" % (lay[i][0], lay[j][0])))
        out[-1] += " " + lay[i][1]
        i = j + 1
    return ", ".join(out)


def totals(tensors):
    def side(ts):
        meas = [t for t in ts if BPW.get(t["type"]) is not None]
        w = sum(weights_of(t) for t in meas)
        b = sum(weights_of(t) * BPW[t["type"]] / 8.0 for t in meas)
        return {"n": len(ts), "weights": sum(weights_of(t) for t in ts), "excluded_n": len(ts) - len(meas),
                "est_bytes": int(b), "bpw": round(b * 8 / w, 2) if w else None}
    exps = [t for t in tensors if "_exps" in t["name"]]
    rest = [t for t in tensors if "_exps" not in t["name"]]
    out = side(tensors)
    out["experts"] = side(exps)
    out["rest"] = side(rest)
    return out


def base_type(tensors, override=None):
    if override:
        return override
    best, bw = None, -1
    wsum = {}
    for t in tensors:
        if t["type"] in SKIP_FAMILY or t["type"].startswith("TYPE_"):
            continue
        wsum[t["type"]] = wsum.get(t["type"], 0) + weights_of(t)
    for name in sorted(wsum):
        if wsum[name] > bw:
            best, bw = name, wsum[name]
    return best


def emit_quantize(tensors, path, base=None):
    b = base_type(tensors, base)
    if b is None:
        raise ToolError("cannot determine a base type (nothing quantizable); pass --base")
    lines = []
    for t in tensors:
        tn = t["type"]
        if tn in EMIT_SKIP or tn.startswith("TYPE_") or tn == b:
            continue
        lines.append("--tensor-type ^%s$=%s" % (re.escape(t["name"]), tn.lower()))
    with open(path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return b, len(lines)


def render(an, md=False):
    rows = []
    rows.append("ARCHITECTURE %s  FILE_TYPE %s  tensors %d"
                % (an["meta"].get("general.architecture", "?"), an["meta"].get("general.file_type", "?"),
                   len(an["tensors"])))
    rs = role_stats(an["tensors"])
    if md:
        rows.append("| role | tensors | weights | mean bpw | types |")
        rows.append("|---|---|---|---|---|")
    for r in sorted(rs):
        d = rs[r]
        hist = ", ".join("%s x%d" % (k, v) for k, v in sorted(d["hist"].items(), key=lambda kv: -kv[1]))
        rl = runlength(d["layers"])
        if md:
            rows.append("| %s | %d | %d | %s | %s |" % (r, d["n"], d["weights"],
                                                        d["bpw"] if d["bpw"] is not None else "-", hist))
        else:
            rows.append("ROLE %-22s tensors %4d weights %14d bpw %-7s %s"
                        % (r, d["n"], d["weights"], d["bpw"] if d["bpw"] is not None else "n/a", hist))
        if rl:
            rows.append("  %s: %s" % (r, rl))
    tot = totals(an["tensors"])
    for label, s in (("TOTAL", tot), ("EXPERTS(_exps)", tot["experts"]), ("REST", tot["rest"])):
        rows.append("%-15s n %6d weights %14d est_bytes %14d bpw %-7s excluded %d"
                    % (label, s["n"], s["weights"], s["est_bytes"], s["bpw"], s["excluded_n"]))
    return rows


def main(argv=None, opener=None):
    ap = argparse.ArgumentParser(prog="gguf_types.py")
    ap.add_argument("src", nargs="+")
    ap.add_argument("--emit-quantize", metavar="FILE")
    ap.add_argument("--base")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args(argv)
    try:
        an = analyze(a.src, opener)
        for ln in render(an, a.md):
            print(ln)
        if a.emit_quantize:
            b, n = emit_quantize(an["tensors"], a.emit_quantize, a.base)
            print("BASE %s -> %d override lines in %s" % (b, n, a.emit_quantize))
    except ToolError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return e.rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
