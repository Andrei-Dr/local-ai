#!/usr/bin/env python3
"""imx_heat.py — per-neuron FFN activation-energy concentration from an imatrix file (probe for the
"dense model driven like our MoE" idea: hot FFN neurons on GPU, cold on CPU needs energy concentrated
in few neurons). The imatrix row of blk.N.ffn_down.weight IS the squared intermediate activation per
neuron of layer N (llama-imatrix records mean-square per INPUT column of every matmul). Stdlib only;
follows tools/imatrix/imatrix.cpp save functions exactly.

usage: imx_heat.py FILE [--match ffn_down] [--json OUT.json]

Formats (detected by magic):
  GGUF   — metadata kv skipped by type; tensor data offsets are relative to the ALIGNED end of the
           header (general.alignment, default 32), NOT to byte 0; per matmul a tensor '<name>.in_sum2' (F32, ne0 = columns,
           ne1 = n_mat) and '<name>.counts' (F32, n_mat values); energy(column, mat) = in_sum2/counts;
           a dense entry has n_mat == 1; entries with several mats average the per-mat energies over
           mats with counts > 0 (chosen deterministically; ffn_down of a dense model is n_mat 1).
  legacy — int32 n_entries; per entry int32 len, name bytes, int32 ncall, int32 nval, then f32[nval]
           values that were stored as (value/count)*ncall per source; energy = value / ncall
           (ncall <= 0 is unusable -> exit 2). A trailing int32 + dataset string follow and are ignored.

Selection: name contains --match AND ends '.weight'; names containing '_exps' (fused MoE experts) are
skipped with a count. Output per layer: n neurons, share of total energy in the top 1/5/10/20/50 % of
neurons (desc sort, prefix sums, ceil fraction of n, floor 1), Gini coefficient, zero-energy count; a
summary line and the verdict word CONCENTRATED (median top-20%% >= 0.80) / MODERATE (>= 0.60) / FLAT.
Exit 2: unreadable file, unknown magic, no matching entries, counts <= 0."""
import argparse, json, math, re, struct, sys
from pathlib import Path

TOP_FRACS = (0.01, 0.05, 0.10, 0.20, 0.50)


class ToolError(Exception):
    pass


# ---------------------------------------------------------------- legacy .dat
def parse_legacy(data):
    try:
        (n_entries,) = struct.unpack_from("<i", data, 0)
    except struct.error:
        raise ToolError("unreadable legacy imatrix (short header)")
    off = 4
    entries = {}
    for ei in range(n_entries):
        try:
            (ln,) = struct.unpack_from("<i", data, off); off += 4
            name = data[off:off + ln].decode("utf-8"); off += ln
            (ncall,) = struct.unpack_from("<i", data, off); off += 4
            (nval,) = struct.unpack_from("<i", data, off); off += 4
        except (struct.error, UnicodeDecodeError):
            raise ToolError("unreadable legacy imatrix (entry %d)" % ei)
        if ncall <= 0:
            raise ToolError("counts <= 0 for %s (ncall %d)" % (name, ncall))
        raw = struct.unpack_from("<%df" % nval, data, off) if nval else ()
        off += 4 * nval
        entries[name] = [v / ncall for v in raw]
    return entries


# ---------------------------------------------------------------- GGUF
_VSZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8, 26: 2}
_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 10: "<Q", 11: "<q", 12: "<d", 26: "<e"}


class _RD:
    def __init__(self, buf):
        self.b, self.p = buf, 0

    def take(self, n):
        if self.p + n > len(self.b):
            raise ToolError("unreadable GGUF imatrix (truncated)")
        v = self.b[self.p:self.p + n]; self.p += n
        return v

    def u32(self):
        return struct.unpack_from("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack_from("<Q", self.take(8))[0]

    def s(self):
        return self.take(self.u64()).decode("utf-8", "replace")

    def skip1(self, t):
        if t == 8:
            self.s()
        elif t == 9:
            et, n = self.u32(), self.u64()
            if et in _VSZ:
                self.take(_VSZ[et] * n)
            else:
                for _ in range(n):
                    self.skip1(et)
        elif t in _VSZ:
            self.take(_VSZ[t])
        else:
            raise ToolError("unknown GGUF value type %d" % t)


def parse_gguf(data):
    r = _RD(data)
    if r.take(4) != b"GGUF":
        raise ToolError("bad magic")
    ver = r.u32()
    if ver not in (2, 3):
        raise ToolError("GGUF imatrix version %d unsupported" % ver)
    n_tensors = r.u64()
    n_kv = r.u64()
    alignment = 32
    for _ in range(n_kv):
        key = r.s()
        t = r.u32()
        if key == "general.alignment" and t == 4:
            alignment = struct.unpack_from("<I", r.take(4))[0]
        else:
            r.skip1(t)
    infos = []
    for _ in range(n_tensors):
        name = r.s()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        tid = r.u32()
        off = r.u64()
        infos.append((name, dims, tid, off))
    pad = (-r.p) % alignment                                 # tensor offsets are relative to the
    return data, infos, alignment, r.p + pad                 # ALIGNED end of the header, not to 0


def gguf_entries(data):
    payload, infos, _align, data_start = parse_gguf(data)
    by_name = {i[0]: i for i in infos}
    entries = {}
    for name, dims, tid, off in infos:
        if not name.endswith(".in_sum2"):
            continue
        base = name[:-len(".in_sum2")]
        cinfo = by_name.get(base + ".counts")
        if cinfo is None or tid != 0:
            continue
        ne0 = dims[0] if len(dims) > 0 else 0
        nmat = dims[1] if len(dims) > 1 else 1
        cvals = struct.unpack_from("<%df" % (cinfo[1][0] * (cinfo[1][1] if len(cinfo[1]) > 1 else 1)),
                                   payload, data_start + cinfo[3])
        sums = struct.unpack_from("<%df" % (ne0 * nmat), payload, data_start + off)
        if cvals and max(cvals) <= 0:
            raise ToolError("counts <= 0 for %s" % base)
        if nmat == 1:
            c = cvals[0] if cvals else 0
            if c <= 0:
                raise ToolError("counts <= 0 for %s" % base)
            entries[base] = [x / c for x in sums]
        else:
            per = [[] for _ in range(ne0)]
            for m in range(nmat):
                if m >= len(cvals) or cvals[m] <= 0:
                    continue
                for i in range(ne0):
                    per[i].append(sums[m * ne0 + i] / cvals[m])
            entries[base] = [sum(v) / len(v) if v else 0.0 for v in per]
    return entries


# ---------------------------------------------------------------- analysis
def select(entries, match):
    sel, skipped = {}, 0
    for name, vec in entries.items():
        if match not in name or not name.endswith(".weight"):
            continue
        if "_exps" in name:
            skipped += 1
            continue
        sel[name] = vec
    return sel, skipped


def stats_for(vec):
    n = len(vec)
    neg = [v for v in vec if v < 0]
    if neg:                                                  # in_sum2 is a sum of squares: < 0 is impossible
        raise ToolError("negative energy (%d values, min %.6g) — the file was misparsed, not a model property" % (len(neg), min(neg)))
    tot = sum(vec)
    if tot <= 0:
        raise ToolError("non-positive total energy (%.6g) — the file was misparsed" % tot)
    srt = sorted(vec, reverse=True)
    shares = []
    cum = 0.0
    ptr = 0
    for f in TOP_FRACS:
        k = max(1, math.ceil(f * n))
        while ptr < k:
            cum += srt[ptr]
            ptr += 1
        shares.append(cum / tot if tot else 0.0)
    mean = tot / n if n else 0.0
    asc = sorted(vec)
    gini = 0.0
    if mean > 0 and n:                                       # exact closed form on ascending order
        wsum = sum((i + 1) * x for i, x in enumerate(asc))
        gini = (2.0 * wsum) / (n * tot) - (n + 1.0) / n
    zeros = sum(1 for v in vec if v == 0.0)
    return {"n": n, "top1": shares[0], "top5": shares[1], "top10": shares[2], "top20": shares[3],
            "top50": shares[4], "gini": gini, "zeros": zeros}


def median(vals):
    s = sorted(vals)
    m = len(s)
    return s[m // 2] if m % 2 else (s[m // 2 - 1] + s[m // 2]) / 2.0


BLK_RE = re.compile(r"(?:^|\b)blk\.(\d+)\b")


def run(path, match="ffn_down", json_out=None):
    try:
        data = Path(path).read_bytes()
    except OSError as e:
        raise ToolError("cannot read %s: %s" % (path, e))
    if data[:4] == b"GGUF":
        entries = gguf_entries(data)
    else:
        try:
            (n_ent,) = struct.unpack_from("<i", data, 0)
            if 0 < n_ent < 100000:
                entries = parse_legacy(data)
            else:
                raise ToolError("unknown imatrix magic %r" % data[:4])
        except ToolError:
            raise
        except Exception:
            raise ToolError("unknown imatrix magic %r" % data[:4])
    sel, skipped = select(entries, match)
    print("entries matching '%s': %d | skipped fused-expert (_exps) entries: %d" % (match, len(sel), skipped))
    if not sel:
        raise ToolError("no matching entries")
    layers = {}
    for name, vec in sel.items():
        m = BLK_RE.search(name)
        lid = int(m.group(1)) if m else -1
        if lid in layers:
            raise ToolError("duplicate layer %d from %s" % (lid, name))
        layers[lid] = (name, vec, stats_for(vec))
    out_layers = {}
    for lid in sorted(layers):
        name, _vec, st = layers[lid]
        print("L%-4d n %6d top1%% %.3f top5%% %.3f top10%% %.3f top20%% %.3f top50%% %.3f gini %.3f zeros %d"
              % (lid, st["n"], st["top1"], st["top5"], st["top10"], st["top20"], st["top50"],
                 st["gini"], st["zeros"]))
        out_layers[str(lid)] = dict(st, name=name)
    med20 = median([s["top20"] for _, _, s in layers.values()])
    med5 = median([s["top5"] for _, _, s in layers.values()])
    gm = median([s["gini"] for _, _, s in layers.values()])
    verdict = "CONCENTRATED" if med20 >= 0.80 else "MODERATE" if med20 >= 0.60 else "FLAT"
    print("IMX_HEAT layers %d | top20%% share: min %.3f median %.3f max %.3f | top5%%: median %.3f | "
          "gini median %.3f | %s"
          % (len(layers),
             min(s["top20"] for _, _, s in layers.values()), med20,
             max(s["top20"] for _, _, s in layers.values()), med5, gm, verdict))
    if json_out:
        summary = {"layers_n": len(layers), "top20_min": min(s["top20"] for _, _, s in layers.values()),
                   "top20_median": med20, "top20_max": max(s["top20"] for _, _, s in layers.values()),
                   "top5_median": med5, "gini_median": gm, "verdict": verdict, "skipped_exps": skipped}
        Path(json_out).write_text(json.dumps({"layers": out_layers, "summary": summary}, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="imx_heat.py")
    ap.add_argument("file")
    ap.add_argument("--match", default="ffn_down")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    try:
        return run(a.file, a.match, a.json)
    except ToolError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
