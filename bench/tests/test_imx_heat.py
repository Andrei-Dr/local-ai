"""tests for bench/imx_heat.py — tiny legacy .dat files packed with struct and a test-side minimal GGUF
writer (F32 tensors, alignment 32) built to the format in tools/imatrix/imatrix.cpp; the two paths must
produce identical numbers. Hand-computed vectors; Gini cross-checked with the O(n^2) definition."""
import io, json, math, struct, tempfile, unittest
from contextlib import redirect_stdout
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import imx_heat

VEC_E = [8.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]              # hand vector, sum 10
NCALL = 3


def gini_def(v):
    n = len(v)
    m = sum(v) / n
    return sum(abs(a - b) for a in v for b in v) / (2.0 * n * n * m)


def legacy_bytes(entries, ncall=NCALL):
    b = struct.pack("<i", len(entries))
    for name, vec in entries:
        raw = b"" if ncall == 0 else struct.pack("<%df" % len(vec), *[x * ncall for x in vec])
        if ncall == 0:
            raw = struct.pack("<%df" % len(vec), *vec)
        b += struct.pack("<i", len(name)) + name.encode() + struct.pack("<ii", ncall, len(vec)) + raw
    return b + struct.pack("<i", 0) + struct.pack("<i", 0)


def gguf_bytes(entries, counts_val=100.0):
    infos, datas = [], []
    off = 0

    def pad(x):
        return (x + 31) // 32 * 32

    kv = b""
    key = "general.type"
    kv += struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", 8) + \
        struct.pack("<Q", 7) + b"imatrix"
    for name, vec in entries:
        for suffix, dims, blob in (
                (".in_sum2", [len(vec), 1], struct.pack("<%df" % len(vec), *[x * counts_val for x in vec])),
                (".counts", [1, 1], struct.pack("<f", counts_val))):
            tname = name + suffix
            datas.append(blob)
            infos.append((tname, dims, blob))
    kv_head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(infos)) + struct.pack("<Q", 1) + kv
    infob = b""
    for name, dims, _blob in infos:
        infob += struct.pack("<Q", len(name)) + name.encode() + struct.pack("<I", len(dims)) + \
            b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<I", 0) + struct.pack("<Q", 0)
    data_start = (len(kv_head) + len(infob) + 31) // 32 * 32
    offs, cur = [], data_start
    for _n, _d, blob in infos:
        offs.append(cur)
        cur = (cur + len(blob) + 31) // 32 * 32
    header = kv_head
    for i, (name, dims, _blob) in enumerate(infos):
        header += struct.pack("<Q", len(name)) + name.encode() + struct.pack("<I", len(dims)) + \
            b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<I", 0) + struct.pack("<Q", offs[i])
    data_region = b""
    for i, (_n, _d, blob) in enumerate(infos):
        pos = data_start + len(data_region)
        assert pos == offs[i], (pos, offs[i])
        data_region += blob
        nxt = (pos + len(blob) + 31) // 32 * 32              # keep tensors alignment-padded
        data_region += b"\0" * (nxt - (pos + len(blob)))
    return header.ljust(data_start, b"\0") + data_region


def run_capture(path, *args):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = imx_heat.main([path] + list(args))
    return rc, buf.getvalue()


class HeatCase(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def wp(self, name, data):
        p = Path(self.d) / name
        p.write_bytes(data)
        return str(p)

    def test_hand_vector_shares_gini(self):
        st = imx_heat.stats_for(VEC_E)
        self.assertEqual(st["n"], 8)
        self.assertAlmostEqual(st["top1"], 0.8, 6)
        self.assertAlmostEqual(st["top5"], 0.8, 6)
        self.assertAlmostEqual(st["top10"], 0.8, 6)
        self.assertAlmostEqual(st["top20"], 0.9, 6)
        self.assertAlmostEqual(st["top50"], 1.0, 6)             # top 4 = 8+1+1+0 = all of it
        self.assertAlmostEqual(st["gini"], gini_def(VEC_E), 9)
        self.assertAlmostEqual(gini_def(VEC_E), 0.8, 6)         # 128/160: (8-1)x4 + (8-0)x10 + (1-0)x20
        self.assertEqual(st["zeros"], 5)

    def test_legacy_reads_divides_by_ncall_and_skips_exps(self):
        p = self.wp("m.dat", legacy_bytes([("blk.0.attn_q.weight", [1.0, 2.0]),
                                           ("blk.0.ffn_down.weight", VEC_E),
                                           ("blk.1.ffn_down_exps.weight", [1.0, 1.0])]))
        rc, out = run_capture(p)
        self.assertEqual(rc, 0)
        self.assertIn("skipped fused-expert (_exps) entries: 1", out)
        self.assertIn("gini 0.800", out)
        self.assertEqual(out.count("L0"), 1)                     # only the ffn_down row

    def test_gguf_matches_legacy_exactly(self):
        ents = [("blk.0.ffn_down.weight", VEC_E), ("blk.0.attn_q.weight", [1.0, 2.0]),
                ("blk.1.ffn_down_exps.weight", [1.0, 1.0])]
        lp = self.wp("m.dat", legacy_bytes(ents))
        gp = self.wp("m.gguf", gguf_bytes(ents))
        lj = str(Path(self.d) / "l.json")
        gj = str(Path(self.d) / "g.json")
        rc1, o1 = run_capture(lp, "--json", lj)
        rc2, o2 = run_capture(gp, "--json", gj)
        self.assertEqual((rc1, rc2), (0, 0))
        self.assertEqual(o1, o2)                                  # identical stdout
        self.assertEqual(json.load(open(lj)), json.load(open(gj)))  # identical json

    def test_verdict_words(self):
        cases = {"CONCENTRATED": [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 "MODERATE": [3.0, 3.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                 "FLAT": [1.0] * 8}
        for word, vec in cases.items():
            p = self.wp("v_%s.gguf" % word, gguf_bytes([("blk.0.ffn_down.weight", vec)]))
            _rc, out = run_capture(p)
            self.assertIn(word, out, (word, out))

    def test_exit_2_cases(self):
        rc, _o = run_capture(str(Path(self.d) / "nofile.dat"))
        self.assertEqual(rc, 2)
        bad = self.wp("bad.dat", b"JUNKjunkjunk")
        self.assertEqual(run_capture(bad)[0], 2)
        attn_only = self.wp("attn.dat", legacy_bytes([("blk.0.attn_q.weight", [1.0])]))
        self.assertEqual(run_capture(attn_only)[0], 2)            # no matching entries
        ncall0 = self.wp("z.dat", legacy_bytes([("blk.0.ffn_down.weight", VEC_E)], ncall=0))
        self.assertEqual(run_capture(ncall0)[0], 2)               # counts <= 0

    def test_json_shape(self):
        p = self.wp("j.gguf", gguf_bytes([("blk.7.ffn_down.weight", VEC_E)]))
        jf = str(Path(self.d) / "x.json")
        run_capture(p, "--json", jf)
        d = json.load(open(jf))
        self.assertIn("7", d["layers"])
        self.assertEqual(d["layers"]["7"]["zeros"], 5)
        self.assertIn("verdict", d["summary"])


if __name__ == "__main__":
    unittest.main()
