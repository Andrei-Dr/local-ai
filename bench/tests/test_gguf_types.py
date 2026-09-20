"""tests for bench/gguf_types.py — synthetic GGUFs built with struct in-process (no gguf-py, nothing
copied from the graft tool), local files for parse/stats/emit cases, and a scripted Range opener for the
URL cases (growth 8 -> 16 MiB, cap => exit 2). Numbers hand-computed; type-table lookups mirrored here
literally, not imported, so a table edit in the tool flips these red on purpose."""
import os, struct, subprocess, sys, tempfile, unittest
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "bench"))
import gguf_types as gt

MIB = 1 << 20


def es(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


# canonical GGUF value-type ids (0 u8 .. 8 str, 9 arr, 10 u64, 12 f64)
_PF = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 10: "Q", 11: "q", 12: "d"}


def ev(t, v):
    if t == 8:
        return es(v)
    if t == 7:
        return struct.pack("<B", 1 if v else 0)
    if t == 9:
        et, items = v
        out = struct.pack("<IQ", et, len(items))
        if et in _PF:                                      # one-shot pack keeps megabyte arrays fast
            out += struct.pack("<%d%s" % (len(items), _PF[et]), *items)
        elif et == 8:
            out += b"".join(es(x) for x in items)
        elif et == 9:                                      # nested arrays are arrays OF STRINGS here
            for it in items:
                out += struct.pack("<IQ", 8, len(it)) + b"".join(es(x) for x in it)
        else:
            raise AssertionError("builder: unsupported array elem %d" % et)
        return out
    return struct.pack(_PF[t], v)


def build(kvs, infos, version=3):
    b = b"GGUF" + struct.pack("<IQQ", version, len(infos), len(kvs))
    for k, t, v in kvs:
        b += es(k) + struct.pack("<I", t) + ev(t, v)
    for name, dims, tid, off in infos:
        b += es(name) + struct.pack("<I", len(dims))
        b += struct.pack("<%d%s" % (len(dims), "I" if version == 2 else "Q"), *dims)
        b += struct.pack("<IQ", tid, off)
    return b


def tmpfile(data, name="m.gguf"):
    d = tempfile.mkdtemp()
    p = Path(d) / name
    p.write_bytes(data)
    return str(p)


PLAIN_INFOS = [("token_embd.weight", [4096, 4096], 1, 0),
               ("blk.0.attn_q.weight", [4096, 4096], 2, 0),
               ("blk.0.ffn_gate_exps.weight", [4096, 4096, 8], 10, 0)]


class T(unittest.TestCase):
    def test_local_parse_counts_and_meta(self):
        path = tmpfile(build([("general.architecture", 8, "qwen35moe"), ("general.file_type", 4, 15),
                              ("qwen35moe.block_count", 4, 40)], PLAIN_INFOS))
        an = gt.analyze([path])
        self.assertEqual([t["name"] for t in an["tensors"]], [n for n, _, _, _ in PLAIN_INFOS])
        self.assertEqual(an["meta"]["general.architecture"], "qwen35moe")
        self.assertEqual(an["meta"]["general.file_type"], 15)
        self.assertEqual(gt.tname(10), "Q2_K")

    def test_kv_value_types_skipped(self):
        kvs = [("u8", 0, 7), ("i8", 1, -1), ("u16", 2, 9), ("i16", 3, -2), ("u32", 4, 11),
               ("i32", 5, -3), ("f32", 6, 1.5), ("bool", 7, True), ("str", 8, "hello"),
               ("u64", 10, 12), ("i64", 11, -13), ("f64", 12, 2.5),
               ("arr_u32", 9, (4, [1, 2, 3])), ("arr_str", 9, (8, ["a", "bc", "d"])),
               ("arr_arr_str", 9, (9, [["x", "y"], ["z"]])),
               ("general.architecture", 8, "qwen35moe"), ("final", 4, 99)]
        path = tmpfile(build(kvs, PLAIN_INFOS))
        an = gt.analyze([path])
        self.assertEqual(len(an["tensors"]), 3)
        self.assertEqual(an["meta"]["general.architecture"], "qwen35moe")

    def test_v2_dims_parsed(self):
        path = tmpfile(build([("general.file_type", 4, 2)],
                             [("token_embd.weight", [8, 8], 0, 0)], version=2))
        an = gt.analyze([path])
        self.assertEqual(an["tensors"][0]["dims"], [8, 8])
        self.assertEqual(an["tensors"][0]["type"], "F32")

    def test_unknown_type_never_crashes_and_is_excluded(self):
        path = tmpfile(build([], PLAIN_INFOS + [("blk.0.mystery", [16, 8], 31, 0)]))
        an = gt.analyze([path])
        self.assertEqual(an["tensors"][3]["type"], "TYPE_31")
        tot = gt.totals(an["tensors"])
        self.assertEqual(tot["excluded_n"], 1)
        # blended bpw over MEASURABLE weights only: (16777216*16 + 16777216*4.5 + 134217728*2.625)/... 
        self.assertEqual(tot["bpw"], 4.15)
        self.assertEqual(tot["est_bytes"], 33554432 + 9437184 + 44040192)

    def test_role_strip_and_runlength(self):
        infos = [("blk.%d.ffn_gate_exps.weight" % i, [8, 8], tid, 0)
                 for i, tid in enumerate([12, 12, 10, 10, 10, 14])]
        rs = gt.role_stats([{"name": n, "dims": d, "type": gt.tname(t), "offset": 0}
                            for n, d, t, _ in infos])
        self.assertEqual(list(rs), ["ffn_gate_exps.weight"])
        rl = gt.runlength(rs["ffn_gate_exps.weight"]["layers"])
        self.assertEqual(rl, "L0-1 Q4_K, L2-4 Q2_K, L5 Q6_K")

    def test_bpw_totals_hand_computed(self):
        tensors = [{"name": "blk.0.a", "dims": [1024, 1024], "type": "Q4_0", "offset": 0},
                   {"name": "b", "dims": [512, 512], "type": "F16", "offset": 0}]
        tot = gt.totals(tensors)
        self.assertEqual(tot["weights"], 1310720)
        self.assertEqual(tot["est_bytes"], 1114112)        # 1048576*4.5/8 + 262144*16/8
        self.assertEqual(tot["bpw"], 6.8)

    def test_experts_vs_rest_split(self):
        tensors = [{"name": "blk.0.ffn_gate_exps.weight", "dims": [8, 8], "type": "Q2_K", "offset": 0},
                   {"name": "blk.0.attn_q.weight", "dims": [8, 8], "type": "Q4_0", "offset": 0}]
        tot = gt.totals(tensors)
        self.assertEqual((tot["experts"]["weights"], tot["experts"]["bpw"]), (64, 2.62))
        self.assertEqual((tot["rest"]["weights"], tot["rest"]["bpw"]), (64, 4.5))

    def test_emit_quantize_base_by_weights_and_escaping(self):
        big = [[12288, 32000], 10]
        tensors = [{"name": "big_experts.weight", "dims": big[0], "type": "Q2_K", "offset": 0}]
        tensors += [{"name": "blk.0.attn_q.weight", "dims": [64, 64], "type": "Q8_0", "offset": 0},
                    {"name": "blk.1.attn_q.weight", "dims": [64, 64], "type": "Q8_0", "offset": 0},
                    {"name": "token_embd.weight", "dims": [64, 64], "type": "F32", "offset": 0}]
        outf = str(Path(tempfile.mkdtemp()) / "qt.txt")
        b, n = gt.emit_quantize(tensors, outf)
        self.assertEqual((b, n), ("Q2_K", 2))               # weights beat the 2-vs-1 tensor COUNT
        lines = open(outf).read().splitlines()
        self.assertEqual(lines[0], "--tensor-type ^blk\\.0\\.attn_q\\.weight$=q8_0")
        self.assertNotIn("token_embd", open(outf).read())   # F32 skipped
        self.assertNotIn("big_experts", open(outf).read())  # equals base

    def test_shard_merge_and_duplicate_error(self):
        a = tmpfile(build([], [("x.w", [4, 4], 2, 0)]), "a.gguf")
        b = tmpfile(build([], [("y.w", [4, 4], 12, 0)]), "b.gguf")
        an = gt.analyze([a, b])
        self.assertEqual([t["name"] for t in an["tensors"]], ["x.w", "y.w"])
        c = tmpfile(build([], [("x.w", [4, 4], 10, 0)]), "c.gguf")
        self.assertEqual(gt.main([a, c]), 2)

    def test_missing_file_exits_2(self):
        self.assertEqual(gt.main(["/nonexistent/x.gguf"]), 2)

    def test_url_grows_ranges(self):
        data = build([("big", 9, (4, [7] * 2_100_000))], PLAIN_INFOS)   # header alone > 8 MiB
        reqs = []

        def opener(url, lo, hi):
            reqs.append((lo, hi))
            return data[lo:hi + 1]

        an = gt.analyze(["http://example/mod.gguf"], opener=opener)
        self.assertEqual(len(an["tensors"]), 3)
        self.assertEqual(reqs[0], (0, 8 * MIB - 1))
        self.assertEqual(reqs[1], (0, 16 * MIB - 1))                      # the doubled window
        self.assertEqual(len(reqs), 2)

    def test_url_cap_exits_2(self):
        # a header advertising a 40-billion-element u32 array: no window up to the cap can ever
        # contain it, so the decoder must double 8->512 MiB and then bail with exit 2
        hdr = b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + es("big") + struct.pack("<II", 9, 4) + \
            struct.pack("<Q", 40_000_000_000)

        def opener(url, lo, hi):
            reqs.append((lo, hi))
            blob = hdr.ljust(hi + 1, b"\0")[:hi + 1]
            return blob

        reqs = []
        rc = gt.main(["http://example/inf.gguf"], opener=opener)
        self.assertEqual(rc, 2)
        self.assertTrue(reqs and max(hi for _, hi in reqs) <= 512 * MIB - 1)
        self.assertIn(512 * MIB - 1, [hi for _, hi in reqs])              # capped fetch happened

    def test_cli_report_prints_roles_and_totals(self):
        path = tmpfile(build([("general.architecture", 8, "qwen35moe")], PLAIN_INFOS))
        self.assertEqual(gt.main([path]), 0)


if __name__ == "__main__":
    unittest.main()
