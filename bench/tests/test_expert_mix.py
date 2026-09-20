"""tests for bench/box/expert_mix.py. The mix-mode tests build tiny GGUFs with gguf-py and SKIP cleanly
(unittest.skipUnless) where gguf/numpy are absent — they run on the box (/ai/.venv) and locally when the
packages are installed. The hot-mode trace tests are pure stdlib and never skip."""
import importlib.util, json, os, struct, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "box"))
HAVE = importlib.util.find_spec("gguf") is not None and importlib.util.find_spec("numpy") is not None
import expert_mix

HAS_QUANTS = False
if HAVE:
    import numpy as _np
    HAS_QUANTS = hasattr(importlib.import_module("gguf.quants"), "quantize")

NE = 32    # experts; 32 keeps one Q8_0 block exactly one channel-row (channel purity is testable)


@unittest.skipUnless(HAVE, "gguf-py / numpy not installed")
class MixCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gguf
        import numpy as np
        cls.np, cls.gguf = np, gguf
        cls.ch = lambda base, mul, e: base[e % 8] * mul[e % 8]
        cls.CONST_H = [10.0 + 10.0 * e for e in range(NE)]           # 10..320, exact in F16
        cls.CONST_L = [2.0 * (1 + e % 16) for e in range(NE)]        # 2..32, exact in F16
        cls.d = tempfile.mkdtemp()
        cls.hi = os.path.join(cls.d, "hi.gguf")
        cls.lo = os.path.join(cls.d, "lo.gguf")
        cls.hi_badshape = os.path.join(cls.d, "hi_bad.gguf")
        emb = cls.np.arange(9, dtype=cls.np.float32).reshape(3, 3)
        cls._mk(cls.hi, True, emb)
        cls._mk(cls.lo, False, emb)
        cls._mk(cls.hi_badshape, True, cls.np.arange(6, dtype=cls.np.float32).reshape(2, 3))

    @classmethod
    def _mk(cls, path, is_hi, emb):
        g = cls.gguf
        consts = cls.CONST_H if is_hi else cls.CONST_L
        w = g.GGUFWriter(path, "qwen35moe")
        w.add_uint32("general.file_type", 15)
        w.add_uint32("qwen35moe.block_count", 2)
        w.add_tensor("token_embd.weight", emb)
        for blk in (0, 1):
            arr = cls.np.zeros((8, NE), cls.np.float32)              # expert axis = LAST
            for e in range(NE):
                arr[:, e] = consts[e]
            name = "blk.%d.ffn_gate_exps.weight" % blk
            w.add_tensor(name, arr.astype(cls.np.float16) if not is_hi else arr)
        w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

    def _channels(self, path, blk=0):
        rd = self.gguf.GGUFReader(path)
        t = [t for t in rd.tensors if t.name == "blk.%d.ffn_gate_exps.weight" % blk][0]
        a = self.np.asarray(self.gguf.quants.dequantize(t.data, t.tensor_type), dtype=self.np.float32)
        return [float(a[(Ellipsis, e)].mean()) for e in range(NE)]

    def _hot(self, mapping):
        p = os.path.join(self.d, "hot.json")
        json.dump(mapping, open(p, "w"))
        return p

    def _run(self, hot, out, container="F16", extra=(), hi=None, err=None):
        argv = ["--hi", hi or self.hi, "--lo", self.lo, "--hot", hot, "--out", out,
                "--container", container] + list(extra)
        return expert_mix.main(argv, err=err) if err else expert_mix.main(argv)

    def test_hot_channel_values_and_metadata(self):
        out = os.path.join(self.d, "mix1.gguf")
        self.assertEqual(self._run(self._hot({"0": [2]}), out), 0)
        m0 = self._channels(out)
        want = [self.CONST_L[e] if e != 2 else self.CONST_H[e] for e in range(NE)]
        self.assertEqual([round(x, 3) for x in m0], [round(x, 3) for x in want])
        self.assertEqual([round(x) for x in self._channels(out, blk=1)],
                         [round(x) for x in self.CONST_L])           # layer absent from HOT.json
        rd = self.gguf.GGUFReader(out)
        for k, v in (("expert_mix.hi_file", "hi.gguf"), ("expert_mix.lo_file", "lo.gguf"),
                     ("expert_mix.container", "F16")):
            self.assertEqual(rd.fields[k].contents(), v)
        self.assertIsInstance(rd.fields["expert_mix.hot_fraction"].contents(), float)
        rlo, rout = self.gguf.GGUFReader(self.lo), self.gguf.GGUFReader(out)
        tlo = [t for t in rlo.tensors if t.name == "token_embd.weight"][0]
        tout = [t for t in rout.tensors if t.name == "token_embd.weight"][0]
        self.assertTrue(self.np.array_equal(tlo.data, tout.data))    # non-expert bytes preserved

    def test_q8_container_rounding(self):
        out = os.path.join(self.d, "mix_q8.gguf")
        self.assertEqual(self._run(self._hot({"0": [0, 1]}), out, container="Q8_0"), 0)
        m = self._channels(out)
        want = [self.CONST_H[e] if e in (0, 1) else self.CONST_L[e] for e in range(NE)]
        for got, wnt in zip(m, want):
            self.assertLessEqual(abs(got - wnt), 1.3 + 0.01 * wnt)   # Q8_0 block-scale slack

    def test_errors_exit_2(self):
        import io
        buf = io.StringIO()
        self.assertEqual(self._run(self._hot({"0": [NE]}), os.path.join(self.d, "e1.gguf"), err=buf), 2)
        self.assertIn("ERROR", buf.getvalue())
        buf = io.StringIO()
        self.assertEqual(self._run(self._hot({"0": [0]}), os.path.join(self.d, "e2.gguf"),
                                   container="NOTATYPE", err=buf), 2)
        self.assertIn("container", buf.getvalue())
        buf = io.StringIO()
        self.assertEqual(self._run(self._hot({"0": [0]}), os.path.join(self.d, "e3.gguf"),
                                   hi=self.hi_badshape, err=buf), 2)
        self.assertIn("mismatch", buf.getvalue())

    def test_dry_run_writes_nothing(self):
        out = os.path.join(self.d, "never.gguf")
        self.assertEqual(self._run(self._hot({"0": [3]}), out, extra=["--dry-run"]), 0)
        self.assertFalse(os.path.exists(out))

    def test_layers_range_excludes_outside(self):
        out = os.path.join(self.d, "mix_layers.gguf")
        self.assertEqual(self._run(self._hot({"0": [0], "1": [1]}), out, extra=["--layers", "1-1"]), 0)
        self.assertEqual([round(x) for x in self._channels(out, blk=0)], [round(x) for x in self.CONST_L])
        m1 = self._channels(out, blk=1)
        self.assertAlmostEqual(m1[1], self.CONST_H[1], delta=0.01)


class HotModeCase(unittest.TestCase):
    def _trace(self, records, path):
        b = b""
        for layer, ids in records:
            n = len(ids) // 2
            b += struct.pack("<iii", layer, n, 2) + struct.pack("<%di" % len(ids), *ids)
        Path(path).write_bytes(b)
        return path

    def test_top_frac_ties_take_lower_id(self):
        d = tempfile.mkdtemp()
        t = self._trace([(0, [0, 0, 0, 1, 1, 1, 2, 2])], d + "/t.bin")   # e0:3 e1:3 e2:2? recount below
        out = d + "/hot.json"
        rc = expert_mix.main(["hot", "--trace", t, "--frac", "0.5", "--out", out])
        self.assertEqual(rc, 0)
        hs = json.load(open(out))
        self.assertEqual(hs["0"], [0, 1])                # n_expert=3 -> want 2, tie 3-3 resolved low-id

    def test_two_traces_combined(self):
        d = tempfile.mkdtemp()
        a = self._trace([(0, [3, 3, 1, 1])], d + "/a.bin")
        b = self._trace([(0, [1, 1, 1, 2, 0, 0])], d + "/b.bin")         # counts e1=5 e3=2 e0=2 e2=1
        out = d + "/hot.json"
        rc = expert_mix.main(["hot", "--trace", a, "--trace", b, "--frac", "0.25", "--out", out])
        self.assertEqual(rc, 0)
        self.assertEqual(json.load(open(out))["0"], [1])

    def test_truncated_trace_exits_2(self):
        d = tempfile.mkdtemp()
        t = Path(d + "/bad.bin")
        t.write_bytes(struct.pack("<iii", 0, 5, 2) + struct.pack("<4i", 0, 1, 2, 3))
        buf = __import__("io").StringIO()
        self.assertEqual(expert_mix.main(["hot", "--trace", str(t), "--frac", "0.5",
                                          "--out", d + "/o.json"], err=buf), 2)


if __name__ == "__main__":
    unittest.main()
