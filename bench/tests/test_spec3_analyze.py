"""tests for bench/box/spec3_analyze.py — synthetic timed dumps with hand-computed H4 / H5 answers."""
import json, subprocess, sys, tempfile, unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "box" / "spec3_analyze.py"


def step(i, n, depth_ms, verify, gpu, step_ms, n_draft=None, replay=False):
    nd = n if n_draft is None else n_draft
    return {"step": i, "task": 1, "n_past": i, "replay": replay, "n_draft": nd, "n_accept": nd,
            "t": {"draft": depth_ms * nd + 0.5, "draft_depth": [depth_ms] * nd, "process": 2.0, "verify": verify,
                  "verify_gpu": gpu, "gpu_splits": 80, "step": step_ms}}


def write(d, depth_ms, verify_of, gpu_of, tps=(50.0, 51.0), s3=(52.0, 53.0)):
    """T n: step = 21 + 9.2 n ms, draft = depth_ms x n + 0.5, verify / GPU from the given functions of n"""
    for n in (1, 2, 3):
        for p, t in enumerate(tps, start=1):
            rows = [step(k, n, depth_ms, verify_of(n), gpu_of(n), 21 + 9.2 * n) for k in range(4)]
            rows.append(step(9, n, 99.0, 99.0, 0.0, 99.0, n_draft=n - 1) if n > 1 else step(9, n, 99.0, 99.0, 0.0, 99.0, replay=True))
            (d / f"spec3_T{n}_{p}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (d / f"spec3_T{n}_{p}.json").write_text(json.dumps([{"prompt": "code", "decode_tps": t, "draft_n": 1, "draft_n_accepted": 1}]))
    for p, t in enumerate(s3, start=1):
        (d / f"spec3_S3_{p}.json").write_text(json.dumps([{"prompt": "code", "decode_tps": t, "draft_n": 1, "draft_n_accepted": 1}]))


class Spec3Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self):
        p = subprocess.run([sys.executable, str(TOOL), str(self.d), "--json", str(self.d / "o.json")], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        return p.stdout, json.loads((self.d / "o.json").read_text())

    def test_m1_has_room(self):
        # draft 3.0 ms per depth: slope 3.0 of a 9.2 ms step slope = 32.6% -> H4 TRUE; n3 verify 32, GPU 19 -> idle 13 >= draft
        # 9.5 -> H5 TRUE. The short-draft and replayed steps (99 ms everywhere) must not count.
        write(self.d, 3.0, lambda n: 20 + 4 * n, lambda n: 10 + 3 * n)
        out, r = self.run_tool()
        self.assertAlmostEqual(r["marginal_step_ms"], 9.2, places=6)
        self.assertAlmostEqual(r["marginal_draft_ms"], 3.0, places=6)
        self.assertAlmostEqual(r["arms"]["T3"]["gpu_idle"], 13.0, places=6)
        self.assertAlmostEqual(r["arms"]["T3"]["draft"], 9.5, places=6)
        self.assertEqual(r["arms"]["T2"]["draft_depth"], [3.0, 3.0])
        self.assertTrue(r["h4"].startswith("TRUE"))
        self.assertTrue(r["h5"].startswith("TRUE"))
        self.assertAlmostEqual(r["dump_overhead_pct"], 100 * (50.5 / 52.5 - 1), places=6)

    def test_m1_dead_and_window_too_small(self):
        # draft 1.0 ms per depth: 10.9% < 15% -> H4 FALSE (M1 dead); idle 2 < draft 3.5 -> H5 FALSE (57%)
        write(self.d, 1.0, lambda n: 20 + 4 * n, lambda n: 30.0)
        out, r = self.run_tool()
        self.assertTrue(r["h4"].startswith("FALSE"))
        self.assertTrue(r["h5"].startswith("FALSE (M1 hides only part: 57%"))

    def test_between(self):
        # draft 2.0 ms per depth: 21.7% -> BETWEEN
        write(self.d, 2.0, lambda n: 20 + 4 * n, lambda n: 10.0)
        _, r = self.run_tool()
        self.assertTrue(r["h4"].startswith("BETWEEN"))


if __name__ == "__main__":
    unittest.main()
