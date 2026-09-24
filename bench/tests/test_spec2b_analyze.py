"""tests for bench/box/spec2b_analyze.py — synthetic spec2b arms (client ids + dumps) with known D1 / D2 numbers."""
import json, subprocess, sys, tempfile, unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "box" / "spec2b_analyze.py"
L = 10     # prompt length
N = 13     # generated tokens: g 0..12; verify steps emit 3 tokens each (n_accept 2) from g = 1


BASE = {"code": 100, "reason": 300, "prose": 500}  # distinct texts per prompt


def gen(prompt, flip_at=None):
    b = BASE[prompt]
    return [b + g if flip_at is None or g < flip_at else b + 1000 + g for g in range(N)]


def arm(ids, eps, margins):
    """client rows + dump lines for one arm: target top-1 logit 1 + eps, top-2 logit 1 - margin(g)"""
    client = [{"prompt": p, "tokens": ids[p]} for p in ids]
    lines, step = [], 0
    for task, p in enumerate(ids, start=1):
        g0 = 1
        while g0 + 2 < N:
            tgt = ids[p][g0:g0 + 3]
            target = [{"lse": 0, "lse_t": None,
                       "top": [[ids[p][g], 0.5, 1.0 + eps], [999, 0.2, 1.0 - margins.get((p, g), 1.0)]]} for g in range(g0, g0 + 3)]
            target.append({"lse": 0, "lse_t": None, "top": [[5, 0.5, 9.0], [6, 0.1, 0.0]]})  # i = 3 > n_accept: ignored
            lines.append({"step": step, "seq": 0, "task": task, "n_past": L + g0 - 1, "replay": False, "n_draft": 3,
                          "n_accept": 2, "draft_tok": tgt + [7], "tgt_tok": tgt, "draft": [], "target": target})
            step += 1
            g0 += 3
    return client, lines


def write(d, arms):
    for name, (client, lines) in arms.items():
        (d / f"spec2b_{name}.json").write_text(json.dumps(client))
        (d / f"spec2b_{name}.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))


class Spec2bCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self):
        p = subprocess.run([sys.executable, str(TOOL), str(self.d), "--json", str(self.d / "v.json")], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        return p.stdout, json.loads((self.d / "v.json").read_text())

    def test_batch_shape_class(self):
        # S2 vs S3 flip on "reason" at g 5 (margin 0.01), |delta| 0.02 before it; T4 vs T3 flip on "prose" at g 7 (margin
        # 0.015), |delta| 0.03 <= 2 x 0.02 -> D1 has a flip and (b) <= 2 x (a): batch-shape class
        same = {k: gen(k) for k in BASE}
        m = {("reason", 5): 0.01, ("prose", 7): 0.015}
        write(self.d, {"S1": arm(same, 0.0, m), "S2": arm(same, 0.0, m),
                       "S3": arm({**same, "reason": gen("reason", 5)}, 0.02, m),
                       "T3": arm(same, 0.0, m), "T4": arm({**same, "prose": gen("prose", 7)}, 0.03, m)})
        out, v = self.run_tool()
        self.assertAlmostEqual(v["max_delta_a"], 0.02, places=6)
        self.assertAlmostEqual(v["max_delta_b"], 0.03, places=6)
        self.assertEqual(v["d1"]["S2-S3"]["reason"]["first_div"], 5)
        # flip margin = the larger run's margin: S3's top-1 sits 0.02 higher -> 0.01 + 0.02
        self.assertAlmostEqual(v["d1"]["S2-S3"]["reason"]["flip_margin"], 0.03, places=6)
        self.assertEqual(v["b"]["prose"]["first_div"], 7)
        self.assertEqual(v["verdict"], "BATCH-SHAPE CLASS")
        self.assertIn("SPEC2B_VERDICT BATCH-SHAPE CLASS", out)

    def test_suspect(self):
        # no D1 flip, (a) |delta| 0.001; T4 flips at margin 0.5 with |delta| 0.3 -> both conditions fail: SUSPECT
        same = {k: gen(k) for k in BASE}
        m = {("reason", 4): 0.5}
        write(self.d, {"S1": arm(same, 0.0, m), "S2": arm(same, 0.0, m), "S3": arm(same, 0.001, m),
                       "T3": arm(same, 0.0, m), "T4": arm({**same, "reason": gen("reason", 4)}, 0.3, m)})
        out, v = self.run_tool()
        self.assertEqual(v["d1_flips"], 0)
        self.assertAlmostEqual(v["margin_threshold"], 0.001, places=6)
        self.assertFalse(v["cond1"])
        self.assertFalse(v["cond2"])
        self.assertEqual(v["verdict"], "SUSPECT")
        self.assertEqual(v["det"]["code"]["first_div"], None)

    def test_delta_rule_alone_fails(self):
        # D1 flips (cond1 true) but (b) delta 0.05 > 2 x (a) 0.02 -> SUSPECT
        same = {k: gen(k) for k in BASE}
        m = {("reason", 5): 0.01}
        write(self.d, {"S1": arm(same, 0.0, m), "S2": arm(same, 0.0, m), "S3": arm({**same, "reason": gen("reason", 5)}, 0.02, m),
                       "T3": arm(same, 0.0, m), "T4": arm(same, 0.05, m)})
        _, v = self.run_tool()
        self.assertTrue(v["cond1"])
        self.assertFalse(v["cond2"])
        self.assertEqual(v["verdict"], "SUSPECT")


if __name__ == "__main__":
    unittest.main()
