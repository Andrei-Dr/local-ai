"""tests for bench/box/spec1_sim.py — a two-step synthetic dump whose policy scores are worked out by hand."""
import json, struct, subprocess, sys, tempfile, unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "box" / "spec1_sim.py"


def top(ids, p):
    """10 entries whose top-10 softmax gives the top-1 exactly p (the server's p_min quantity): logit 0 for the top-1,
    log((1 - p) / (9 p)) for the other nine; the stored full-vocab p is deliberately different (0.99) and must be ignored"""
    ids = list(ids) + [900 + j for j in range(10 - len(ids))]
    import math
    lo = math.log((1 - p) / (9 * p))
    return [[t, 0.99, 0.0 if j == 0 else lo] for j, t in enumerate(ids)]


def step(i, task, n_accept, p1, d1, tgt):
    return {"step": i, "seq": 0, "task": task, "n_past": i, "replay": False, "n_draft": 3, "n_accept": n_accept,
            "draft_tok": [d1[0], 6, 7], "tgt_tok": tgt, "sampler": {"temp": 0}, "p_min": 0,
            "draft": [{"tok": d1[0], "kept": True, "lse": 0, "lse_t": None, "top": top(d1, p1[0])},
                      {"tok": 6, "kept": True, "lse": 0, "lse_t": None, "top": top([6, 40], p1[1])},
                      {"tok": 7, "kept": True, "lse": 0, "lse_t": None, "top": top([7, 41], p1[2])}],
            "target": []}


# one MoE layer, k = 1. Step X (task 1): every depth accepted, p1 0.9, four fresh experts -> NEW n = 0..3 = 1, 2, 3, 4.
# Step Y (task 2): depth 1 rejected (p1 0.3), the target's token 11 is the draft's rank 2; experts a, a, b, b -> NEW 1, 1, 2, 2.
STEPS = [step(0, 1, 3, (0.9, 0.9, 0.9), (5, 20, 21, 22), [5, 6, 7, 99]),
         step(1, 2, 0, (0.3, 0.9, 0.9), (5, 11, 12, 13), [11])]
ROUTE = {0: [100, 101, 102, 103], 1: [200, 200, 201, 201]}


def write_dump(d, name):
    (d / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in STEPS))
    b = b""
    for s, ids in ROUTE.items():
        b += struct.pack("<4i", 0x30525053, s, 1, 1) + struct.pack("<2i", 0, 4) + struct.pack("<4i", *ids)
    (d / f"{name}.route").write_bytes(b)


def write_sib(d):
    # path 0..7 with fresh experts; one alternative at p = 4 of target rank 2 with a fresh expert -> sibling cost 1 unit
    rec = lambda kind, pos, tok, rank, e: struct.pack("<4if2i", kind, pos, tok, rank, 0.1, 1, 1) + struct.pack("<2i", 0, e)
    b = struct.pack("<2i", 0x30424953, 1) + rec(1, 4, 90, 1, 500)
    for p in range(8):
        b += rec(2 if p == 4 else 0, p, 10 + p, 0 if p == 4 else -1, 300 + p)
    (d / "s.bin").write_bytes(b)


class Spec1SimCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        write_dump(self.d, "t0")
        write_dump(self.d, "card")
        write_sib(self.d)

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *extra):
        d = self.d
        p = subprocess.run([sys.executable, str(TOOL), "--t0", str(d / "t0.jsonl"), "--t0-route", str(d / "t0.route"),
                            "--card", str(d / "card.jsonl"), "--card-route", str(d / "card.route"), "--sib", str(d / "s.bin"),
                            "--json", str(d / "out.json"), *extra], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        return p.stdout, json.loads((d / "out.json").read_text())

    def test_hand_scored_policies(self):
        # t_fixed 10, t_draft 1, c 1:
        #   chain 3: X 10+3+4 = 17 ms, 4 tok | Y 10+3+2 = 15 ms, 1 tok       -> 5 / 32 ms = 156.25 t/s
        #   chain 1: X 13, 2 | Y 12, 1 -> 120.00         confidence stop (0.30 < theta <= 0.90): X 17, 4 | Y n = 0: 10+1+1, 1
        #   -> 5 / 29 = 172.41 (grid picks the first best, 0.35)   L3 rank 2 always: X 18, 4 | Y 16, 2 (sibling 11 hit) -> 176.47
        #   L4 k 2 (0.30 < theta <= 0.90): X 17, 4 | Y depth 1 + sibling: 10+1+(1+1), 2 -> 6 / 30 = 200.00
        out, res = self.run_tool("--t-fixed", "10", "--t-draft", "1", "--c", "1")
        self.assertIn("UNCALIBRATED", out)
        self.assertAlmostEqual(res["baseline"]["T0"], 156.25, places=2)
        pol = res["policies"]
        self.assertAlmostEqual(pol["chain n=1"]["T0"]["tps"], 120.0, places=2)
        self.assertAlmostEqual(pol["confidence stop"]["card"]["tps"], 5000 / 29, places=2)
        self.assertEqual(pol["confidence stop"]["card"]["grid"], "theta 0.35")
        self.assertAlmostEqual(pol["L3 rank 2 always"]["T0"]["tps"], 6000 / 34, places=2)
        self.assertAlmostEqual(pol["L4"]["T0"]["tps"], 200.0, places=2)
        self.assertEqual(pol["L4"]["T0"]["grid"], "theta 0.35 k 2")
        self.assertIn("UNCALIBRATED (not a verdict): TREE / SIBLING LINE: OPEN", out)

    def test_calibration_fit_recovers_the_model(self):
        # E_NEW code (task 1) n = 0..4 = 1, 2, 3, 4, 5 (n = 4 extrapolated), reason (task 2) = 1, 1, 2, 2, 2; step times built
        # from t_fixed 20, t_draft 2, c 0.5 -> the fit returns them and every arm is exact
        cal = self.d / "cal"
        cal.mkdir()
        e = {"code": [1, 2, 3, 4, 5], "reason": [1, 1, 2, 2, 2]}
        for n in range(5):
            rows = []
            for p in ("code", "reason"):
                steps = 10
                rows.append({"prompt": p, "predicted_n": 10 if n == 0 else 25, "steps": 0 if n == 0 else steps,
                             "predicted_ms": steps * (20 + 2 * n + 0.5 * e[p][n]), "draft_n": 0, "draft_n_accepted": 0})
            (cal / f"spec1cal_n{n}_1.json").write_text(json.dumps(rows))
        out, res = self.run_tool("--cal", str(cal))
        f = res["fit"]
        self.assertAlmostEqual(f["t_fixed"], 20, places=6)
        self.assertAlmostEqual(f["t_draft"], 2, places=6)
        self.assertAlmostEqual(f["c"], 0.5, places=6)
        self.assertTrue(f["ok"])
        self.assertIn("FIT OK", out)
        self.assertNotIn("UNCALIBRATED", out)
        # one arm 20% slow -> no fit within 3%, and no policy can pass
        rows = json.loads((cal / "spec1cal_n2_1.json").read_text())
        for r in rows:
            r["predicted_ms"] *= 1.2
        (cal / "spec1cal_n2_1.json").write_text(json.dumps(rows))
        out, res = self.run_tool("--cal", str(cal))
        self.assertFalse(res["fit"]["ok"])
        self.assertIn("NO VERDICT: the calibration fit did not pass the 3% rule", out)


if __name__ == "__main__":
    unittest.main()
