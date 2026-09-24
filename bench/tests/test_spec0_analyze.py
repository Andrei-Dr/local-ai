"""tests for bench/box/spec0_analyze.py — synthetic dumps with known answers, run as subprocesses."""
import json, struct, subprocess, sys, tempfile, unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "box" / "spec0_analyze.py"


def top(ids):
    return [[t, 0.1, 1.0] for t in ids] + [[900 + j, 0.01, 0.0] for j in range(10 - len(ids))]


def step(i, n_accept, tgt_first, draft_top1=(5, 6, 7), d1_top=(5, 11, 12, 13), replay=False):
    """a 3-token draft; depth 1's draft top list starts with d1_top; tgt_tok = accepted drafts then the correction"""
    draft = list(draft_top1)
    tgt = draft[:n_accept] + ([tgt_first] if n_accept < 3 else [99])
    return {"step": i, "seq": 0, "task": 1, "n_past": 10 + i, "replay": replay, "n_draft": 3, "n_accept": n_accept,
            "draft_tok": draft, "tgt_tok": tgt, "sampler": {"temp": 0}, "p_min": 0,
            "draft": [{"tok": draft[0], "kept": True, "lse": 0, "lse_t": None, "top": top(d1_top)}] +
                     [{"tok": t, "kept": True, "lse": 0, "lse_t": None, "top": top([t, 40, 41])} for t in draft[1:]],
            "target": [{"lse": 0, "lse_t": None, "top": top([1])} for _ in range(4)]}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def sib_record(kind, pos, tok, rank, prob, layers):
    k = len(next(iter(layers.values())))
    b = struct.pack("<4if2i", kind, pos, tok, rank, prob, len(layers), k)
    for il, ids in sorted(layers.items()):
        b += struct.pack(f"<i{k}i", il, *ids)
    return b


class Spec0Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *args):
        p = subprocess.run([sys.executable, str(TOOL), *args, "--json", str(self.d / "out.json")],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        return p.stdout, json.loads((self.d / "out.json").read_text())

    def test_h1_h2_known_counts(self):
        # 10 steps: depth 1 rejected 4x (targets 11 = draft rank 2, 13 = rank 4, 50 = not in top-4, 50), accepted 6x;
        # of the 6: depth 2 rejected 3x, accepted 3x; of those 3: depth 3 accepted 1x. One replayed step is ignored.
        rows = [step(0, 0, 11), step(1, 0, 13), step(2, 0, 50), step(3, 0, 50),
                step(4, 1, 8), step(5, 1, 8), step(6, 1, 8), step(7, 2, 9), step(8, 2, 9), step(9, 3, 0),
                step(10, 0, 11, replay=True)]
        write_jsonl(self.d / "t0.jsonl", rows)
        out, res = self.run_tool("--t0", str(self.d / "t0.jsonl"))
        h1 = res["T0"]["h1"]
        self.assertEqual(res["T0"]["steps"], 10)
        self.assertAlmostEqual(h1["1"]["acc"], 0.6)
        self.assertAlmostEqual(h1["2"]["acc"], 0.5)    # 3 of 6
        self.assertAlmostEqual(h1["3"]["acc"], 1 / 3)  # 1 of 3
        self.assertAlmostEqual(h1["3"]["acc_given_d1"], 1 / 6)
        self.assertEqual(res["T0"]["h1_verdict"], "MIXED (one depth collapses)")  # 0.8 * 0.6 = 0.48: d2 0.5 no, d3 0.33 yes
        self.assertEqual(res["T0"]["h2"]["n_rejected"], 4)
        self.assertAlmostEqual(res["T0"]["h2"]["p_top2_4"], 0.5)
        self.assertEqual(res["h2_verdict"], "TRUE at T = 0 (L3 / L4 open)")
        # one more depth-2 rejection: d2 = 3 of 7 = 0.43 < 0.8 * 7 / 11 = 0.51 -> both depths collapse
        write_jsonl(self.d / "t0.jsonl", rows + [step(11, 1, 8)])
        _, res = self.run_tool("--t0", str(self.d / "t0.jsonl"))
        self.assertEqual(res["T0"]["h1_verdict"], "TRUE (collapse)")

    def test_h1_no_collapse_and_h2_dead(self):
        rows = [step(i, 3, 0) for i in range(8)] + [step(8, 0, 50), step(9, 0, 50)]
        write_jsonl(self.d / "t0.jsonl", rows)
        write_jsonl(self.d / "card.jsonl", [step(0, 0, 11), step(1, 0, 12), step(2, 0, 50)])
        _, res = self.run_tool("--t0", str(self.d / "t0.jsonl"), "--card", str(self.d / "card.jsonl"))
        self.assertEqual(res["T0"]["h1_verdict"], "FALSE (no collapse: L1 demoted)")
        self.assertEqual(res["T0"]["h2"]["p_top2_4"], 0.0)
        self.assertAlmostEqual(res["card"]["h2"]["p_top2_4"], 2 / 3)
        self.assertEqual(res["h2_verdict"], "DEAD at T = 0; OPEN for T > 0 (card sampler >= 0.30)")

    def test_route_alignment_and_new_experts(self):
        rows = [step(0, 1, 8), step(1, 0, 50)]
        write_jsonl(self.d / "t0.jsonl", rows)
        rec = b""
        # step 0: 4 tokens x k 2 on one layer; step 1: the rejected tokens reuse step 0's experts except one new id each
        for s, toks in ((0, [(1, 2), (3, 4), (5, 6), (7, 8)]), (1, [(1, 2), (3, 100), (5, 101), (7, 102)])):
            rec += struct.pack("<4i", 0x30525053, s, 1, 2) + struct.pack("<2i", 0, 4)
            rec += struct.pack("<8i", *[e for t in toks for e in t])
        (self.d / "t0.route").write_bytes(rec)
        _, res = self.run_tool("--t0", str(self.d / "t0.jsonl"), "--route", str(self.d / "t0.route"))
        r = res["route"]
        self.assertEqual(r["misaligned"], 0)
        # step 0 (empty LRU): accepted depth 1 -> 2 new; rejected depths 2, 3 -> 2 each. step 1: 3 rejected -> 1 new each
        self.assertAlmostEqual(r["new_per_layer_accepted"], 2.0)
        self.assertAlmostEqual(r["new_per_layer_rejected"], (2 + 2 + 1 + 1 + 1) / 5)

    def test_h3_sibling_new_experts(self):
        # 2 layers, k 2. path tokens 0..7 route to distinct pairs; LRU with 4 slots per layer.
        path = {p: {0: (2 * p, 2 * p + 1), 1: (50 + 2 * p, 51 + 2 * p)} for p in range(8)}
        b = struct.pack("<2i", 0x30424953, 2)
        # sampled p = 4: batch = path 3..6; LRU (4 slots) after path 0..2 = experts of path 1, 2.
        # alt A (rank 1): layer 0 = (0, 2): 0 evicted -> new, 2 in LRU; layer 1 = (56, 58): both in batch -> 0 new
        #   -> (1 + 0) / 2 = 0.5.  alt B (rank 2): layer 0 = (200, 201) new; layer 1 = (202, 52): 52 is path 1 = LRU
        #   -> (2 + 1) / 2 = 1.5.  mean 1.0
        b += sib_record(1, 4, 90, 1, 0.2, {0: (0, 2), 1: (56, 58)})
        b += sib_record(1, 4, 91, 2, 0.1, {0: (200, 201), 1: (202, 52)})
        for p in range(8):
            b += sib_record(2 if p == 4 else 0, p, 10 + p, 0 if p == 4 else -1, 0.6 if p == 4 else 0.0, path[p])
        (self.d / "s.bin").write_bytes(b)
        write_jsonl(self.d / "t0.jsonl", [step(0, 3, 0)])
        out, res = self.run_tool("--t0", str(self.d / "t0.jsonl"), "--sib", str(self.d / "s.bin"), "--slots", "4")
        h3 = res["h3"]
        self.assertEqual(h3["n_siblings"], 2)
        self.assertAlmostEqual(h3["new_per_layer"], 1.0)
        self.assertAlmostEqual(h3["by_target_rank"]["2"], 0.5)
        self.assertAlmostEqual(h3["by_target_rank"]["3"], 1.5)
        self.assertAlmostEqual(h3["path_token_new_vs_lru"], 2.0)  # path 4's experts are all outside the LRU
        self.assertEqual(res["h3_verdict"], "TRUE (width cheap)")


if __name__ == "__main__":
    unittest.main()
