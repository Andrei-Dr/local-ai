"""tests for bench/qreport.py — synthetic fixtures built from the exact echo formats of bench/box/kld1.sh,
bench/box/hq1.sh (qual.py summary + jsonl schemas) and bench/box/lq1.sh (longctx summary + jsonl schemas).
No real results exist yet; parsers must be tolerant: unknown lines ignored, missing fields None, empty
input answers "no ..." with exit 0."""
import json, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import qreport

KLD_LOG = """##### kld1_base_q6 | 03:00:00 | saving reference logits (slow: the file streams from NVMe)
    Final estimate PPL/Q = 8.12
##### kld1_IQ2_M | 03:41:00 | 11040160768 bytes
    Mean    KLD: 0.08765 ± 0.00231
    Maximum KLD: 4.12000
    99.9%   KLD: 1.05000
    99.0%   KLD: 0.51000
    Median  KLD: 0.04000
    Same top p: 87.10000 %
    Mean PPL(Q): 9.40000
    Mean PPL(Q)/PPL(base): 1.15700
##### kld1_K2-expQ2K-downQ3K | 04:02:00 | 12900000000 bytes
    Mean    KLD: 0.06234 ± 0.00198 %
    Maximum KLD: 3.90000
    99.9%   KLD: 0.99000
    99.0%   KLD: 0.48000
    Median  KLD: 0.03100
    Same top p: 90.2 %
    Mean PPL(Q)/PPL(base): 1.094
##### kld1_OOMSUCK | 04:40:00 | 9000000000 bytes
    ggml_backend_cuda_buffer_type_alloc_buffer: allocating 201326592 bytes, out of memory
"""


def sj(path, obj):
    Path(path).write_text(json.dumps(obj))


def hard_pair(tmp, base="A", test="B"):
    d = Path(tmp)
    items = []
    rows_a, rows_b = [], []
    for i in range(20):
        s = ["aime", "math_l5", "humaneval_plus"][i % 3]
        ia = f"{s}/{i}"
        ok_a = i % 2 == 0
        ok_b = i % 4 != 0                       # B loses ids divisible by 4 (base-right, test-wrong)
        rows_a.append({"id": ia, "set": s, "ok": ok_a, "tokens": 900 + i, "tps": 30.0,
                       "finish": "stop" if i % 2 else "length", "empty": False, "text": "x"})
        rows_b.append({"id": ia, "set": s, "ok": ok_b, "tokens": 800, "tps": 31.0,
                       "finish": "stop", "empty": False, "text": "x"})
    for lab, rows in ((base, rows_a), (test, rows_b)):
        (d / f"{lab}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        sets = {}
        for s in ("aime", "math_l5", "humaneval_plus"):
            rs = [r for r in rows if r["set"] == s]
            p = sum(r["ok"] for r in rs) / len(rs)
            sets[s] = {"n": len(rs), "correct": sum(r["ok"] for r in rs), "pct": round(100 * p, 1),
                       "se_pct": round(100 * (p * (1 - p) / len(rs)) ** 0.5, 1),
                       "truncated": sum(r["finish"] == "length" for r in rs),
                       "empty": sum(r["empty"] for r in rs)}
        sj(d / f"{lab}.summary.json", {"label": lab, "thinking": True, "sets": sets,
                                       "mean_tokens": 850.0, "mean_decode_tps": 30.5,
                                       "truncated": sum(r["finish"] == "length" for r in rows),
                                       "empty": 0, "items": len(rows)})
    return str(d)


LQ_SUMMARY_Q4 = {"label": "lq1_q4", "depths": {
    "4096": {"n": 14, "needle_n": 10, "needle_pct": 100.0, "needle_se_pct": 0.0, "vt_mean": 1.0},
    "32768": {"n": 14, "needle_n": 10, "needle_pct": 90.0, "needle_se_pct": 9.5, "vt_mean": 0.929,
              "prefix_reused": False}}, "needle_pct_by_position": {"0.0-0.2": {"pct": 100.0, "n": 4}}}
LQ_ROWS_Q4 = [
    {"depth": 4096, "kind": "needle", "index": 0, "position": 0.0, "score": 1.0, "decode_tps": 3.2},
    {"depth": 32768, "kind": "needle", "index": 0, "position": 0.0, "score": 0.0, "decode_tps": 2.9},
    {"depth": 32768, "kind": "needle", "index": 1, "position": 0.11, "score": 1.0, "decode_tps": 2.9}]
LQ_SUMMARY_F16 = {"label": "lq1_f16", "depths": {
    "4096": {"n": 14, "needle_n": 10, "needle_pct": 100.0, "vt_mean": 1.0},
    "16384": {"n": 14, "needle_n": 10, "needle_pct": 100.0, "vt_mean": 1.0},
    "32768": {"n": 14, "needle_n": 10, "needle_pct": 100.0, "vt_mean": 0.929}},
    "needle_pct_by_position": {}}
LQ_ROWS_F16 = [
    {"depth": 4096, "kind": "needle", "index": 0, "position": 0.0, "score": 1.0, "decode_tps": 3.3},
    {"depth": 16384, "kind": "needle", "index": 0, "position": 0.0, "score": 1.0, "decode_tps": 3.1},
    {"depth": 32768, "kind": "needle", "index": 0, "position": 0.0, "score": 1.0, "decode_tps": 3.0},
    {"depth": 32768, "kind": "needle", "index": 1, "position": 0.11, "score": 1.0, "decode_tps": 3.0}]


class KldCase(unittest.TestCase):
    def test_parses_blocks_names_and_notes(self):
        blocks = qreport.parse_kld(KLD_LOG)
        by = {b["label"]: b for b in blocks}
        self.assertEqual(by["kld1_IQ2_M"]["metrics"]["Mean KLD"], 0.08765)
        self.assertAlmostEqual(by["kld1_IQ2_M"]["metrics"]["Mean KLD err"], 0.00231)
        self.assertEqual(by["kld1_K2-expQ2K-downQ3K"]["metrics"]["Same top p"], 90.2)
        self.assertTrue(by["kld1_OOMSUCK"].get("oom"))
        self.assertEqual(by["kld1_IQ2_M"]["bytes"], 11040160768)
        self.assertNotIn("kld1_base_q6", [b for b in blocks if b.get("candidate")])

    def test_table_sort_ratio_and_both_verdict_branches(self):
        md = qreport.report_kld(KLD_LOG, md=True)
        self.assertLess(md.index("| K2-expQ2K-downQ3K |"), md.index("| IQ2_M |"))  # sorted by mean KLD
        self.assertIn("x0.71", md)                                     # 0.06234/0.08765
        self.assertIn("BEST", md)
        self.assertIn("out of memory", md)                             # dead candidate surfaced
        log2 = KLD_LOG.replace("99.0%   KLD: 0.51000", "99.0%   KLD: 0.30000")   # IQ2_M best tail
        md2 = qreport.report_kld(log2)
        self.assertIn("disagree", md2)

    def test_empty_input_clean(self):
        md = qreport.report_kld("")
        self.assertIn("no kld1", md.lower())
        self.assertEqual(qreport.main(["kld", "--log", "/nonexistent/x.log"]), 0)


class HardCase(unittest.TestCase):
    def test_finished_only_and_truncation_columns(self):
        with tempfile.TemporaryDirectory() as td:
            d = hard_pair(td)
            md = qreport.report_hard(d, ["A", "B"], base="A")
            self.assertIn("finished-only", md)
            self.assertIn("AIME", md)
            self.assertIn("trunc", md.lower())
            # A: ids 0..19 step 2 ok (10); half were truncated; finished-only = 5 ok / 10 finished = 50.0
            line = next(ln for ln in md.splitlines() if "finished-only" in ln and "50.0" in ln)
            self.assertIsNotNone(line)

    def test_paired_section_uses_paired_module(self):
        with tempfile.TemporaryDirectory() as td:
            md = qreport.report_hard(hard_pair(td), ["A", "B"], base="A")
            self.assertIn("verdict", md.lower())
            self.assertIn("B", md)
            # 5 discordant pairs on n=20: A-right/B-wrong only -> B NON-INFERIOR style verdict row
            self.assertRegex(md, r"wins\s+\d+\s+loses\s+\d+")

    def test_missing_base_message_not_crash(self):
        with tempfile.TemporaryDirectory() as td:
            d = hard_pair(td)
            (Path(d) / "A.jsonl").unlink()
            md = qreport.report_hard(d, ["A", "B"], base="A")
            self.assertIn("missing", md.lower())
            self.assertEqual(qreport.main(["hard", "--results", d, "--labels", "A,B", "--base", "A"]), 0)


class LqCase(unittest.TestCase):
    def fixture(self, td, with_f16=True):
        d = Path(td)
        sj(d / "lq1_q4.longctx.summary.json", LQ_SUMMARY_Q4)
        (d / "lq1_q4.longctx.jsonl").write_text("\n".join(json.dumps(r) for r in LQ_ROWS_Q4) + "\n")
        if with_f16:
            sj(d / "lq1_f16.longctx.summary.json", LQ_SUMMARY_F16)
            (d / "lq1_f16.longctx.jsonl").write_text("\n".join(json.dumps(r) for r in LQ_ROWS_F16) + "\n")
        return str(d)

    def test_grid_with_missing_depth_and_noreuse(self):
        with tempfile.TemporaryDirectory() as td:
            md = qreport.report_lq(self.fixture(td))
            self.assertIn("NOREUSE", md)
            self.assertIn("32768", md)
            self.assertRegex(md, r"16384.*(100\.0|\| -).*" )
            lq16 = next(ln for ln in md.splitlines() if ln.startswith("| 16384"))
            self.assertTrue("–" in lq16 or " - " in lq16 or "—" in lq16)   # q4 lacks that depth

    def test_delta_vs_f16_paired(self):
        with tempfile.TemporaryDirectory() as td:
            md = qreport.report_lq(self.fixture(td))
            self.assertIn("vs f16", md.replace("lq1_", ""))
            self.assertRegex(md, r"lost 1 gained 0|−1")

    def test_arm_without_f16_overlap_note(self):
        with tempfile.TemporaryDirectory() as td:
            md = qreport.report_lq(self.fixture(td, with_f16=False))
            self.assertIn("no f16", md.lower())

    def test_empty_dir_says_so(self):
        with tempfile.TemporaryDirectory() as td:
            md = qreport.report_lq(td)
            self.assertIn("no lq1", md.lower())


if __name__ == "__main__":
    unittest.main()
