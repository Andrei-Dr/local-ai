"""tests for bench/jobreport.py — synthetic records through the importable report(); subprocess only for
the unknown-job exit and CLI file writing."""
import json, subprocess, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jobreport

TOOL = Path(__file__).resolve().parent.parent / "jobreport.py"


def prec(label, completed=True, tel=None, vram=3400, hit=42.0, code=20.0):
    r = {"kind": "specbench", "label": label, "ts": "2026-09-01T00:00:00+0300", "completed": completed,
         "vram_mib": vram, "model": "m.gguf", "git": {"commit": "deadbeef0"},
         "moe_cache": {"hit_rate_pct": hit} if hit is not None else None,
         "prompts": [{"prompt": "code", "decode_tps": code, "wall_tps": code, "prefill_tps": 300.0,
                      "acceptance": 0.8}]}
    if tel is not None:
        r["telemetry"] = tel
    return r


JOB = {"title": "T", "hyp": "HYPOTHESIS TEXT", "cmp": [("cmp one", r"^jb$", r"^jt$", ["decode_tps"])]}


class JobreportCase(unittest.TestCase):
    def setUp(self):
        jobreport.JOBS["_t1"] = JOB
        self.addCleanup(jobreport.JOBS.pop, "_t1", None)

    def test_complete_arms_write_tables_and_headings(self):
        recs = [prec("jb", code=20.0), prec("jt", code=25.0)]
        text, rc = jobreport.report("_t1", recs)
        self.assertEqual(rc, 0)
        self.assertTrue(text.startswith("# _t1 — T"), text[:40])
        self.assertIn("HYPOTHESIS TEXT", text)
        self.assertIn("## cmp one", text)
        self.assertIn("**decode_tps**", text)
        self.assertIn("WIN", text)                          # +25% clears the 1% floor at n=1
        self.assertIn("- `jb`", text)
        self.assertIn("- `jt`", text)                       # runs-seen lines

    def test_missing_arm_is_pending_not_failure(self):
        text, rc = jobreport.report("_t1", [prec("jb")])
        self.assertEqual(rc, 0)
        self.assertIn("_pending: no completed rows for ^jt$_", text)
        self.assertIn("- `jb`", text)                       # base arm label shows in runs-seen anyway
        t2 = jobreport.report("_t1", [prec("jb"), prec("jt", completed=False)])[0]
        self.assertIn("_pending: no completed rows for ^jt$_", t2)  # incomplete rows do not lift pending

    def test_unknown_job_exit_two(self):
        p = subprocess.run([sys.executable, str(TOOL), "nosuchjob", "--ledger", "/nonexistent.jsonl"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("unknown job", p.stderr)
        self.assertIn("p4c1", p.stderr)                     # registry named in the message

    def test_runs_line_dashes_missing_fields(self):
        recs = [prec("jb", tel={"power_w_avg": 75.2}), prec("jt", vram=None, hit=None)]
        text, rc = jobreport.report("_t1", recs)
        jb_ln = next(ln for ln in text.splitlines() if "`jb`" in ln)
        self.assertIn("power_w_avg 75.2", jb_ln)
        for k in ("gpu_util_avg", "pcie_rx_gbs_max", "mem_avail_mib_min", "swap_used_mib_max"):
            self.assertIn(f"{k} -", jb_ln)
        jt_ln = next(ln for ln in text.splitlines() if "`jt`" in ln)
        self.assertIn("vram -", jt_ln)
        self.assertIn("hit -", jt_ln)

    def test_cli_writes_registry_job_to_explicit_out(self):
        tmp = Path(tempfile.mkdtemp())
        led = tmp / "ledger.jsonl"
        led.write_text("\n".join(json.dumps(r) for r in
                                 [prec("lat_base"), prec("lat_pinned", code=30.0), prec("zz_other")]) + "\n")
        p = subprocess.run([sys.executable, str(TOOL), "lat1", "--ledger", str(led), "--out", str(tmp / "sub" / "lat1.md")],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        body = (tmp / "sub" / "lat1.md").read_text()       # parent dirs created
        self.assertIn("WIN", body)                          # lat_base vs lat_pinned table rendered
        self.assertIn("_pending: no completed rows for ^lat_nographs$_", body)  # other levers pending
        self.assertNotIn("zz_other", body)                  # appendix limited to job labels


    def test_new_registry_jobs_and_quality_table(self):
        # real registry: mtp1 exists and renders three pending comparisons on an empty ledger
        text, rc = jobreport.report("mtp1", [])
        self.assertEqual(rc, 0)
        self.assertEqual(text.count("_pending:"), 3)
        for name in ("r1", "n2b", "kq1g"):
            self.assertIn(name, jobreport.JOBS)

    def test_quality_table_reference_and_latest_set(self):
        jobreport.JOBS["_tq"] = {"title": "QT", "hyp": "h", "cmp": [("c", r"^qq_b$", r"^qq_t$", ["decode_tps"])],
                                 "qual": ["cfg_bias"]}
        self.addCleanup(jobreport.JOBS.pop, "_tq", None)
        def qual(label, ts, gsm=None, he=None, mlp=None):
            sets = {}
            for name, sd in (("gsm8k", gsm), ("humaneval", he), ("mmlu_pro", mlp)):
                if sd:
                    sets[name] = sd
            return {"kind": "quality", "label": label, "ts": ts, "quality": {"sets": sets}}
        recs = [qual(jobreport.QUAL_REF, "2026-09-01T00:00:00+0300",
                     gsm={"pct": 100.0, "se_pct": 0.0}, he={"pct": 95.1, "se_pct": 3.4}, mlp={"pct": 80.0, "se_pct": 4.7}),
                qual("cfg_bias", "2026-09-01T00:00:00+0300", gsm={"pct": 92.0, "se_pct": 3.8}),
                qual("cfg_bias", "2026-09-09T00:00:00+0300", gsm={"pct": 96.0, "se_pct": 2.8}),   # newer gsm8k wins
                prec("qq_b"), prec("qq_t", code=30.0)]
        text, rc = jobreport.report("_tq", recs)
        self.assertEqual(rc, 0)
        qt = text.split("## Quality", 1)[1].split("## Runs seen", 1)[0]
        self.assertIn("| label | gsm8k | humaneval | mmlu_pro |", qt)
        self.assertLess(qt.index(jobreport.QUAL_REF), qt.index("`cfg_bias`"))   # reference first
        self.assertIn("100.0 ±0.0", qt)
        self.assertIn("96.0 ±2.8", qt)                                          # latest per (label,set)
        row_bias = next(ln for ln in qt.splitlines() if "cfg_bias" in ln)
        self.assertIn("| - | - |", row_bias)                                    # absent sets dash
        self.assertIn("WIN", text)                                               # cmp table still renders


if __name__ == "__main__":
    unittest.main()
