"""tests for bench/abreport.py — synthetic ledgers in temp files, run as subprocesses."""
import json, subprocess, sys, tempfile, unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parent.parent / "abreport.py"


def prec(label, ts, code, acc=0.5, completed=True, vram=3400, commit="deadbeef0", kind="code"):
    return {"kind": "specbench", "label": label, "ts": ts, "completed": completed, "vram_mib": vram,
            "model": "m.gguf", "git": {"commit": commit}, "moe_cache": {"hit_rate_pct": 42.0},
            "prompts": [{"prompt": kind, "tokens": 200, "decode_tps": code, "wall_tps": code,
                         "prefill_tps": 300.0, "acceptance": acc}]}


def run(recs, args):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for r in recs:
        tmp.write(json.dumps(r) + "\n")
    tmp.close()
    try:
        return subprocess.run([sys.executable, str(TOOL), tmp.name] + args, capture_output=True, text=True)
    finally:
        Path(tmp.name).unlink()


class AbreportCase(unittest.TestCase):
    def test_ab_exact_delta_noise_verdict(self):
        recs = [prec("ab_b1", "2026-09-01T00:00:00+0300", 20.0), prec("ab_b2", "2026-09-01T00:00:00+0300", 21.0),
                prec("ab_t1", "2026-09-01T00:00:00+0300", 22.0, commit="cafef00d1"),
                prec("ab_t2", "2026-09-01T00:00:00+0300", 22.4, commit="cafef00d1")]
        p = run(recs, ["ab_b", "ab_t"])
        self.assertEqual(p.returncode, 0, p.stderr)
        for s in ("20.50 (n=2)", "22.20 (n=2)", "+8.29%", "4.88%", "WIN"):
            self.assertIn(s, p.stdout)
        self.assertIn("ALL", p.stdout)          # ALL row present
        self.assertIn("cafef00d1", p.stdout)    # per-arm commit lines
        self.assertIn("base:", p.stdout)

    def test_latest_by_ts_dedupe(self):
        recs = [prec("dup", "2026-09-01T00:00:00+0300", 10.0), prec("dup", "2026-09-09T00:00:00+0300", 90.0),
                prec("t1", "2026-09-01T00:00:00+0300", 90.0)]
        p = run(recs, ["^dup$", "^t1$"])
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("90.00 (n=1)", p.stdout)
        self.assertNotIn("10.00", p.stdout)     # superseded record's value is gone
        self.assertIn("+0.00%", p.stdout)
        self.assertIn("flat", p.stdout)

    def test_incomplete_records_ignored(self):
        recs = [prec("b1", "2026-09-01T00:00:00+0300", 20.0, acc=None),
                prec("b2", "2026-09-01T00:00:00+0300", 500.0, completed=False),
                prec("t1", "2026-09-01T00:00:00+0300", 21.0, acc=None)]
        p = run(recs, ["^b", "^t1$"])
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("20.00 (n=1)", p.stdout)
        self.assertNotIn("500.00", p.stdout)
        self.assertIn("acceptance code -", p.stdout)  # real ledgers carry acceptance=null

    def test_empty_arm_exit_two(self):
        p = run([prec("b1", "2026-09-01T00:00:00+0300", 20.0)], ["^b", "^zzz$"])
        self.assertEqual(p.returncode, 2)
        self.assertIn("arm 'test'", p.stderr)

    def test_md_table_shape(self):
        recs = [prec("b1", "2026-09-01T00:00:00+0300", 20.0), prec("t1", "2026-09-01T00:00:00+0300", 21.0)]
        p = run(recs, ["^b", "^t1", "--md"])
        self.assertEqual(p.returncode, 0, p.stderr)
        lines = p.stdout.splitlines()
        self.assertTrue(lines[0].startswith("| prompt"), lines[0])
        self.assertTrue(lines[1].startswith("|---"), lines[1])


if __name__ == "__main__":
    unittest.main()
