"""tests for bench/ledger2md.py scoreboard(): knowledge truncation as a bound (<=), clean-enough competition,
picker eligibility, and latest-record-per-label/set selection. Synthetic records only; no files, no network."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ledger2md


def q(label, ts="2026-09-01T00:00:00+0300", model="m.gguf", math=95.0, code=90.0,
     kn=80.0, kn_n=70, cut=0, tps=30.0):
    """Minimal quality record scoreboard() consumes. cut=None = per-set count unknown (old record)."""
    sets = {"gsm8k": {"n": 50, "pct": math, "se_pct": 2.8},
            "humaneval": {"n": 41, "pct": code, "se_pct": 4.1}}
    if kn is not None:
        s = {"n": kn_n, "pct": kn, "se_pct": 4.7}
        if cut is not None:
            s["truncated"] = cut
        sets["mmlu_pro"] = s
    truncated = cut if (cut is not None and cut > 0) else (0 if cut == 0 else 3)
    return {"kind": "quality", "label": label, "model": model, "ts": ts,
            "quality": {"sets": sets, "mean_decode_tps": tps, "truncated": truncated}}


class RenderCase(unittest.TestCase):
    def test_clean_enough_bound_and_beats_clean_lower(self):
        md = ledger2md.scoreboard([q("sx_cap2", kn=80.0, cut=2, tps=30), q("sx_clean", kn=71.4, cut=0, tps=20)])
        self.assertIn("80.0 (<=82.9)", md)  # 80.0 + 100*2/70 = 82.857
        self.assertIn("80.0 (<=82.9) ✅", md)
        self.assertIn("knowledge=sx_cap2", md)
        self.assertNotIn("71.4 ✅", md)
        self.assertNotIn("knowledge winner unresolved", md)

    def test_contaminated_known_cut_flag_plus_bound(self):
        md = ledger2md.scoreboard([q("sx_cut19", kn=62.9, cut=19)])
        self.assertIn("62.9! (<=90.0)", md)  # 62.9 + 100*19/70 = 90.04
        self.assertIn("knowledge=?", md)
        self.assertIn("knowledge winner unresolved", md)

    def test_unknown_cut_flags_without_bound(self):
        md = ledger2md.scoreboard([q("sx_old", kn=60.0, cut=None)])
        self.assertIn("60.0!", md)
        self.assertNotIn("60.0 (", md)


class PickerCase(unittest.TestCase):
    def test_fastest_eligible_wins_contaminated_never_eligible(self):
        recs = [q("sx_fast", kn=70.0, cut=0, tps=40.0),      # clean, eligible, mid speed
                q("sx_slowok", kn=71.0, cut=1, tps=30.0),    # clean-enough (1.4pts <= 4.7 SE), eligible
                q("sx_zerp", kn=85.0, cut=20, tps=50.0)]     # contaminated (28.6pts > SE), fastest but barred
        md = ledger2md.scoreboard(recs)
        self.assertIn("WINNER: `sx_fast`", md)
        self.assertIn("knowledge=sx_slowok", md)             # highest floor pct among competitors
        ranks = [ln for ln in md.split("## Best config", 1)[1].splitlines() if ln.startswith("|")]
        self.assertFalse([ln for ln in ranks if "`sx_zerp`" in ln])  # never in the eligible rank table
        self.assertIn("Excluded: `sx_zerp` (knowledge contaminated", md)


class LatestRecordCase(unittest.TestCase):
    def test_latest_record_per_label_wins(self):
        recs = [q("sx_dup", ts="2026-09-01T00:00:00+0300", kn=40.0, cut=25),
                q("sx_dup", ts="2026-09-10T00:00:00+0300", kn=80.0, cut=0)]
        md = ledger2md.scoreboard(recs)
        self.assertIn("knowledge=sx_dup", md)
        self.assertNotIn("40.0!", md)
        self.assertNotIn("(<=", md)  # latest record is zero-cut: rendered clean, no bound

    def test_newer_record_without_a_set_inherits_the_previous(self):
        recs = [q("sx_partial", ts="2026-09-01T00:00:00+0300", kn=40.0, cut=25),
                q("sx_partial", ts="2026-09-10T00:00:00+0300", kn=None)]
        md = ledger2md.scoreboard(recs)
        self.assertIn("40.0! (<=75.7)", md)  # knowledge comes from the OLDER record carrying that set

if __name__ == "__main__":
    unittest.main()
