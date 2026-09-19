"""tests for bench/ledger2md.py scoreboard(): knowledge truncation as a bound (<=), clean-enough competition,
picker eligibility, latest-record-per-label/set selection, and the top-5 leaderboards. Synthetic records
only; no files, no network."""
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


def sb(label, ts="2026-09-01T00:00:00+0300", model="m.gguf", decodes=(30.0,), args="",
       cache={"slots": 96, "hit_rate_pct": 55.0}, vram=3400, commit="deadbeef0", completed=True):
    """Minimal specbench record the speed leaderboard consumes."""
    return {"kind": "specbench", "label": label, "ts": ts, "model": model, "completed": completed,
            "args": args, "vram_mib": vram, "git": {"commit": commit}, "moe_cache": cache,
            "prompts": [{"prompt": f"p{i}", "decode_tps": d} for i, d in enumerate(decodes)]}


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
        ranks = [ln for ln in md.split("## Best config", 1)[1].split("Closest misses", 1)[0].splitlines()
                 if ln.startswith("|")]
        self.assertFalse([ln for ln in ranks if "`sx_zerp`" in ln])  # never in the eligible rank table
        miss = md.split("Closest misses", 1)[1]                       # exclusion reasons now live in that table
        self.assertIn("`sx_zerp`", miss)
        self.assertIn("knowledge contaminated", miss)


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

class TopFiveCase(unittest.TestCase):
    def board(self, md, name):
        return md.split(f"### {name}", 1)[1].split("###", 1)[0].split("## ", 1)[0]

    def test_math_top5_order_and_tie_notes(self):
        recs = [q("m_a", math=96.0), q("m_b", math=94.0), q("m_c", math=93.3), q("m_d", math=89.0),
                q("m_e", math=84.0), q("m_f", math=79.0)]
        sec = self.board(ledger2md.scoreboard(recs), "math (GSM8K)")
        body = [ln for ln in sec.splitlines() if ln.startswith("| ") and not ln.startswith("| rank") and "---" not in ln]
        self.assertLessEqual(len(body), 5)                            # capped
        self.assertNotIn("`m_f`", sec)                                # sixth row dropped
        pcts = [float(ln.split("|")[4].strip().split(" ")[0]) for ln in body]
        self.assertEqual(pcts, sorted(pcts, reverse=True))            # desc
        self.assertEqual(sec.count("tie with #1 within 1 SE"), 2)      # exactly 94.0 and 93.3 (>= 96 - 2.8 SE)

    def test_knowledge_competitive_before_contaminated(self):
        recs = [q("k_cont", kn=85.0, cut=20, tps=50.0),   # contaminated floor pct 85 (band 28.6 > 4.7 SE)
                q("k_clean", kn=78.0, cut=0, tps=30.0),
                q("k_enough", kn=70.0, cut=2, tps=25.0)]  # clean-enough (2.9 <= 4.7)
        sec = self.board(ledger2md.scoreboard(recs), "knowledge (MMLU-Pro)")
        self.assertLess(sec.index("`k_clean`"), sec.index("`k_cont`"))     # despite lower pct
        self.assertLess(sec.index("`k_enough`"), sec.index("`k_cont`"))
        self.assertIn("floor only, not ranked against clean rows", sec)
        self.assertIn("70.0 (<=72.9)", sec)                                 # main-table bound rendering reused
        self.assertIn("85.0!", sec)

    def test_speed_board_from_specbench_latest_completed_notes(self):
        recs = [sb("sp_a", ts="2026-09-01T00:00:00+0300", decodes=(30.0,)),
                sb("sp_a", ts="2026-09-09T00:00:00+0300", decodes=(50.0, 40.0),
                   args="--spec-type draft-mtp --spec-draft-n-max 2", cache=None),
                sb("sp_b", decodes=(45.0,), args=""),
                sb("sp_c", decodes=(900.0,), completed=False)]
        sec = self.board(ledger2md.scoreboard(recs), "speed (decode tok/s)")
        self.assertIn("50.0 (mean 45.0 over 2 prompts)", sec)   # latest per label, max over prompts
        self.assertNotIn("30.0 (", sec)                          # superseded record gone
        self.assertNotIn("900.0", sec)                           # incomplete record ignored
        self.assertIn("MTP n=2", sec)
        self.assertIn("no cache", sec)                           # sp_a latest: moe_cache null
        self.assertIn("no spec", sec)                            # sp_b: plain args
        self.assertIn("slots 96", sec)                           # sp_b default cache present

    def test_closest_misses_carries_reasons(self):
        recs = [q("ok_cfg", kn=70.0, cut=0, tps=40.0), q("bad_cfg", kn=30.0, cut=20, tps=50.0)]
        md = ledger2md.scoreboard(recs)
        miss = md.split("Closest misses", 1)[1]
        self.assertIn("`bad_cfg`", miss)
        self.assertIn("knowledge contaminated", miss)
        self.assertNotIn("Excluded:", md)                        # old paragraph is gone
        self.assertLess(md.index("WINNER"), md.index("Closest misses"))


if __name__ == "__main__":
    unittest.main()
