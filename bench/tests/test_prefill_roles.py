"""tests for bench/ledger2md.py: prefill columns (specbench long prompt + longpf 9.3k) in the scoreboard matrix, the
STABLE / LEGACY / TEST build comparison, and the longpf ledger table. Synthetic records only."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ledger2md

V2, OLD, OV = "/ai/src/llama.cpp-v2/build75", "/ai/src/llama.cpp-mainline/build75", "/ai/src/llama.cpp-ov/build75"


def q(label, model="k2.gguf"):
    sets = {"gsm8k": {"n": 50, "pct": 95.0, "se_pct": 2.8}, "humaneval": {"n": 41, "pct": 90.0, "se_pct": 4.1},
            "mmlu_pro": {"n": 70, "pct": 80.0, "se_pct": 4.7, "truncated": 0}}
    return {"kind": "quality", "label": label, "model": model, "ts": "2026-09-01T00:00:00+0300",
            "quality": {"sets": sets, "mean_decode_tps": 30.0, "truncated": 0}}


def sbp(label, build, rows, model="k2.gguf", completed=True, ts="2026-09-23T07:00:00+0300"):
    """specbench record: rows = [(prompt, decode_tps, prefill_tps), ...]"""
    return {"kind": "specbench", "label": label, "ts": ts, "model": model, "build": build, "completed": completed,
            "args": "-ub 128", "git": {"commit": "abc"}, "moe_cache": None,
            "prompts": [{"prompt": p, "decode_tps": d, "prefill_tps": f} for p, d, f in rows]}


def lp(label, build, prefill, decode, model="k2.gguf", completed=True, env=None):
    return {"kind": "longpf", "label": label, "ts": "2026-09-23T07:40:00+0300", "model": model, "build": build,
            "completed": completed, "args": "-c 12288", "env": env, "git": None, "source": "backfill",
            "longpf": {"prompt_n": 9279, "prefill_tps": prefill, "decode_tps": decode, "wall_s": 25.6}}


RECS = [q("k2_cfg"),
        sbp("new_a", V2, [("code", 58.0, 60.0), ("long", 47.0, 395.1)]),
        sbp("old_a", OLD, [("code", 57.5, 40.0), ("long", 49.1, 48.2)]),
        sbp("ov_a", OV, [("code", 59.9, 61.0), ("long", 46.0, 390.0)]),
        lp("pf_new", V2, 403.2, 49.8), lp("pf_old", OLD, 47.3, 47.3), lp("pf_bad", V2, 999.0, 1.0, completed=False)]


class PrefillMatrixCase(unittest.TestCase):
    def test_matrix_has_prefill_columns_best_per_model(self):
        md = ledger2md.scoreboard(RECS)
        head = [ln for ln in md.splitlines() if ln.startswith("| model |")][0]
        self.assertIn("prefill 2.2k tok/s", head)
        self.assertIn("prefill 9.3k tok/s", head)
        row = [ln for ln in md.splitlines() if "`k2_cfg`" in ln][0]
        self.assertIn("395.1", row)          # best specbench long-prompt prefill for the model
        self.assertIn("403.2", row)          # best completed longpf prefill
        self.assertNotIn("999.0", md)        # incomplete longpf never counts


class BuildRolesCase(unittest.TestCase):
    def sec(self):
        md = ledger2md.scoreboard(RECS)
        return md.split("## Builds: STABLE vs LEGACY vs TEST", 1)[1].split("\n## ", 1)[0]

    def test_rows_per_role_with_best_values_and_labels(self):
        s = self.sec()
        stable = [ln for ln in s.splitlines() if ln.startswith("| STABLE")][0]
        legacy = [ln for ln in s.splitlines() if ln.startswith("| LEGACY")][0]
        test = [ln for ln in s.splitlines() if ln.startswith("| TEST llama.cpp-ov")][0]
        self.assertIn("403.2", stable); self.assertIn("395.1", stable); self.assertIn("`new_a`", stable)
        self.assertIn("47.3", legacy); self.assertIn("48.2", legacy)
        self.assertIn("59.9", test)
        self.assertLess(s.index("| STABLE"), s.index("| LEGACY"))

    def test_stable_vs_legacy_ratios(self):
        s = self.sec()
        self.assertIn("prefill 9.3k 8.52x", s)   # 403.2 / 47.3
        self.assertIn("prefill 2.2k 8.20x", s)   # 395.1 / 48.2
        self.assertIn("decode code +0.9%", s)    # 58.0 / 57.5


class IdentityModeCase(unittest.TestCase):
    def test_identity_runs_excluded_by_env_or_legacy_label(self):
        recs = RECS + [dict(sbp("new_sync", V2, [("code", 10.0, 5.0), ("long", 10.0, 5.0)]), env="LLAMA_MOE_CACHE_SYNC=1"),
                       sbp("promo9_id_x", OLD, [("code", 11.0, 6.0), ("long", 11.0, 6.0)])]      # no env: label rule
        s = ledger2md.scoreboard(recs).split("## Builds: STABLE vs LEGACY vs TEST", 1)[1].split("\n## ", 1)[0]
        self.assertNotIn("new_sync", s)
        self.assertNotIn("promo9_id_x", s)
        self.assertIn("decode code +0.9%", s)   # medians unchanged by the excluded rows

    def test_env_recorded_non_sync_run_counts(self):
        r = dict(sbp("new_env", V2, [("code", 70.0, 5.0)]), env="GGML_CUDA_FA_TILE_MIN_BATCH=32")
        self.assertFalse(ledger2md.identity_mode(r))
        self.assertTrue(ledger2md.identity_mode(dict(r, env="LLAMA_MOE_CACHE_SYNC=1 X=1")))
        self.assertTrue(ledger2md.identity_mode(dict(r, env={"LLAMA_MOE_CACHE_SYNC": "1"})))   # older dict-shaped env
        self.assertFalse(ledger2md.identity_mode(dict(r, env={"GGML_X": "1"})))


class LongpfLedgerCase(unittest.TestCase):
    def test_longpf_table_rows(self):
        md = "\n".join(ledger2md.longpf_md(RECS))
        row = [ln for ln in md.splitlines() if "`pf_new`" in ln][0]
        self.assertIn("STABLE", row)
        self.assertIn("403.2", row)
        self.assertIn("9279", row)
        bad = [ln for ln in md.splitlines() if "`pf_bad`" in ln][0]
        self.assertIn("FAILED", bad)


if __name__ == "__main__":
    unittest.main()


class ServedArcCase(unittest.TestCase):
    """The served configuration is a (build, model) pair: the arc table follows it across build AND model changes."""
    STEPS = [("LEGACY", OLD, "k2.gguf"), ("STABLE build", V2, "k2.gguf"), ("STABLE", V2, "k2q6.gguf")]
    RECS = [sbp("old_a", OLD, [("code", 50.0, 40.0), ("long", 40.0, 48.0)]),
            sbp("new_a", V2, [("code", 58.0, 60.0), ("long", 47.0, 390.0)]),
            sbp("new_b", V2, [("code", 60.0, 60.0), ("long", 49.0, 400.0)]),
            sbp("q6_a", V2, [("code", 64.0, 60.0), ("long", 49.0, 480.0)], model="k2q6.gguf"),
            sbp("q6_ov", OV, [("code", 99.0, 60.0)], model="k2q6.gguf"),  # other tree: not part of the arc
            lp("pf_old", OLD, 47.0, 47.0), lp("pf_q6", V2, 510.0, 55.0, model="k2q6.gguf")]

    def arc(self):
        return "\n".join(ledger2md.served_arc(self.RECS, self.STEPS))

    def test_one_row_per_step_with_medians(self):
        md = self.arc()
        self.assertIn("## Served arc", md)
        self.assertRegex(md, r"\| LEGACY \| LEGACY \| k2 \| 50\.0 \| 40\.0 \| 48\.0 \| 47\.0 \| 47\.0 \| 2 \|")
        self.assertRegex(md, r"\| STABLE build \| STABLE \| k2 \| 59\.0 \| 48\.0 \| 395\.0 \| - \| - \| 2 \|")
        self.assertRegex(md, r"\| STABLE \| STABLE \| k2q6 \| 64\.0 \| 49\.0 \| 480\.0 \| 510\.0 \| 55\.0 \| 2 \|")
        self.assertNotIn("99.0", md)

    def test_last_step_vs_first_step(self):
        self.assertIn("**STABLE vs LEGACY (median vs median):** decode code +28.0%, decode long +22.5%, prefill 2.2k 10.00x, "
                      "prefill 9.3k 10.85x, decode after 9.3k +17.0%", self.arc())

    def test_scoreboard_includes_the_arc(self):
        self.assertIn("## Served arc", ledger2md.scoreboard(self.RECS))
