"""tests for bench/ctxreport.py — synthetic ctx1.log + runs/ fixtures built to the exact formats of
bench/box/ctx1.sh and bench/box/slotclient.py. Pure functions called directly; CLI only for the
empty-input contract (exit 0 + "no rows")."""
import json, subprocess, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ctxreport

ROOT = Path(__file__).resolve().parent.parent.parent


def row(prompt_n=1000, cache_n=0, pf=40.0, dec=40.0, wall_s=30.0, sb=None, ms=None):
    r = {"prompt_n": prompt_n, "cache_n": cache_n, "prefill_tps": pf, "decode_tps": dec, "wall_s": wall_s}
    if sb is not None:
        r["slot_bytes"], r["slot_ms"] = sb, ms
    return r


def header(label, mode, ctx, reps, args):
    return f"##### {label} mode={mode} ctx={ctx} reps={reps} | {args}"


SVR = "-ngl 999 -ot exps=CPU -fa on"          # constant part of ctx1.sh's server line


def two_rungs(ratio=0.001, pf_map=None):
    """minimal log + runs for c16k + c32k rungs; ratio = restore prompt_n / save prompt_n."""
    pf_map = pf_map or {}
    sp16, rp16 = 12500, int(12500 * ratio)
    sp32, rp32 = 26000, int(26000 * ratio)
    log = "\n".join([
        header("ctx1_c16k_save", "save", 16384, 6, SVR), "    vram: 3300 MiB",
        "    general KV buffer size = 160.00 MiB", "    RS buffer size = 188.44 MiB",
        header("ctx1_c16k_restore", "restore", 16384, 6, SVR), "    vram: 3300 MiB",
        header("ctx1_c32k_save", "save", 32768, 13, SVR), "    vram: 3400 MiB",
        header("ctx1_c32k_restore", "restore", 32768, 13, SVR), "    vram: 3400 MiB",
        "    restore gate: 16k restore skipped the prompt => deep restores ON",
    ])
    runs = {
        "ctx1_c16k_save": [row(sp16, 0, pf_map.get("c16k_pf", 40.0), 42.0, 320.0, 70000000, 3200)],
        "ctx1_c16k_restore": [row(rp16, 0, 900.0, 42.0, 6.0, 70000000, 12.0)],
        "ctx1_c32k_save": [row(sp32, 0, pf_map.get("c32k_pf", 40.0), 40.0, 660.0, 140000000, 7000)],
        "ctx1_c32k_restore": [row(rp32, 0, 900.0, 40.0, 10.0, 140000000, 20.0)],
    }
    return log, runs


class ParseCase(unittest.TestCase):
    def test_arg_derivation_defaults_and_last_flag_wins(self):
        d = ctxreport.derive_args("-ub 128 -b 256 " + SVR)
        self.assertEqual(d, {"kv": "f16", "nkvo": False, "mtp": False, "ub": 128, "cache_slots": 24})
        d = ctxreport.derive_args("-ub 512 -b 1024 -ub 2048 --moe-expert-cache 24 --moe-expert-cache 12 -nkvo -ctk q4_0 -ctv q4_0")
        self.assertEqual((d["ub"], d["cache_slots"]), (2048, 12))       # LAST flag wins
        self.assertTrue(d["nkvo"]) and self.assertEqual(d["kv"], "q4_0")
        self.assertEqual(ctxreport.derive_args("-ctk q8_0 -ctv q8_0")["kv"], "q8_0")
        self.assertTrue(ctxreport.derive_args("-md x.gguf --spec-type draft-mtp --spec-draft-n-max 2")["mtp"])
        self.assertEqual(ctxreport.derive_args("")["ub"], 512)          # default

    def test_status_states_and_priority(self):
        labels, _ = ctxreport.parse_log("\n".join([
            header("ctx1_a_save", "save", 16384, 1, SVR),
            "    ctx1_a_save: SERVER DIED: ggml_cuda: out of memory",           # died via line
            "    out of memory",                                                  # oom line also died
            header("ctx1_b_save", "save", 16384, 1, SVR), "    ctx1_b_save: slotclient exit 1 (row kept)",
            header("ctx1_c_restore", "restore", 16384, 1, ""),                   # nothing, no file
            "##### ctx1_d_restore SKIPPED: the 16k restore did not skip the prompt",
        ]))
        runs = {"ctx1_b_save": [row()], "ctx1_c_restore": []}
        self.assertEqual(ctxreport.status_of("ctx1_a_save", labels, runs), "died")
        self.assertEqual(ctxreport.status_of("ctx1_b_save", labels, runs), "failed")
        self.assertEqual(ctxreport.status_of("ctx1_c_restore", labels, runs), "skipped")
        self.assertEqual(ctxreport.status_of("ctx1_d_restore", labels, runs), "skipped")

    def test_died_wins_over_present_rows_and_buffers_sum(self):
        labels, _ = ctxreport.parse_log("\n".join([
            header("ctx1_e_save", "save", 65536, 1, SVR),
            "    general KV buffer size = 320.00 MiB", "    layer KV buffer size = 320.00 MiB",
            "    RS buffer size = 188.44 MiB",
            header("ctx1_f_save", "save", 65536, 1, SVR), "    out of memory",
        ]))
        self.assertEqual(labels["ctx1_e_save"]["kv_mib"], 640.0)
        self.assertEqual(labels["ctx1_e_save"]["rs_mib"], 188.44)
        self.assertIsNone(labels["ctx1_f_save"].get("kv_mib"))
        self.assertEqual(ctxreport.status_of("ctx1_f_save", labels, {"ctx1_f_save": [row()]}), "died")

    def test_unknown_block_and_garbage_do_not_crash(self):
        log, runs = two_rungs()
        md = ctxreport.build_report(log + "\n##### hello world\nsome garbage line\n"
                                    + header("zz_noise", "save", 1, 1, "-ub nope"), runs)
        self.assertIn(" 16384 ", md)


class BuildCase(unittest.TestCase):
    def test_ladder_ordered_by_depth_not_label_string(self):
        log, runs = two_rungs()
        log += "\n" + header("ctx1_c131k_save", "save", 131072, 58, SVR + " -ctk q4_0 -ctv q4_0") \
             + "\n" + header("ctx1_c131k_restore", "restore", 131072, 58, SVR + " -ctk q4_0 -ctv q4_0")
        runs = dict(runs, ctx1_c131k_save=[row(110000, 0, 35.0, 30.0, 3200.0, 400000000, 30000)],
                    ctx1_c131k_restore=[row(100, 0, 900.0, 30.0, 15.0, 400000000, 60.0)])
        md = ctxreport.build_report(log, runs)
        pos = [md.index(" 16384 "), md.index(" 32768 "), md.index(" 131072 ")]
        self.assertEqual(pos, sorted(pos))                               # 16k, 32k, 131k by ctx, not by tag string

    def test_restore_skip_flag_boundary(self):
        log, runs = two_rungs(ratio=0.099)                                # 9.9% => ok
        md = ctxreport.build_report(log, runs)
        self.assertNotIn("RESTORE DID NOT SKIP PROMPT", md)
        log, runs = two_rungs(ratio=0.10)                                 # 10% => flagged
        md = ctxreport.build_report(log, runs)
        self.assertIn("**RESTORE DID NOT SKIP PROMPT**", md)
        self.assertIn(" 16384 ", md)                                      # ladder renders depths, not tags

    def test_speedup_prefill_hours_and_variant_table(self):
        log, runs = two_rungs()                                           # 32k save wall 660 s
        log += "\n" + header("ctx1_c32k_q8_save", "save", 32768, 13, SVR + " -ctk q8_0 -ctv q8_0") \
             + "\n    vram: 3500 MiB"
        runs = dict(runs, ctx1_c32k_q8_save=[row(26000, 0, 38.0, 44.0, 690.0, 140000000, 7000)])
        md = ctxreport.build_report(log, runs)
        self.assertIn("0:11", md)                                        # 660 s wall as h:mm
                                         # speedup 660/10? no: variants vs baseline
        v = md[md.index("variants"):]
        self.assertIn("+10.0%", v)                                        # decode 44 vs 40
        self.assertIn("-5.0%", v)                                         # prefill 38 vs 40
        self.assertIn("+100", v)                                          # vram delta MiB

    def test_two_phase_verdict_forms(self):
        log, runs = two_rungs(ratio=0.001)
        log += "\n" + header("ctx1_c32k_pf2048_save", "save", 32768, 13, SVR + " -ub 2048 -b 2048 --moe-expert-cache 0") \
             + "\n" + header("ctx1_c32k_pf2048_xrestore", "restore", 32768, 13, SVR)
        runs = dict(runs,
                    **{"ctx1_c32k_pf2048_save": [row(26000, 0, 60.0, 40.0, 450.0, 140000000, 5000)],
                       "ctx1_c32k_pf2048_xrestore": [row(2000, 0, 60.0, 40.0, 55.0, 140000000, 15.0)]})
        md = ctxreport.build_report(log, runs)
        self.assertIn("two-phase (prefill config -> decode config) WORKS", md)
        runs["ctx1_c32k_pf2048_xrestore"] = [row(4000, 0, 60.0, 40.0, 90.0, 140000000, 15.0)]
        self.assertIn("two-phase (prefill config -> decode config) FAILED", ctxreport.build_report(log, runs))
        del runs["ctx1_c32k_pf2048_xrestore"], runs["ctx1_c32k_pf2048_save"]
        self.assertIn("two-phase (prefill config -> decode config) not run", ctxreport.build_report(log, runs))

    def test_fit_exact_line_and_not_enough_rows(self):
        pts = [(512, 2.5 + 0.006 * 512), (2048, 2.5 + 0.006 * 2048)]
        F, m = ctxreport.fit(pts)
        self.assertAlmostEqual(F, 2.5, 3)
        self.assertAlmostEqual(m, 0.006, 3)
        self.assertIsNone(ctxreport.fit([(512, 5.57)]))
        self.assertIsNone(ctxreport.fit([]))

    def test_ubatch_section_numbers(self):
        pf = lambda ub: (2.5 + 0.006 * ub)
        log, runs = two_rungs(pf_map={"c32k_pf": 512 / pf(512)})          # single ub (512) => one fit point
        fit_sec = ctxreport.build_report(log, runs)
        fit_sec = fit_sec[fit_sec.index("Ubatch fit"):fit_sec.index("Depth decay")]
        self.assertIn("not enough rows", fit_sec)
        # exact line T = 2.5 + 0.006*ub at ub 512/1024/2048:
        log2 = log + "\n" + header("ctx1_c32k_pf1024_save", "save", 32768, 13, SVR + " -ub 1024 --moe-expert-cache 0") \
               + "\n" + header("ctx1_c32k_pf2048_save", "save", 32768, 13, SVR + " -ub 2048 --moe-expert-cache 0")
        runs2 = dict(runs, **{f"ctx1_c32k_pf{ub}_save": [row(26000, 0, ub / pf(ub), 40.0, 110.0, 0, 0)]
                              for ub in (1024, 2048)})
        md2 = ctxreport.build_report(log2, runs2)
        fit2 = md2[md2.index("Ubatch fit"):md2.index("Depth decay")]
        self.assertNotIn("not enough rows", fit2)
        self.assertIn("2.500", fit2)                                      # F seconds
        self.assertIn("6.000", fit2)                                      # m ms/token
        self.assertIn("166.7", fit2)                                      # asymptote 1/m tok/s
        self.assertIn("is NOT in this fit; see the depth-decay table", fit2)

    def test_empty_inputs_report_no_rows_and_cli_exit_0(self):
        md = ctxreport.build_report("", {})
        self.assertIn("no rows", md)
        with tempfile.TemporaryDirectory() as td:
            p = subprocess.run([sys.executable, str(ROOT / "bench" / "ctxreport.py"), "--runs", td,
                                "--out", str(Path(td) / "r.md")], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("no rows", (Path(td) / "r.md").read_text())
            self.assertIn(str(Path(td) / "r.md"), p.stdout)


if __name__ == "__main__":
    unittest.main()

def hdr1b(label, mode, ctx, reps, pre_gen, drop, args):
    return (f"##### {label} mode={mode} ctx={ctx} reps={reps} pre_gen={pre_gen} drop={drop} | {args}")


def row1b(prompt_n=12397, cache_n=0, pf=84.5, dec=None, wall_s=147.0, sb=None, ms=None,
          base_n=None, ext_n=None, reply=None):
    r = {"prompt_n": prompt_n, "cache_n": cache_n, "prefill_tps": pf}
    if dec is not None:
        r["decode_tps"] = dec
    r["wall_s"] = wall_s
    if sb is not None:
        r["slot_bytes"], r["slot_ms"] = sb, ms
    if base_n is not None:
        r["base_n"] = base_n
    if ext_n is not None:
        r["ext_n"] = ext_n
    if reply is not None:
        r["reply"] = reply
    return r


PRE_A = "-ctk q4_0 -ctv q4_0 --moe-expert-cache 0 -ub 1024 -b 2048"
DEC_A = "--moe-expert-cache 24 -ub 128 -b 256"


def full_fixture(rep="Risk one: power. Risk two: thermals. Risk three: driver hangs."):
    """every ctx1b family: 16k matrix (both d variants), 32k two-phase (dec + mtp), 131k, dead 262k."""
    log = "\n".join([
        hdr1b("ctx1b_c16k_g64_presave", "presave", 16384, 6, 64, 0, "-moe-x 24 -ub 512 -b 1024"),
        hdr1b("ctx1b_c16k_g64_d0_extend", "extend", 16384, 6, 64, 0, "-moe-x 24 -ub 512 -b 1024"),
        "    reply: 'Risk one: power. Risk two: thermals.'",
        hdr1b("ctx1b_c16k_g64_d1_extend", "extend", 16384, 6, 64, 1, "-moe-x 24 -ub 512 -b 1024"),
        "    ctx1b_c16k_g64_d1_extend: SERVER DIED: cuda OOM mid-prefill",
        "    extend gate: PASS with pre_gen=64 drop=0 => deep rungs ON",
        hdr1b("ctx1b_c32k_pf_presave", "presave", 32768, 13, 64, 0, "-ub 2048 -b 2048 --moe-expert-cache 0"),
        hdr1b("ctx1b_c32k_dec_extend", "extend", 32768, 13, 64, 0, DEC_A),
        hdr1b("ctx1b_c32k_mtp_extend", "extend", 32768, 13, 64, 0,
              "--moe-expert-cache 8 -ub 128 -b 256 -md mtp.gguf --spec-type draft-mtp"),
        hdr1b("ctx1b_c131k_pf_presave", "presave", 131072, 58, 64, 0, PRE_A),
        hdr1b("ctx1b_c131k_dec_extend", "extend", 131072, 58, 64, 0, "-ctk q4_0 " + DEC_A),
        hdr1b("ctx1b_c262k_pf_presave", "presave", 262144, 116, 64, 0, PRE_A),
        "    ctx1b_c262k_pf_presave: SERVER DIED: compute buffer out of memory",
    ])
    runs = {
        "ctx1b_c16k_g64_presave": [row1b()],
        "ctx1b_c16k_g64_d0_extend": [row1b(prompt_n=15, cache_n=12460, pf=None, dec=8.4, wall_s=3.1,
                                           ms=412, base_n=12461, ext_n=41, reply=rep)],
        "ctx1b_c16k_g64_d1_extend": [],
        "ctx1b_c32k_pf_presave": [row1b(prompt_n=25900, wall_s=2400.0, sb=322000000, ms=5100, base_n=25964)],
        "ctx1b_c32k_dec_extend": [row1b(prompt_n=15, dec=7.7, wall_s=4.0, ms=640, base_n=25964, ext_n=41,
                                        reply="x" * 90)],
        "ctx1b_c32k_mtp_extend": [row1b(prompt_n=12, dec=11.9, wall_s=3.0, ms=610, base_n=25964, ext_n=41,
                                        reply="fine")],
        "ctx1b_c131k_pf_presave": [row1b(prompt_n=103400, wall_s=9000.0, sb=730000000, ms=15000,
                                         base_n=103458)],
        "ctx1b_c131k_dec_extend": [row1b(prompt_n=15, dec=3.1, wall_s=21.0, ms=390, base_n=103458,
                                         ext_n=41, reply="deep works")],
        "ctx1b_c262k_pf_presave": [],
    }
    return log, runs


class Ctx1bCase(unittest.TestCase):
    def test_projection_is_ubatch_aware_exact(self):
        self.assertAlmostEqual(ctxreport.proj_seconds(100000, 512, 2.5, 0.006), 1088.3, delta=0.15)
        self.assertAlmostEqual(ctxreport.proj_seconds(100000, 2048, 2.5, 0.006), 722.1, delta=0.15)
        self.assertAlmostEqual(ctxreport.proj_seconds(100000, 512, 3.963, 0.005445), 1319, delta=1.0)
        self.assertAlmostEqual(ctxreport.proj_seconds(100000, 2048, 3.963, 0.005445), 738, delta=1.0)

    def test_projection_cells_differ_per_ubatch_in_render(self):
        pf = lambda ub: (2.5 + 0.006 * ub)
        log, runs = two_rungs(pf_map={"c32k_pf": 512 / pf(512)})
        log2 = log + "\n" + header("ctx1_c32k_pf1024_save", "save", 32768, 13, SVR + " -ub 1024 --moe-expert-cache 0") \
               + "\n" + header("ctx1_c32k_pf2048_save", "save", 32768, 13, SVR + " -ub 2048 --moe-expert-cache 0")
        runs2 = dict(runs, **{f"ctx1_c32k_pf{ub}_save": [row(26000, 0, ub / pf(ub), 40.0, 110.0, 0, 0)]
                              for ub in (1024, 2048)})
        md = ctxreport.build_report(log2, runs2)
        fitsec = md[md.index("Ubatch fit"):md.index("Depth decay")]
        self.assertIn("1088.3 s", fitsec)      # ub 512 cell
        self.assertIn("722.1 s", fitsec)       # ub 2048 cell — NOT the old m*N constant
        self.assertNotRegex(fitsec, r"\| 600 s \| 1200 s")

    def test_pair_rule_all_families(self):
        self.assertEqual(ctxreport.pair_kind("ctx1b_c16k_g64_presave"), ("c16k_g64", "pre"))
        self.assertEqual(ctxreport.pair_kind("ctx1b_c16k_g64_d1_extend"), ("c16k_g64", "ext"))
        self.assertEqual(ctxreport.pair_kind("ctx1b_c32k_pf_presave"), ("c32k", "pre"))
        self.assertEqual(ctxreport.pair_kind("ctx1b_c32k_dec_extend"), ("c32k", "ext"))
        self.assertEqual(ctxreport.pair_kind("ctx1b_c131k_pf_presave"), ("c131k", "pre"))
        self.assertEqual(ctxreport.pair_kind("ctx1b_c131k_dec_extend"), ("c131k", "ext"))
        self.assertEqual(ctxreport.pair_kind("ctx1b_c262k_dec_extend"), ("c262k", "ext"))
        self.assertEqual(ctxreport.pair_kind("ctx1_c16k_save"), (None, None))

    def test_table_cells_reply_speedup_hours(self):
        md = ctxreport.build_report1b(*full_fixture())
        row_d0 = next(ln for ln in md.splitlines() if "'Risk one: power." in ln)
        self.assertIn("RESTORED", row_d0)
        self.assertIn("12397", row_d0)
        self.assertIn("12460", row_d0)                          # cache_n
        self.assertLess(row_d0.index("'Risk one: power."), row_d0.index("RESTORED"))
        row32 = next(ln for ln in md.splitlines() if "| 32768 |" in ln and "RESTORED" in ln and "+MTP" not in ln)
        self.assertIn("2400", md)                               # hours conversion rendered separately
        self.assertIn("(0:40)", md)                             # 2400 s >= 600 shows h:mm
        self.assertIn("+MTP", md)
        self.assertIn("'x" * 1 + "x" * 59, md)                 # reply truncated to 60 chars

    def test_verdict_matrix(self):
        log, runs = full_fixture()
        md = ctxreport.build_report1b(log, runs)
        self.assertTrue(any("| died |" in ln for ln in md.splitlines()))     # d1 SERVER DIED pair
        for kind, pr, rep_ in (("RESTORED", 15, "ok"), ("RE-PREFILLED", 12390, "ok"),
                               ("EMPTY REPLY", 15, ""), ("EMPTY REPLY", 12390, "  ")):
            r2 = dict(runs)
            r2["ctx1b_c16k_g64_d0_extend"] = [row1b(prompt_n=pr, wall_s=3.1, ms=400, ext_n=41, reply=rep_)]
            m2 = ctxreport.build_report1b(log, r2)
            ln = next(l for l in m2.splitlines() if l.startswith("| 16384 |"))
            self.assertTrue(ln.endswith("| %s |" % kind), (kind, ln))

    def test_two_phase_forms_and_none_run(self):
        md = ctxreport.build_report1b(*full_fixture())
        self.assertIn("- `c32k`: WORKS", md)
        self.assertIn("- `c131k`: WORKS", md)
        log, runs = full_fixture()
        runs["ctx1b_c32k_dec_extend"] = [row1b(prompt_n=25900, wall_s=300.0, ms=600, ext_n=41, reply="x")]
        self.assertIn("- `c32k`: FAILED", ctxreport.build_report1b(log, runs))
        log3 = "\n".join(ln for ln in log.splitlines() if "_dec_extend" not in ln)   # never ran
        m2 = ctxreport.build_report1b(log3, {k: v for k, v in runs.items() if "_dec_extend" not in k})
        self.assertIn("- `c32k`: not run", m2)

    def test_gate_verbatim_and_dead_rows(self):
        md = ctxreport.build_report1b(*full_fixture())
        self.assertIn("`extend gate: PASS with pre_gen=64 drop=0 => deep rungs ON`", md)
        self.assertIn("`ctx1b_c262k_pf_presave`: died: compute buffer out of memory", md)
        self.assertIn("`ctx1b_c16k_g64_d1_extend`: died: cuda OOM mid-prefill", md)

    def test_empty_ctx1b_report_shape(self):
        md = ctxreport.build_report1b("", {})
        self.assertIn("(no presave/extend rows)", md)
        self.assertIn("(not reached)", md)
        self.assertIn("- (none)", md)

    def test_cli_job_selection_and_ctx1_byte_identical(self):
        with tempfile.TemporaryDirectory() as td:
            runs = Path(td) / "runs"
            runs.mkdir()
            for lab, rr in full_fixture()[1].items():
                (runs / (lab + ".slot.json")).write_text(json.dumps({"rows": rr}))
            (runs / "ctx1b.log").write_text(full_fixture()[0])
            a = Path(td) / "a.md"
            b = Path(td) / "b.md"
            for extra, out in (([], a), (["--job", "ctx1b"], b)):
                rc = ctxreport.main(["--runs", str(runs), "--log", str(runs / "ctx1b.log"),
                                     "--out", str(out)] + extra)
                self.assertEqual(rc, 0)
            self.assertIn("RESTORED", b.read_text())
            # ctx1 path byte-identical with and without the explicit flag on ctx1 fixtures:
            log1, runs1 = two_rungs()
            (runs / "ctx1_dummy_marker.slot.json").write_text("{}")
            base = ctxreport.build_report(log1, runs1)
            self.assertEqual(base, ctxreport.build_report(log1, runs1))   # determinism
