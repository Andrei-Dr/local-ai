"""tests for bench/box/spec4_analyze.py — synthetic nsys sqlite exports with hand-computed per-class growth and H6-H8."""
import json, sqlite3, subprocess, sys, tempfile, unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "box" / "spec4_analyze.py"
MS = 1_000_000  # ns
NAMES = {"gdn": "void gated_delta_net_cuda<128, false, true>(const float *)",
         "chain": "void mul_mat_vec_q<(ggml_type)10, 4, false, false, false>(const void *)",
         "dense": "void mul_mat_vec_q<(ggml_type)12, 4, false, false, false>(const void *)",
         "fa": "void flash_attn_ext_f16<128, 128, 4>(const char *)",
         "norm": "void rms_norm_f32<1024, true>(const float *)",
         "odd": "void mystery_kernel(float *)"}


def make_db(path, durs, launch_ms, n_steps=5, unassigned=False):
    """durs: class -> ms per step on the target stream (7); the draft stream (9) runs one FA kernel between steps"""
    c = sqlite3.connect(path)
    c.execute("create table StringIds (id integer primary key, value text)")
    c.execute("create table CUPTI_ACTIVITY_KIND_KERNEL (start int, end int, streamId int, correlationId int, demangledName int, shortName int)")
    c.execute("create table CUPTI_ACTIVITY_KIND_RUNTIME (start int, end int, correlationId int, nameId int, globalTid int)")
    ids = {k: i for i, k in enumerate(NAMES, start=1)}
    for k, i in ids.items():
        c.execute("insert into StringIds values (?, ?)", (i, NAMES[k]))
    c.execute("insert into StringIds values (100, 'cudaLaunchKernel_v7000')")
    t, corr = 0, 0

    def kern(stream, name, dur_ms):
        nonlocal t, corr
        corr += 1
        c.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?, ?, ?, ?)", (t, t + int(dur_ms * MS), stream, corr, ids[name], ids[name]))
        c.execute("insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?, ?, ?, 100, 1)", (t - 50_000, t - 50_000 + int(launch_ms * MS), corr))
        t += int(dur_ms * MS)

    kern(7, "gdn", 5.0); kern(9, "fa", 1.0)   # warm-up request, then the client's pause
    t += 2_000 * MS
    kern(7, "gdn", 9.0); kern(7, "dense", 9.0)  # prompt pass (dropped)
    kern(9, "fa", 1.0)
    for _ in range(n_steps):
        for k in ("gdn", "chain", "dense", "fa", "norm", "odd"):
            kern(7, k, durs[k])
            if unassigned and k == "chain":
                kern(11, "norm", 0.05)  # a stray stream inside the step: must not split it
        t += 1 * MS
        kern(9, "fa", 1.0)  # MTP catch-up / next draft
        t += 1 * MS
    c.commit()
    c.close()


class Spec4Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self):
        p = subprocess.run([sys.executable, str(TOOL), str(self.d / "n1.sqlite"), str(self.d / "n3.sqlite"), "--json",
                            str(self.d / "o.json")], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        return p.stdout, json.loads((self.d / "o.json").read_text())

    def test_gdn_dominates(self):
        # n1 per step 1.0 / 1.0 / 2.0 / 0.5 / 0.5 / 0.1 = 5.1 ms, n3 3.0 / 2.0 / 2.4 / 0.7 / 0.6 / 0.1 = 8.8 ms -> busy +1.85 ms per
        # position, GDN +1.0 (54%) -> H7 TRUE; mapped growers 1.85 = 100% -> H6 TRUE; launch 6 x 0.01 -> 6 x 0.02 ms: +0.03 -> H8 FALSE
        make_db(self.d / "n1.sqlite", {"gdn": 1.0, "chain": 1.0, "dense": 2.0, "fa": 0.5, "norm": 0.5, "odd": 0.1}, 0.01, unassigned=True)
        make_db(self.d / "n3.sqlite", {"gdn": 3.0, "chain": 2.0, "dense": 2.4, "fa": 0.7, "norm": 0.6, "odd": 0.1}, 0.02)
        out, r = self.run_tool()
        self.assertEqual(r["n1"]["steps"], 5)
        self.assertAlmostEqual(r["n1"]["busy_ms"], 5.1, places=6)
        self.assertAlmostEqual(r["busy_growth_per_position"], 1.85, places=6)
        g = r["class_growth_per_position"]
        self.assertAlmostEqual(g["GDN (gated delta rule + conv)"], 1.0, places=6)
        self.assertAlmostEqual(g["cache-chain mul_mat_id"], 0.5, places=6)
        self.assertAlmostEqual(g["dense mul_mat"], 0.2, places=6)
        self.assertIn("UNMAPPED", out)
        self.assertIn("mystery_kernel", out)
        self.assertTrue(r["h6"].startswith("TRUE"))
        self.assertTrue(r["h7"].startswith("TRUE (GDN 54%"))
        self.assertTrue(r["h8"].startswith("FALSE (host launch +0.030"))

    def test_gdn_not_the_target(self):
        # GDN +0.1 of +1.3 ms per position (8%) -> H7 FALSE, top grower = the cache chain; launch +1.6 ms per position -> H8 TRUE
        make_db(self.d / "n1.sqlite", {"gdn": 1.0, "chain": 1.0, "dense": 2.0, "fa": 0.5, "norm": 0.5, "odd": 0.1}, 0.1)
        make_db(self.d / "n3.sqlite", {"gdn": 1.2, "chain": 3.0, "dense": 2.4, "fa": 0.5, "norm": 0.5, "odd": 0.1}, 0.6333333)
        _, r = self.run_tool()
        self.assertTrue(r["h7"].startswith("FALSE (GDN 8%: not the target; the top grower is cache-chain mul_mat_id)"))
        self.assertTrue(r["h8"].startswith("TRUE"))


if __name__ == "__main__":
    unittest.main()
