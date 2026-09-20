"""tests for bench/qual/paired.py — pure functions on synthetic id->ok maps + two subprocess CLI cases
(empty intersection exit 2, --md rendering). Numbers are hand-computed: loses 5 / wins 3 of 200 =>
diff -1.00%, se 1.41, CI [-3.77, +1.77], exact McNemar p 0.7266 (= 2 * sum_{i<=3} C(8,i)/2^8)."""
import json, subprocess, sys, tempfile, unittest
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "bench" / "qual"))
import paired

BIG_B = {}                                  # helper-built maps below


def mk(**pairspec):
    """pairspec: id -> ("win"|"lose"|"both"|"neither") wrt BASE-side."""
    b, t = {}, {}
    for i, w in pairspec.items():
        bid = int(i[1:])
        b[bid] = {"set": "s", "ok": w in ("both", "lose")}
        t[bid] = {"set": "s", "ok": w in ("both", "win")}
    return b, t


class P(unittest.TestCase):
    def test_hand_computed_case(self):
        b, t = mk(**{f"q{i}": ("lose" if i < 5 else "win" if i < 8 else "both")
                     for i in range(200)})
        r = paired.compare(b, t, margin=2.0)
        s = r["ALL"]
        self.assertEqual((s["n"], s["loses"], s["wins"]), (200, 5, 3))
        self.assertEqual((s["diff"], s["se"]), (-1.0, 1.41))
        self.assertEqual(tuple(s["ci"]), (-3.77, 1.77))
        self.assertEqual(s["p"], 0.7266)
        self.assertAlmostEqual(s["discordance"], 0.04, 7)
        self.assertEqual(s["verdict"], "UNDECIDED")   # CI low -3.768 < -margin (-2)

    def test_duplicate_ids_keep_last_line(self):
        rows, bad = paired.loads_jsonl('{"id": 1, "set": "s", "ok": true}\n{"id": 1, "set": "s", "ok": false}\n')
        self.assertEqual(rows[1]["ok"], False)
        self.assertEqual(bad, 0)

    def test_ids_outside_intersection_are_counted_and_excluded(self):
        b = {1: {"set": "s", "ok": True}, 2: {"set": "s", "ok": True}, 3: {"set": "s", "ok": False}}
        t = {1: {"set": "s", "ok": True}, 4: {"set": "s", "ok": True}}
        r = paired.compare(b, t)
        self.assertEqual((r["only_base"], r["only_test"]), (2, 1))
        self.assertEqual(r["ALL"]["n"], 1)

    def test_zero_discordant(self):
        b, t = mk(**{f"q{i}": "both" for i in range(50)})
        s = paired.compare(b, t)["ALL"]
        self.assertEqual((s["p"], s["diff"], s["verdict"]), (1.0, 0.0, "NON-INFERIOR"))

    def test_verdicts_worse_and_undecided(self):
        b, t = mk(**{f"q{i}": ("lose" if i < 60 else "win" if i < 62 else "both") for i in range(200)})
        s = paired.compare(b, t, margin=2.0)["ALL"]
        self.assertEqual(s["verdict"], "WORSE")
        self.assertTrue(s["ci"][1] < 0)
        b2, t2 = mk(**{f"q{i}": ("lose" if i < 12 else "win" if i < 19 else "both") for i in range(200)})
        s2 = paired.compare(b2, t2, margin=2.0)["ALL"]
        self.assertEqual(s2["verdict"], "UNDECIDED")
        self.assertEqual(s2["needed_n"], 913)                    # ceil(1.96^2*0.095/0.02^2)
        s3 = paired.compare(b2, t2, margin=8.0)["ALL"]           # wider margin: CI lower -6.76 > -8
        self.assertEqual(s3["verdict"], "NON-INFERIOR")
        self.assertEqual(s3["needed_n"], 58)                      # ceil(1.96^2*0.095/0.08^2)

    def test_sets_filter(self):
        b = {1: {"set": "gsm8k", "ok": True}, 2: {"set": "humaneval", "ok": True}}
        t = {1: {"set": "gsm8k", "ok": True}, 2: {"set": "humaneval", "ok": False}}
        r = paired.compare(b, t, sets=["humaneval"])
        self.assertIn("humaneval", r["sets"])
        self.assertNotIn("gsm8k", r["sets"])
        self.assertEqual(r["ALL"]["n"], 1)

    def test_empty_intersection_exits_2(self):
        fb = Path(tempfile.mkdtemp()) / "b.jsonl"
        ft = Path(tempfile.mkdtemp()) / "t.jsonl"
        fb.write_text('{"id": 1, "set": "s", "ok": true}\n')
        ft.write_text('{"id": 2, "set": "s", "ok": true}\n')
        p = subprocess.run([sys.executable, str(BASE / "bench" / "qual" / "paired.py"), str(fb), str(ft)],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("INTERSECTION", (p.stdout + p.stderr).upper())

    def test_md_rendering_and_malformed_line_count(self):
        d = tempfile.mkdtemp()
        fb, ft = Path(d) / "b.jsonl", Path(d) / "t.jsonl"
        lines = "".join('{"id": %d, "set": "s", "ok": %s}\n' % (i, "true" if i < 190 else "false")
                        for i in range(200))
        tb = "".join('{"id": %d, "set": "s", "ok": %s}\n' % (i, "true") for i in range(200))
        fb.write_text(lines + "NOT JSON\n")
        ft.write_text(tb)
        p = subprocess.run([sys.executable, str(BASE / "bench" / "qual" / "paired.py"),
                            str(fb), str(ft), "--md"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("|", p.stdout)
        self.assertIn("malformed", p.stderr)                     # skipped with a count
        self.assertEqual(p.stdout.count("NON-INFERIOR") + p.stdout.count("UNDECIDED")
                        + p.stdout.count("WORSE"), 1)            # verdict exactly once (ALL)


if __name__ == "__main__":
    unittest.main()
