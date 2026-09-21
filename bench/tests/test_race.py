"""offline tests for race.py: chain pairing, drops, the per-N counters, verdict words, exit 2."""
import io
import json
import sys, unittest
from pathlib import Path
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "qual"))
import race


def row(id_, set_, fin, tok, ok):
    return {"id": id_, "set": set_, "finish": fin, "tokens": tok, "ok": ok}


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def parse(out):
    """RACE set=... lines -> {(set, N): {field: int}}"""
    got = {}
    for ln in out.splitlines():
        f = ln.split()
        if len(f) > 3 and f[0] == "RACE" and f[1].startswith("set=") and f[2].startswith("N="):
            nn = int(f[2].split("=")[1])
            got[(f[1][4:], nn)] = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in f[3:]}
    return got


class HandCase(unittest.TestCase):
    # a: no chain ever finishes. b: two finishers tied at 60 (earlier file wins the tie, both ok).
    # c: the CHEAPEST finisher is wrong (40) and a slower chain is right (90) -> first_ok false, any_ok true.
    # d: cut in BASE, present in salt1, ABSENT from salt2 -> dropped, counted. x: not cut -> dropped, counted.
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        write(d / "base.jsonl", [row("a", "math_l5", "length", 100, False), row("b", "math_l5", "length", 100, False),
                                 row("c", "math_l5", "length", 100, False), row("d", "aime", "length", 100, False),
                                 row("x", "aime", "stop", 50, True)])
        write(d / "s1.jsonl", [row("a", "math_l5", "length", 100, False), row("b", "math_l5", "stop", 60, True),
                               row("c", "math_l5", "stop", 40, False), row("d", "aime", "stop", 70, True)])
        write(d / "s2.jsonl", [row("a", "math_l5", "length", 100, False), row("b", "math_l5", "stop", 60, True),
                               row("c", "math_l5", "stop", 90, True)])
        self.d = d

    def tearDown(self):
        self.tmp.cleanup()

    def runcap(self, args):
        buf = io.StringIO()
        rc = race.run(args, out=buf)
        return rc, buf.getvalue()

    def test_counts_drop_reasons_and_reflect_every_strategy(self):
        jj = self.d / "out.json"
        rc, out = self.runcap([str(self.d / "base.jsonl"), str(self.d / "s1.jsonl"), str(self.d / "s2.jsonl"), "--json", str(jj)])
        self.assertEqual(rc, 0)
        self.assertIn("analyzable=3 dropped_missing_in_salted=1 dropped_not_cut=1", out)
        g = parse(out)
        self.assertEqual(set(g), {("math_l5", 1), ("math_l5", 2), ("math_l5", 3), ("_all_", 1), ("_all_", 2), ("_all_", 3)})
        for nn in (1, 2, 3):
            self.assertEqual(g[("math_l5", nn)], g[("_all_", nn)])
        n1 = {"items": 3, "finished": 0, "first_ok": 0, "any_ok": 0, "majority_ok": 0, "race_tok": 300, "retry_tok": 300}
        self.assertEqual(g[("math_l5", 1)], n1)   # only the cut base chain: nothing finished, both strategies pay 100 each
        n2 = {"items": 3, "finished": 2, "first_ok": 1, "any_ok": 1, "majority_ok": 1, "race_tok": 400, "retry_tok": 500}
        self.assertEqual(g[("math_l5", 2)], n2)   # a pays 2x100 / 200; b 2x60 / 100+60; c 2x40 / 100+40 (wrong, so no first_ok)
        n3 = {"items": 3, "finished": 2, "first_ok": 1, "any_ok": 2, "majority_ok": 1, "race_tok": 600, "retry_tok": 600}
        self.assertEqual(g[("math_l5", 3)], n3)   # b tie at 60 ok; c now any_ok via the 90 chain, majority 1 of 2 is not strict
        self.assertIn("RACE_VERDICT seed", out)   # 2 of 3 items finishable >= 0.5
        jd = json.loads(jj.read_text())
        self.assertEqual(jd["verdict"], "seed")
        self.assertEqual(jd["dropped"], {"missing_in_salted": 1, "not_cut": 1})
        self.assertEqual([{k: r[k] for k in ("set", "N", "finished", "race_tok")} for r in jd["rows"]],
                         [{"set": s, "N": n, "finished": g[(s, n)]["finished"], "race_tok": g[(s, n)]["race_tok"]}
                          for s in ("math_l5", "_all_") for n in (1, 2, 3)])

    def test_stat_tie_goes_to_the_earlier_file(self):
        ch = [row("b", "s", "length", 100, False), row("b", "s", "stop", 60, True), row("b", "s", "stop", 60, False)]
        s = race.stat(ch, 3)
        self.assertEqual((s["finished"], s["first_ok"], s["any_ok"], s["majority_ok"]), (1, 1, 1, 0))
        self.assertEqual(s["race_tok"], 3 * 60)
        self.assertEqual(s["retry_tok"], 100 + 60)   # retry stops at the FIRST finished chain in file order


class VerdictWords(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(race.verdict(5, 10), "seed")      # >= 0.5
        self.assertEqual(race.verdict(10, 10), "seed")
        self.assertEqual(race.verdict(2, 10), "problem")   # <= 0.2
        self.assertEqual(race.verdict(0, 10), "problem")
        self.assertEqual(race.verdict(3, 10), "mixed")
        self.assertEqual(race.verdict(4, 10), "mixed")


class ExitTwo(unittest.TestCase):
    def test_bad_inputs_exit_two(self):
        tmp = tempfile.TemporaryDirectory()
        d = Path(tmp.name)
        buf = io.StringIO()
        self.assertEqual(race.run([str(d / "nope.jsonl"), str(d / "also-nope.jsonl")], out=buf), 2)   # unreadable
        write(d / "junk.jsonl", [])
        (d / "junk.jsonl").write_text("this is not jsonl {\n")
        self.assertEqual(race.run([str(d / "junk.jsonl"), str(d / "junk.jsonl")], out=buf), 2)        # unparsable
        write(d / "base.jsonl", [row("a", "s", "stop", 10, True)])
        write(d / "salt.jsonl", [row("a", "s", "stop", 10, True)])
        self.assertEqual(race.run([str(d / "base.jsonl"), str(d / "salt.jsonl")], out=buf), 2)        # nothing cut -> no items
        self.assertEqual(race.run([str(d / "base.jsonl")], out=buf), 2)                                # base only
        with self.assertRaises(SystemExit) as cm:
            race.run([])
        self.assertEqual(cm.exception.code, 2)                                                         # no files at all
        tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
