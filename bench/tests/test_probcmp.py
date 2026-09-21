"""offline tests for probcmp: per-token logprob drift between two decode arms."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "box"))
import probcmp


def tok(i, top):
    return {"id": i, "top": [list(x) for x in top]}


class Compare(unittest.TestCase):
    def test_identical_runs(self):
        a = [tok(5, [(5, -0.1), (7, -2.5)]), tok(9, [(9, -0.01), (2, -5.0)])]
        r = probcmp.compare(a, a)
        self.assertTrue(r["identical"])
        self.assertEqual((r["tokens"], r["same_prefix"], r["pairs"], r["max_dlogprob"]), (2, 2, 4, 0.0))

    def test_drift_is_measured_on_shared_candidates_only(self):
        a = [tok(5, [(5, -0.10), (7, -2.5)])]
        b = [tok(5, [(5, -0.13), (8, -2.4)])]
        r = probcmp.compare(a, b)
        self.assertTrue(r["identical"])
        self.assertEqual(r["pairs"], 1)
        self.assertAlmostEqual(r["max_dlogprob"], 0.03, places=6)

    def test_stops_at_the_first_divergent_token(self):
        a = [tok(1, [(1, -0.1)]), tok(2, [(2, -0.7), (3, -0.8)]), tok(4, [(4, -0.1)])]
        b = [tok(1, [(1, -0.1)]), tok(3, [(2, -0.8), (3, -0.7)]), tok(9, [(9, -9.0)])]
        r = probcmp.compare(a, b)
        self.assertFalse(r["identical"])
        self.assertEqual((r["tokens"], r["same_prefix"], r["pairs"]), (2, 1, 3))
        self.assertAlmostEqual(r["max_dlogprob"], 0.1, places=6)

    def test_length_mismatch_is_not_identical(self):
        a = [tok(1, [(1, -0.1)])]
        self.assertFalse(probcmp.compare(a, a + a)["identical"])


if __name__ == "__main__":
    unittest.main()
