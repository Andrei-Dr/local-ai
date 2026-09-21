"""offline tests for the HARD quality sets (aime, math_l5, humaneval_plus): number parsing, scoring, selection."""
import sys, unittest
from fractions import Fraction
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "qual"))
import fetch, qual


class ParseNumber(unittest.TestCase):
    def test_plain_forms(self):
        for s, want in [("204", 204), ("-7", -7), ("3.5", Fraction(7, 2)), ("1,000", 1000), ("$42$", 42), ("17.", 17),
                        ("3/4", Fraction(3, 4)), ("\\frac{3}{4}", Fraction(3, 4)), ("-\\dfrac{1}{2}", Fraction(-1, 2)),
                        ("\\boxed{\\frac{14}{3}}", Fraction(14, 3)), ("\\tfrac{-5}{8}", Fraction(-5, 8))]:
            self.assertEqual(qual.parse_number(s), Fraction(want), s)

    def test_rejects_non_numbers(self):
        for s in ["\\left( 3, \\frac{\\pi}{2} \\right)", "2\\sqrt{3}", "x+1", "(3, 4)", "90^\\circ", "", "\\frac{1}{0}", "1/0", "\\text{Evelyn}"]:
            self.assertIsNone(qual.parse_number(s), s)


class Scoring(unittest.TestCase):
    def test_aime_last_answer_line_wins(self):
        it = {"gold": "204"}
        self.assertTrue(qual.score("aime", it, "so maybe 17.\nAnswer: 17\nwait, recount.\nAnswer: 204"))
        self.assertFalse(qual.score("aime", it, "Answer: 204\nActually no.\nAnswer: 205"))

    def test_aime_boxed_fallback_and_leading_zeros(self):
        self.assertTrue(qual.score("aime", {"gold": "73"}, "thus $\\boxed{073}$"))
        self.assertTrue(qual.score("aime", {"gold": "73"}, "**Answer:** 073"))
        self.assertFalse(qual.score("aime", {"gold": "73"}, "no final answer here, 73 appears mid-text only"))

    def test_math_fraction_and_decimal_equivalence(self):
        self.assertTrue(qual.score("math_l5", {"gold": "\\frac{14}{3}"}, "work...\nAnswer: 14/3"))
        self.assertTrue(qual.score("math_l5", {"gold": "\\frac{1}{2}"}, "Answer: $0.5$"))
        self.assertTrue(qual.score("math_l5", {"gold": "12"}, "so \\boxed{12}."))
        self.assertFalse(qual.score("math_l5", {"gold": "12"}, "Answer: 13"))
        self.assertFalse(qual.score("math_l5", {"gold": "12"}, "Answer: twelve"))

    def test_prompts_and_caps_exist(self):
        for k in ("aime", "math_l5", "humaneval_plus"):
            self.assertIn(k, qual.MAX_TOKENS)
        self.assertIn("Answer:", qual.prompt_for("aime", {"question": "Q?"}))
        self.assertIn("Answer:", qual.prompt_for("math_l5", {"question": "Q?"}))
        self.assertIn("```python", qual.prompt_for("humaneval_plus", {"prompt": "def f():\n"}))


MATH = [{"problem": f"p{i}", "answer": a, "level": lv, "unique_id": f"test/x/{i}.json", "subject": "s", "solution": ""}
        for i, (a, lv) in enumerate([("12", 5), ("\\sqrt{2}", 5), ("\\frac{1}{3}", 5), ("7", 3), ("(1,2)", 5), ("-4", 5), ("0.25", "5")])]


def fake_rows(dataset, config, split, offset, length):
    if "aime_2024" in dataset:
        rs = [{"id": str(60 + i), "problem": f"a24-{i}", "answer": str(100 + i)} for i in range(30)]
    elif "aime_2025" in dataset:
        rs = [{"id": str(i), "problem": f"a25-{i}", "answer": f"0{i}" if i < 10 else str(i)} for i in range(30)]
    elif "MATH-500" in dataset:
        rs = MATH
    else:
        rs = [{"task_id": f"HumanEval/{i}", "prompt": f"def f{i}():\n", "test": "def check(c): pass", "entry_point": f"f{i}"} for i in range(164)]
    return rs[offset:offset + length], len(rs)


class SelectHard(unittest.TestCase):
    def setUp(self):
        self.sel = fetch.select_hard(fake_rows, math_n=3)

    def test_aime_both_years_ids_and_integer_gold(self):
        a = self.sel["aime"]
        self.assertEqual(len(a), 60)
        self.assertEqual(a[0]["id"], "aime/2024-60")
        self.assertEqual(a[30]["id"], "aime/2025-0")
        self.assertEqual(a[30]["gold"], "0")            # "00" normalized
        self.assertEqual(len({x["id"] for x in a}), 60)

    def test_math_level5_numeric_only_prefix_stable(self):
        m = self.sel["math_l5"]
        self.assertEqual([x["question"] for x in m], ["p0", "p2", "p5"])   # sqrt, level 3, tuple dropped; order kept
        self.assertEqual([x["question"] for x in fetch.select_hard(fake_rows, math_n=2)["math_l5"]], ["p0", "p2"])
        self.assertTrue(m[0]["id"].startswith("math_l5/"))

    def test_humaneval_plus_same_tasks_as_base_every_4th(self):
        h = self.sel["humaneval_plus"]
        self.assertEqual(len(h), 41)
        self.assertEqual(h[1]["id"], "HumanEvalPlus/4")
        self.assertEqual(h[1]["entry_point"], "f4")


class ThinkingTrace(unittest.TestCase):
    def test_repeat_frac_separates_a_loop_from_live_reasoning(self):
        live = " ".join(f"step {i} gives value {i * i} so move on" for i in range(400))
        loop = "wait, let me recheck the sum of the digits again. " * 300
        self.assertLess(qual.repeat_frac(live), 0.2)
        self.assertGreater(qual.repeat_frac(loop), 0.9)
        self.assertEqual(qual.repeat_frac(""), 0.0)
        self.assertEqual(qual.repeat_frac("too short to judge"), 0.0)

    def test_trace_fields_keep_the_tail_not_the_whole_chain(self):
        t = qual.trace_fields("a" * 5000 + " THE END")
        self.assertEqual(t["think_chars"], 5008)
        self.assertTrue(t["think_tail"].endswith("THE END"))
        self.assertLessEqual(len(t["think_tail"]), qual.THINK_TAIL)
        self.assertEqual(qual.trace_fields(""), {"think_chars": 0, "think_tail": "", "repeat": 0.0})

    def test_aime_cap_follows_the_vendor_budget(self):
        self.assertGreaterEqual(qual.MAX_TOKENS["aime"] * 8, 38912)
        self.assertEqual(qual.MAX_TOKENS["math_l5"] * 8, 32768)  # the card's budget for normal queries


class ThinkingSampler(unittest.TestCase):
    def test_thinking_uses_the_vendor_sampler_with_a_per_item_seed(self):
        b = qual.request_body("p", 100, True, "aime/2024-60")
        self.assertEqual((b["temperature"], b["top_p"], b["top_k"], b["min_p"], b["presence_penalty"]), (1.0, 0.95, 20, 0.0, 1.5))
        c = qual.request_body("p", 100, True, "HumanEval/3", "humaneval_plus")
        self.assertEqual((c["temperature"], c["presence_penalty"]), (0.6, 0.0))
        self.assertEqual(b["seed"], qual.request_body("p", 100, True, "aime/2024-60")["seed"])
        self.assertNotEqual(b["seed"], qual.request_body("p", 100, True, "aime/2024-61")["seed"])
        self.assertTrue(0 <= b["seed"] < 2 ** 31)

    def test_non_thinking_stays_greedy(self):
        b = qual.request_body("p", 100, False, "gsm8k/1")
        self.assertEqual(b["temperature"], 0)
        self.assertNotIn("seed", b)

    def test_rows_from_another_sampler_are_rerun(self):
        greedy = {"id": "x", "set": "aime", "finish": "stop", "tokens": 10}
        self.assertFalse(qual.row_is_current(greedy, think=True))
        self.assertTrue(qual.row_is_current(dict(greedy, sampler=qual.THINK_SAMPLER_TAG), think=True))
        self.assertTrue(qual.row_is_current(greedy, think=False))
        cut = dict(greedy, sampler=qual.THINK_SAMPLER_TAG, finish="length", tokens=100)
        self.assertFalse(qual.row_is_current(cut, think=True))  # cut off under an older, smaller cap


class SetSpec(unittest.TestCase):
    def test_per_set_cap_overrides_the_global_limit(self):
        self.assertEqual(qual.parse_sets("math_l5,humaneval_plus,aime:15"), [("math_l5", 0), ("humaneval_plus", 0), ("aime", 15)])
        self.assertEqual(qual.parse_sets("gsm8k,aime:15", 50), [("gsm8k", 50), ("aime", 15)])


if __name__ == "__main__":
    unittest.main()
