"""offline tests for bench/qual/fetch.py selection logic — fake rows_fn carrying absolute offsets, no network."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "qual"))
import fetch

TOTALS = {"gsm8k": 1319, "humaneval": 164, "mmlu_pro": 12030}


def fake_rows(dataset, config, split, offset, length):
    key = "gsm8k" if "gsm8k" in dataset else "humaneval" if "humaneval" in dataset else "mmlu_pro"
    total = TOTALS[key]
    n = max(0, min(length, total - offset))
    out = []
    for i in range(offset, offset + n):  # every row carries its absolute offset
        if key == "gsm8k":
            out.append({"question": f"Q{i}", "answer": f"blurb #### {i}"})
        elif key == "humaneval":
            out.append({"task_id": f"HE/{i}", "prompt": "p", "test": "t", "entry_point": "e"})
        else:
            out.append({"question_id": i, "category": f"c{i % 10}", "question": f"M{i}", "options": ["a"], "answer": "A"})
    return out, total


def old_mmlu_ids():
    """Verbatim today's pre-flags logic over fake_rows: 14 pages x page[::20], category-interleaved."""
    _, total = fake_rows("TIGER-Lab/MMLU-Pro", "default", "test", 0, 1)
    stride = total // 14
    m = []
    for k in range(14):
        page, _ = fake_rows("TIGER-Lab/MMLU-Pro", "default", "test", k * stride + stride // 2, 100)
        for r in page[::20]:
            m.append({"id": f"mmlu_pro/{r['question_id']}", "category": r["category"]})
    seen, keyed = {}, []
    for it in m:
        seen[it["category"]] = seen.get(it["category"], 0) + 1
        keyed.append((seen[it["category"]], it["category"], it))
    return [it["id"] for _, _, it in sorted(keyed, key=lambda x: x[:2])]


class SelectCase(unittest.TestCase):
    def test_defaults_reproduce_today(self):
        sel = fetch.select_datasets(fake_rows)
        self.assertEqual([it["id"] for it in sel["gsm8k"]], [f"gsm8k/{i}" for i in range(50)])
        self.assertEqual([it["id"] for it in sel["humaneval"]], [f"HE/{i}" for i in range(0, 164, 4)])
        self.assertEqual([it["id"] for it in sel["mmlu_pro"]], old_mmlu_ids())

    def test_mmlu_rows_20_extends_as_nested_prefix(self):
        base = fetch.select_datasets(fake_rows)["mmlu_pro"]
        ext = fetch.select_datasets(fake_rows, mmlu_rows=20)["mmlu_pro"]
        ids_b, ids_x = [it["id"] for it in base], [it["id"] for it in ext]
        self.assertEqual(len(ext), 280)
        self.assertEqual(ids_x[:70], ids_b)
        self.assertEqual(len(set(ids_x)), 280)  # base∩extra empty, no duplicates anywhere

    def test_gsm8k_200_extends_as_nested_prefix(self):
        base = fetch.select_datasets(fake_rows)["gsm8k"]
        ext = fetch.select_datasets(fake_rows, gsm_n=200)["gsm8k"]
        self.assertEqual(len(ext), 200)
        self.assertEqual([it["id"] for it in ext[:50]], [it["id"] for it in base])


if __name__ == "__main__":
    unittest.main()
