"""tests for bench/closeout.py: the pure drift checks (box scripts vs repo mirror, finished jobs without a notes.md entry)."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import closeout


class HashDiffCase(unittest.TestCase):
    def test_parse_sha256sum(self):
        out = "aa  x.sh\nbb  y.py\nsha256sum: z.sh: No such file or directory\n"
        self.assertEqual(closeout.parse_sha256sum(out), {"x.sh": "aa", "y.py": "bb"})

    def test_differs_missing_and_box_only(self):
        d = closeout.hash_diff({"a.sh": "1", "b.sh": "2", "c.sh": "3"}, {"a.sh": "1", "b.sh": "9", "d.sh": "4"})
        self.assertEqual(d, {"differs": ["b.sh"], "missing_on_box": ["c.sh"], "box_only": ["d.sh"]})


class NotesCase(unittest.TestCase):
    ROWS = [{"id": "k2q6b", "status": "done", "ended": "2026-09-24 00:34:12"},
            {"id": "lq1", "status": "done", "ended": "2026-09-24 00:02:43"},
            {"id": "qx3", "status": "failed", "ended": "2026-09-23 18:23:06"},
            {"id": "old", "status": "done", "ended": "2026-09-01 10:00:00"},
            {"id": "lq2", "status": "running", "ended": "-"}]

    def test_recent_finished_jobs_without_a_notes_entry(self):
        notes = "### 2026-09-24 00:34 — k2q6b: ...\n- qx3 FAILED at 18:12\n"
        self.assertEqual(closeout.jobs_without_notes(self.ROWS, notes, since="2026-09-20"), ["lq1"])

    def test_id_must_match_as_a_word(self):
        notes = "k2q6b and lq10 and xqx3"
        self.assertEqual(closeout.jobs_without_notes(self.ROWS, notes, since="2026-09-20"), ["lq1", "qx3"])


if __name__ == "__main__":
    unittest.main()
