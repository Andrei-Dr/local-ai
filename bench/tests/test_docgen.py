"""tests for bench/docgen.py: generated-block replacement, the series.toml <-> mainline-series 1:1 check, and the renders that
read stable/stable.env. Synthetic inputs only."""
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import docgen

ENV = '''# comment
MODEL="m.gguf"
MTP_HEAD="h.gguf"
MTP_VOCAB="v.bin"
SERVE_ENV="A=1 B=2"
SERVE_ARGS="-ngl 999 -fa on"
SERVE_CTX_DEFAULT="4096"
SERVE_CTX_4096="--moe-expert-cache 21 -b 2048"
SERVE_CTX_12288="--moe-expert-cache 18 -b 4096"
LLAMA_CPP_REPO="https://example/llama.cpp"
LLAMA_CPP_BASE="abc1234"
LLAMA_CPP_SERIES="0001-0002"
CMAKE_FLAGS="-DGGML_CUDA=ON"
STABLE_SINCE="2026-09-24"
'''


class BlockCase(unittest.TestCase):
    def test_replaces_only_between_markers(self):
        t = "a\n" + docgen.begin("x") + "\nold\n" + docgen.end("x") + "\nb\n"
        out = docgen.replace_block(t, "x", "new")
        self.assertEqual(out, "a\n" + docgen.begin("x") + "\nnew\n" + docgen.end("x") + "\nb\n")

    def test_missing_markers_is_an_error(self):
        with self.assertRaises(docgen.DocgenError):
            docgen.replace_block("no markers here", "x", "new")


class EnvCase(unittest.TestCase):
    def test_parse_and_serving_command_per_context(self):
        env = docgen.parse_env(ENV)
        self.assertEqual(env["MODEL"], "m.gguf")
        cmd = docgen.serving_command(env, "12288")
        self.assertEqual(cmd, "A=1 B=2 llama-server -m m.gguf -md h.gguf -c 12288 -ngl 999 -fa on --moe-expert-cache 18 -b 4096 "
                              "(+ LLAMA_MTP_VOCAB_FILE=v.bin)")


class EnvStrictCase(unittest.TestCase):
    def test_a_line_bash_reads_but_the_parser_cannot_is_an_error(self):
        with self.assertRaises(docgen.DocgenError):
            docgen.parse_env('A="1"   # trailing comment\n')


class ServedStepsCase(unittest.TestCase):
    STEPS = [("LEGACY", "/old", "k2.gguf"), ("STABLE", "/new", "k2q6.gguf")]

    def test_current_stable_is_the_last_step(self):
        self.assertEqual(docgen.check_served_steps({"STABLE_BOX_BUILD": "/new", "MODEL": "k2q6.gguf"}, self.STEPS), [])

    def test_promotion_without_a_new_step_is_reported(self):
        errs = docgen.check_served_steps({"STABLE_BOX_BUILD": "/new", "MODEL": "k3.gguf"}, self.STEPS)
        self.assertTrue(errs and "SERVED_STEPS" in errs[0], errs)


class SeriesCase(unittest.TestCase):
    def series(self, files, entries):
        d = Path(tempfile.mkdtemp())
        for f in files:
            (d / f).write_text("patch")
        return docgen.check_series(d, [{"file": e, "theme": "cache"} for e in entries], {"cache": {}})

    def test_one_to_one_passes(self):
        self.assertEqual(self.series(["0001-a.patch", "0002-b.patch"], ["0001-a.patch", "0002-b.patch"]), [])

    def test_patch_without_entry_and_entry_without_patch_are_reported(self):
        errs = self.series(["0001-a.patch", "0002-b.patch"], ["0001-a.patch", "0003-c.patch"])
        self.assertTrue(any("0002-b.patch" in e and "no entry" in e for e in errs), errs)
        self.assertTrue(any("0003-c.patch" in e and "no patch file" in e for e in errs), errs)

    def test_unknown_theme_is_reported(self):
        d = Path(tempfile.mkdtemp())
        (d / "0001-a.patch").write_text("p")
        errs = docgen.check_series(d, [{"file": "0001-a.patch", "theme": "nope"}], {"cache": {}})
        self.assertTrue(any("theme" in e for e in errs), errs)



QUEUE = ("old\tdone\t1\t2026-09-19 23:35:29\t2026-09-19 23:40:51\t0\tbash /ai/bench/old.sh > old.log 2>&1\n"
         "fin\tfailed\t1\t2026-09-23 18:07:01\t2026-09-23 18:23:06\t1\tbash /ai/bench/fin.sh > fin.log 2>&1\n"
         "run\trunning\t1\t2026-09-24 00:35:12\t-\t-\tbash /ai/bench/run.sh x > run.log 2>&1; grep -q RUN_DONE run.log\n"
         "nxt\tpending\t0\t-\t-\t-\tbash /ai/bench/nxt.sh > nxt.log 2>&1\n")
SCRIPTS = {"run.sh": "#!/bin/bash\n# RUN: does the thing at depth? (long explanation\n# continues)\nsource x\n",
           "nxt.sh": "#!/bin/bash\nset -e\n# NXT — next measurement\n"}


class QueueCase(unittest.TestCase):
    def test_parse_and_script_purpose(self):
        rows = docgen.parse_queue(QUEUE)
        self.assertEqual([r["id"] for r in rows], ["old", "fin", "run", "nxt"])
        self.assertEqual(docgen.script_name(rows[2]["cmd"]), "run.sh")
        self.assertEqual(docgen.purpose(SCRIPTS["run.sh"]), "RUN: does the thing at depth? (long explanation")
        self.assertEqual(docgen.purpose(SCRIPTS["nxt.sh"]), "NXT — next measurement")

    def test_board_lists_running_then_pending_then_recent_finished(self):
        md = docgen.queue_board(docgen.parse_queue(QUEUE), SCRIPTS, recent=1)
        self.assertLess(md.index("`run`"), md.index("`nxt`"))
        self.assertIn("| running | `run` | 2026-09-24 00:35 | RUN: does the thing at depth?", md)
        self.assertIn("`fin` failed (rc 1) 2026-09-23 18:23", md)
        self.assertNotIn("`old`", md)


class LinkCase(unittest.TestCase):
    def test_broken_markdown_link_and_missing_repo_path(self):
        d = Path(tempfile.mkdtemp())
        (d / "docs").mkdir()
        (d / "ok.md").write_text("x")
        (d / "docs" / "a.md").write_text("[fine](../ok.md) [gone](../RESUME.md) [web](https://x.y/z) `bench/box/k.sh` `ok.md`\n")
        errs = docgen.broken_links(d, ["docs/a.md"], path_check=["docs/a.md"])
        self.assertTrue(any("RESUME.md" in e for e in errs), errs)
        self.assertTrue(any("bench/box/k.sh" in e for e in errs), errs)
        self.assertFalse(any("ok.md" in e and "gone" not in e and "RESUME" not in e for e in errs), errs)


class HeadlineCase(unittest.TestCase):
    def test_headline_from_arc_medians(self):
        rows = [{"name": "LEGACY", "code": 56.1, "reason": 56.4, "edit": 60.6, "pf93": 47.4, "dec93": 47.5},
                {"name": "STABLE", "code": 64.3, "reason": 65.4, "edit": 70.4, "pf93": 510.7, "dec93": 55.0}]
        self.assertEqual(docgen.headline(rows), "prompt reading ~511 tokens/s at 9.3k tokens (LEGACY 47, 10.8x); writing "
                                                "55-70 tokens/s (64-70 on short prompts, 55 after a 9.3k-token prompt)")



class ReviewFixCase(unittest.TestCase):
    def test_headline_without_measurements_does_not_crash(self):
        self.assertIn("no complete measurements", docgen.headline([{"name": "A", "pf93": None}, {"name": "B", "pf93": None}]))

    def test_serve_env_tokens_must_be_key_value(self):
        self.assertEqual(docgen.check_env({"SERVE_ENV": "A=1 B_2=x"}), [])
        self.assertTrue(docgen.check_env({"SERVE_ENV": "A=1 oops"}))

    def test_embedded_quote_rejected_like_the_box_parser(self):
        with self.assertRaises(docgen.DocgenError):
            docgen.parse_env('A="x"y"\n')

    def test_other_contexts_listed_exactly(self):
        env = docgen.parse_env(ENV + 'SERVE_CTX_14096="--x"\n')
        self.assertIn("At `-c 14096`", docgen.serving_block(env))
        self.assertIn("At `-c 12288`", docgen.serving_block(env))


if __name__ == "__main__":
    unittest.main()
