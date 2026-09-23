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


if __name__ == "__main__":
    unittest.main()
