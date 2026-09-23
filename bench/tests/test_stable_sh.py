"""tests for bench/box/stable.sh against the repo's stable/stable.env: prefixed variables, env export that respects an arm's
own values, and per-context server flags."""
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def sh(script, **env):
    base = {"PATH": "/usr/bin:/bin", "STABLE_ENV_FILE": str(ROOT / "stable" / "stable.env"), "STABLE_MODELS": "/m"}
    r = subprocess.run(["bash", "-c", f"source {ROOT}/bench/box/stable.sh && {script}"], capture_output=True, text=True,
                       env={**base, **env})
    return r.returncode, r.stdout.strip(), r.stderr.strip()


class StableShCase(unittest.TestCase):
    def test_prefixed_vars_do_not_clash_with_a_jobs_model(self):
        rc, out, _ = sh('echo "$MODEL|$STABLE_MODEL_PATH|$STABLE_BUILD"', MODEL="/ai/models/variant.gguf")
        self.assertEqual(rc, 0)
        model, path, build = out.split("|")
        self.assertEqual(model, "/ai/models/variant.gguf")
        self.assertTrue(path.startswith("/m/") and path.endswith("K2q6-denseQ4K.gguf"), path)
        self.assertEqual(build, "/ai/src/llama.cpp-v2/build75")

    def test_export_env_keeps_an_arms_own_value(self):
        rc, out, _ = sh("stable_export_env && echo $GGML_SCHED_MOE_PREFETCH $GGML_CUDA_FA_TILE_MIN_BATCH $LLAMA_MTP_VOCAB_FILE",
                        GGML_SCHED_MOE_PREFETCH="0")
        self.assertEqual(out, "0 32 /m/mtp-Qwen3.6-35B-A3B-vocab49k.bin")

    def test_args_per_context_and_unknown_context(self):
        rc, out, _ = sh("stable_args 12288")
        self.assertEqual(rc, 0)
        self.assertTrue(out.startswith("-md /m/mtp-Qwen3.6-35B-A3B-Q4_0.gguf -ngl 999"), out)
        self.assertTrue(out.endswith("--moe-expert-cache 18 -b 4096"), out)
        rc, _, err = sh("stable_args 999")
        self.assertNotEqual(rc, 0)
        self.assertIn("no STABLE setting", err)


if __name__ == "__main__":
    unittest.main()


class StableServerCase(unittest.TestCase):
    def test_server_gets_stable_flags_then_arm_flags_and_env(self):
        import tempfile
        d = Path(tempfile.mkdtemp())
        (d / "bin").mkdir()
        (d / "bin" / "llama-server").write_text('#!/bin/sh\necho "ARGS $*"\necho "ENV $GGML_SCHED_MOE_PREFETCH $LEDGER_STABLE"\n')
        (d / "bin" / "llama-server").chmod(0o755)
        rc, out, err = sh(f"stable_server 4096 t1 --moe-expert-cache 20 && wait $STABLE_PID && cat {d}/server_t1.log",
                          BUILD=str(d), STABLE_LOGDIR=str(d), GGML_SCHED_MOE_PREFETCH="0")
        self.assertEqual(rc, 0, err)
        args = out.splitlines()[0]
        self.assertIn("-m /m/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-K2q6-denseQ4K.gguf -c 4096", args)
        self.assertTrue(args.endswith("--moe-expert-cache 21 -b 2048 --moe-expert-cache 20"), args)
        self.assertEqual(out.splitlines()[1], "ENV 0 2026-09-24")


class ReviewFixShCase(unittest.TestCase):
    def test_stable_server_does_not_leak_ledger_stable_into_the_job_shell(self):
        import tempfile
        d = Path(tempfile.mkdtemp())
        (d / "bin").mkdir()
        (d / "bin" / "llama-server").write_text('#!/bin/sh\necho "SERVER $LEDGER_STABLE"\n')
        (d / "bin" / "llama-server").chmod(0o755)
        rc, out, err = sh(f'stable_server 4096 t2 && wait $STABLE_PID && cat {d}/server_t2.log && echo "SHELL [$LEDGER_STABLE]"',
                          BUILD=str(d), STABLE_LOGDIR=str(d))
        self.assertEqual(out.splitlines(), ["SERVER 2026-09-24", "SHELL []"], err)

    def test_serve_sh_keeps_a_model_override(self):
        import tempfile
        d = Path(tempfile.mkdtemp())
        (d / "bin").mkdir()
        (d / "models").mkdir()
        (d / "bin" / "llama-server").write_text('#!/bin/sh\necho "$*"\n')
        (d / "bin" / "llama-server").chmod(0o755)
        for f in ("other.gguf", "mtp-Qwen3.6-35B-A3B-Q4_0.gguf"):
            (d / "models" / f).write_text("x")
        r = subprocess.run([str(ROOT / "stable" / "serve.sh")], capture_output=True, text=True,
                           env={"PATH": "/usr/bin:/bin", "LLAMA_BIN": str(d / "bin"), "MODELS": str(d / "models"), "MODEL": "other.gguf"})
        self.assertIn(f"-m {d}/models/other.gguf", r.stdout, r.stderr)
