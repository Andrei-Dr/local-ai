"""offline tests for mon.py: CPU used by processes that are not part of the benchmark."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "box"))
import mon


class Ours(unittest.TestCase):
    def test_bench_processes_are_ours(self):
        for comm in ("llama-server", "llama-bench", "nvidia-smi", "perf"):
            self.assertTrue(mon.is_ours(comm), comm)

    def test_everything_else_is_foreign(self):
        for comm in ("node", "esbuild", "Runner.Worker", "postgres"):
            self.assertFalse(mon.is_ours(comm), comm)


class ForeignDelta(unittest.TestCase):
    def test_delta_counts_only_foreign_ticks(self):
        a = {1: ("llama-server", 1000, 5), 2: ("node", 50, 6)}
        b = {1: ("llama-server", 1500, 5), 2: ("node", 250, 6)}
        self.assertEqual(mon.foreign_delta(a, b), {"node": 200})

    def test_new_process_counts_from_zero(self):
        self.assertEqual(mon.foreign_delta({}, {7: ("esbuild", 90, 1)}), {"esbuild": 90})

    def test_pid_reuse_is_a_new_process(self):
        a = {2: ("node", 500, 10)}
        b = {2: ("node", 3, 99)}
        self.assertEqual(mon.foreign_delta(a, b), {"node": 3})

    def test_kernel_worker_renaming_itself_is_the_same_process(self):
        a = {9: ("kworker/1:2-wg-crypt-wg0", 90000, 4)}
        b = {9: ("kworker/1:2-mm_percpu_wq", 90002, 4)}
        self.assertEqual(mon.foreign_delta(a, b), {"kworker/1:2-mm_percpu_wq": 2})

    def test_same_comm_is_summed(self):
        a = {2: ("node", 0, 1), 3: ("node", 0, 2)}
        b = {2: ("node", 100, 1), 3: ("node", 40, 2)}
        self.assertEqual(mon.foreign_delta(a, b), {"node": 140})


class ForeignSummary(unittest.TestCase):
    def test_quiet_box(self):
        s = mon.foreign_summary([{"kworker": 1}, {}, {"sshd": 2}], hz=100)
        self.assertEqual(s["foreign_cpu_pct_max"], 2.0)
        self.assertFalse(s["foreign_cpu_flag"])

    def test_build_beside_the_benchmark_is_flagged_and_named(self):
        s = mon.foreign_summary([{"node": 350, "esbuild": 120}, {"node": 300}, {}], hz=100)
        self.assertEqual(s["foreign_cpu_pct_max"], 470.0)
        self.assertEqual(s["foreign_cpu_pct_avg"], 256.7)
        self.assertEqual(s["foreign_cpu_top"], "node")
        self.assertTrue(s["foreign_cpu_flag"])

    def test_no_samples(self):
        s = mon.foreign_summary([], hz=100)
        self.assertEqual(s["foreign_cpu_pct_avg"], 0.0)
        self.assertFalse(s["foreign_cpu_flag"])


if __name__ == "__main__":
    unittest.main()
