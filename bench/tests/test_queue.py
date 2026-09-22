"""tests for bench/box/queue.sh — the runner against a scratch QDIR (GPU check off, zero idle sleeps, a BUSY_RE that never
matches). Needs bash + flock + setsid (the box; macOS has no flock -> the tests skip there).

Contract under test: a CLEAN stop (systemctl stop / shutdown = TERM to the whole control group) pauses the running job — it
goes back to pending and the try is NOT counted, so a long job interrupted by nightly downtime is never failed out; a CRASH
(runner killed without its trap) still counts a try and is bounded by MAX_TRIES."""
import os, shutil, signal, subprocess, tempfile, time, unittest
from pathlib import Path

QUEUE = Path(__file__).resolve().parent.parent / "box" / "queue.sh"
HAVE = all(shutil.which(x) for x in ("bash", "flock", "setsid"))


@unittest.skipUnless(HAVE, "needs bash + flock + setsid (run on the box)")
class QueueCase(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.env = dict(os.environ, QDIR=self.d, GPU_CHECK="0", IDLE_SLEEP="0", POLL="0.2",
                        BUSY_RE="no-such-process-xyzzy", MAX_TRIES="2")

    def q(self, *args):
        return subprocess.run(["bash", str(QUEUE), *args], env=self.env, capture_output=True, text=True)

    def row(self, jid):
        for line in Path(self.d, "queue.tsv").read_text().splitlines():
            f = line.split("\t")
            if f[0] == jid:
                return f
        raise KeyError(jid)

    def start_runner(self):
        return subprocess.Popen(["setsid", "bash", str(QUEUE), "run"], env=self.env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def wait_for(self, pred, secs=10):
        t = time.time() + secs
        while time.time() < t:
            if pred():
                return True
            time.sleep(0.05)
        return False

    def running(self, jid):
        return lambda: self.row(jid)[1] == "running" and Path(self.d, "started").exists()

    def test_clean_stop_pauses_without_counting_a_try(self):
        self.q("add", "long", "touch started; sleep 30")
        p = self.start_runner()
        self.assertTrue(self.wait_for(self.running("long")))
        os.killpg(p.pid, signal.SIGTERM)                 # what systemd does to the control group
        p.wait(10)
        f = self.row("long")
        self.assertEqual(f[1], "pending")
        self.assertEqual(f[2], "0")
        self.assertIn("PAUSE long", Path(self.d, "queue.log").read_text())

    def test_repeated_clean_stops_never_fail_the_job(self):
        # starts 1-3 run until stopped; start 4 completes
        self.q("add", "long", "n=$(( $(cat n 2>/dev/null || echo 0) + 1 )); echo $n > n; [ $n -ge 4 ] && exit 0; touch started; sleep 30")
        for _ in range(3):                               # 3 clean stops > MAX_TRIES 2
            Path(self.d, "started").unlink(missing_ok=True)
            p = self.start_runner()
            self.assertTrue(self.wait_for(self.running("long")))
            os.killpg(p.pid, signal.SIGTERM); p.wait(10)
            self.assertEqual(self.row("long")[1], "pending")
        p = self.start_runner()
        self.assertTrue(self.wait_for(lambda: self.row("long")[1] == "done"))
        os.killpg(p.pid, signal.SIGTERM); p.wait(10)
        self.assertEqual(self.row("long")[2], "1")       # only the completing start counted

    def test_crash_still_counts_and_is_bounded(self):
        self.q("add", "crashy", "touch started; sleep 30")
        for tries in ("1", "2"):                         # two crashes (no trap runs: SIGKILL = power loss)
            Path(self.d, "started").unlink(missing_ok=True)
            p = self.start_runner()
            self.assertTrue(self.wait_for(self.running("crashy")))
            os.killpg(p.pid, signal.SIGKILL); p.wait(10)
            self.assertEqual(self.row("crashy")[1:3], ["running", tries])
        p = self.start_runner()                          # MAX_TRIES 2 exhausted: the next start fails it
        self.assertTrue(self.wait_for(lambda: self.row("crashy")[1] == "failed"))
        os.killpg(p.pid, signal.SIGTERM); p.wait(10)
        self.assertEqual(self.row("crashy")[5], "interrupted")

    def test_failing_job_is_still_failed(self):
        self.q("add", "bad", "exit 3")
        p = self.start_runner()
        self.assertTrue(self.wait_for(lambda: self.row("bad")[1] == "failed"))
        os.killpg(p.pid, signal.SIGTERM); p.wait(10)
        self.assertEqual(self.row("bad")[5], "3")


if __name__ == "__main__":
    unittest.main()
