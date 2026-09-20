"""tests for specclient.py env overrides / text fields / LONG=1 prompt, and for textdiff.py.
Run: ~/dev/.venv/bin/python -m unittest discover -s bench/tests -v -- stdlib only, localhost only."""
import hashlib, json, os, subprocess, sys, tempfile, threading, unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
SPECC = BASE / "bench" / "box" / "specclient.py"
TEXTDIFF = BASE / "bench" / "textdiff.py"
DOC = BASE / "bench" / "box" / "w4_doc.txt"
CONTENT = "The second train catches the first at 6pm. " + "x" * 200
TIMINGS = {"predicted_per_second": 42.5, "prompt_per_second": 311.0, "prompt_n": 25, "draft_n": 4, "draft_n_accepted": 3}
POSTED = []

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        POSTED.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        body = json.dumps({"choices": [{"message": {"content": CONTENT}}],
                           "usage": {"completion_tokens": 8}, "timings": TIMINGS}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a):
        pass

class ClientCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = DOC.read_text(encoding="utf-8")
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        POSTED.clear()
        self.tmp = tempfile.TemporaryDirectory()
        env = dict(os.environ, URL=f"http://127.0.0.1:{self.server.server_address[1]}/v1/chat/completions",
                   OUT=self.tmp.name, EDIT="1", LONG="1")
        self.proc = subprocess.run([sys.executable, str(SPECC), "unittest", "0"], env=env,
                                   capture_output=True, text=True, timeout=60)
        self.addCleanup(self.tmp.cleanup)

    def test_four_rows_fields_and_long_prompt(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr)
        self.assertEqual(len(self.proc.stdout.splitlines()), 4, self.proc.stdout)
        d = json.loads((Path(self.tmp.name) / "unittest.client.json").read_text(encoding="utf-8"))
        self.assertEqual([r["prompt"] for r in d["rows"]], ["code", "reason", "edit", "long"])
        sha = hashlib.sha256(CONTENT.encode("utf-8")).hexdigest()
        for r in d["rows"]:
            self.assertEqual(r["text"], CONTENT)
            self.assertEqual(r["text_sha256"], sha)
            self.assertEqual(r["text_head"], CONTENT[:80])
        self.assertEqual(len(POSTED), 4)
        long_msg = POSTED[3]["messages"][0]["content"]
        self.assertIn(self.doc, long_msg)
        self.assertTrue(long_msg.startswith("Read the document below."))
        self.assertLess(len(self.doc), 9000)

class VarHandler(BaseHTTPRequestHandler):
    """Timings cycle through SEQ by call index (warmup calls consume an index too); content alternates
    when VAR['rotate'] is set, so text_stable can be proven both ways."""
    SEQ = [10.0, 20.0, 30.0]
    CONTENTS = ["A" * 210, "B" * 210]
    VAR = {"rotate": False}

    def do_POST(self):
        c = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        POSTED.append(c)
        i = len(POSTED) - 1
        text = self.CONTENTS[i % 2] if self.VAR["rotate"] else self.CONTENTS[0]
        body = json.dumps({"choices": [{"message": {"content": text}}],
                           "usage": {"completion_tokens": 8},
                           "timings": {"predicted_per_second": self.SEQ[i % 3], "prompt_per_second": self.SEQ[i % 3],
                                       "prompt_n": 25, "draft_n": 4, "draft_n_accepted": 3}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class RepeatCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), VarHandler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def run_client(self, **extra):
        POSTED.clear()
        tmp = tempfile.TemporaryDirectory()
        env = dict(os.environ, URL=f"http://127.0.0.1:{self.server.server_address[1]}/v1/chat/completions",
                   OUT=tmp.name, EDIT="1", LONG="1", **extra)
        p = subprocess.run([sys.executable, str(SPECC), "unittest", "0"], env=env, capture_output=True, text=True,
                           timeout=120)
        self.addCleanup(tmp.cleanup)
        d = json.loads((Path(tmp.name) / "unittest.client.json").read_text(encoding="utf-8")) if p.returncode == 0 else None
        return p, d

    def test_repeats_means_lists_warmup_text_stable(self):
        VarHandler.VAR["rotate"] = True
        self.addCleanup(VarHandler.VAR.update, {"rotate": False})
        p, d = self.run_client(REPEATS="3", WARMUP="1")
        self.assertEqual(p.returncode, 0, p.stderr)
        # 1 warmup + 4 kinds x 3 repeats
        self.assertEqual(len(POSTED), 13)
        self.assertEqual(POSTED[0]["max_tokens"], 32)                      # throwaway tiny cap...
        self.assertIn("LRU cache", POSTED[0]["messages"][0]["content"])     # ...of the FIRST prompt
        self.assertEqual(POSTED[1]["max_tokens"], 200)                      # recorded runs use GEN
        rows = {r["prompt"]: r for r in d["rows"]}
        for j, kind in enumerate(("code", "reason", "edit", "long")):
            r = rows[kind]
            # call indexes per kind fall on (1,2,3)%3 etc -> same [20,30,10] rotation for every kind
            self.assertEqual(r["decode_tps_runs"], [20.0, 30.0, 10.0], kind)
            self.assertEqual(r["decode_tps"], 20.0, kind)
            self.assertEqual(r["prefill_tps_runs"], [20.0, 30.0, 10.0], kind)
            self.assertEqual(r["tokens"], 24, kind)                         # summed, not averaged
            self.assertEqual(r["draft_n"], 12, kind)
            self.assertEqual(r["acceptance"], 0.75, kind)
            self.assertFalse(r["text_stable"], kind)                        # consecutive runs alternate contents
            # this kind's first recorded call sits at global index 1+3*j -> alternating letter per kind
            self.assertEqual(r["text"], VarHandler.CONTENTS[(1 + 3 * j) % 2], kind)
            self.assertEqual(r["text_sha256"], hashlib.sha256(VarHandler.CONTENTS[(1 + 3 * j) % 2].encode()).hexdigest())
            self.assertEqual(len(r["wall_tps_runs"]), 3)
        for ln in p.stdout.splitlines():
            self.assertIn("| runs 3 spread 100.0%", ln)                     # (30-10)/20

    def test_repeat1_static_is_compatible(self):
        VarHandler.VAR["rotate"] = False
        p, d = self.run_client(REPEATS="2")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(len(POSTED), 8)                                    # no warmup calls
        r = d["rows"][0]
        self.assertEqual(r["decode_tps_runs"], [10.0, 20.0])                  # no warmup: calls 0,1
        self.assertEqual(r["decode_tps"], 15.0)
        self.assertTrue(r["text_stable"])
        self.assertIn("| runs 2 spread 66.7%", p.stdout.splitlines()[0])       # (20-10)/15
        self.assertEqual(d["rows"][0]["tokens"], 16)


class TextdiffCase(unittest.TestCase):
    def run_diff(self, ra, rb):
        with tempfile.TemporaryDirectory() as td:
            pa, pb = Path(td) / "a.client.json", Path(td) / "b.client.json"
            pa.write_text(json.dumps(ra), encoding="utf-8")
            pb.write_text(json.dumps(rb), encoding="utf-8")
            return subprocess.run([sys.executable, str(TEXTDIFF), str(pa), str(pb)], capture_output=True, text=True)

    def test_identical_exits_zero(self):
        rows = {"rows": [{"prompt": "code", "text_head": "abc", "text": "abcdef"}]}
        p = self.run_diff(rows, rows)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertEqual(p.stdout.strip(), "code  IDENTICAL")

    def test_divergent_reports_char_offset(self):
        ra = {"rows": [{"prompt": "code", "text_head": "abc", "text": "abcdef"}]}
        rb = {"rows": [{"prompt": "code", "text_head": "abc", "text": "abcXef"}]}
        p = self.run_diff(ra, rb)
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertEqual(p.stdout.strip(), "code  DIVERGES at char 3: 'def' vs 'Xef'")

if __name__ == "__main__":
    unittest.main()
