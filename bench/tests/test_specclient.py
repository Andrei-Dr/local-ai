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
