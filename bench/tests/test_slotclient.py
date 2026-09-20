"""tests for bench/box/slotclient.py — fake llama-server (chat + /slots/0 save/restore) on localhost;
asserts call ORDER per mode, request bodies, parsed rows/lines, HTTP-error exit."""
import json, os, subprocess, sys, tempfile, threading, unittest, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
CLIENT = BASE / "bench" / "box" / "slotclient.py"
REG = []                                              # (kind, body) per request, cleared per run
RESP = {"chat_prompt_n": 5120, "chat_cache_n": 0, "fail_slots": False}


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        p = urllib.parse.urlparse(self.path)
        kind = "chat" if p.path == "/v1/chat/completions" else urllib.parse.parse_qs(p.query)["action"][0]
        REG.append((kind, body))
        if kind != "chat" and RESP["fail_slots"]:
            self.send_response(500)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b("{}"[0:2]))
            return
        if kind == "chat":
            out = {"choices": [{"message": {"content": "ok"}}], "usage": {"completion_tokens": 64},
                   "timings": {"prompt_n": RESP["chat_prompt_n"], "cache_n": RESP["chat_cache_n"],
                               "prompt_per_second": 900.0, "predicted_per_second": 31.5}}
        elif kind == "save":
            out = {"success": True, "n_saved": 1, "n_written": 1234567, "timings": {"save_ms": 67.0}}
        else:
            out = {"success": True, "n_restored": 1, "n_read": 1234567, "timings": {"restore_ms": 12.5}}
        b = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


class SlotCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.doc = "DOC-SENTENCE " * 300
        cls.docfile = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
        cls.docfile.write(cls.doc)
        cls.docfile.close()
        cls.q = "\n\nBased on the text above, summarize its current status and the next steps in one short paragraph."
        cls.content = cls.doc + "\n" + cls.doc + cls.q

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        os.unlink(cls.docfile.name)

    def run_mode(self, mode, label="lbl", **extra):
        tmp = tempfile.TemporaryDirectory()
        env = dict(os.environ, URL=f"http://127.0.0.1:{self.server.server_address[1]}", OUT=tmp.name,
                   MODE=mode, PROMPT_FILE=self.docfile.name, REPS="2")
        env.update(extra)
        start = len(REG)
        p = subprocess.run([sys.executable, str(CLIENT), label], env=env, capture_output=True, text=True, timeout=60)
        d = json.loads((Path(tmp.name) / f"{label}.slot.json").read_text()) if p.returncode == 0 else None
        return p, d, REG[start:]

    def test_cold_one_chat_call(self):
        p, d, calls = self.run_mode("cold")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual([k for k, _ in calls], ["chat"])
        self.assertEqual(calls[0][1]["messages"][0]["content"], self.content)
        self.assertEqual(d["rows"], [d["rows"][0]])
        r = d["rows"][0]
        self.assertEqual((r["prompt_n"], r["prefill_tps"], r["decode_tps"]), (5120, 900.0, 31.5))
        self.assertIn("prompt_n 5120", p.stdout)
        self.assertIn("slot - bytes in - ms", p.stdout)
        self.assertEqual(d["mode"], "cold")
        self.assertEqual(d["prompt_chars"], len(self.content))

    def test_save_chats_then_saves_named_file(self):
        p, d, calls = self.run_mode("save")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual([k for k, _ in calls], ["chat", "save"])
        self.assertEqual(calls[1][1], {"filename": "lbl.slot"})
        r = d["rows"][0]
        self.assertEqual((r["slot_bytes"], r["slot_ms"]), (1234567, 67.0))
        self.assertIn("slot 1234567 bytes in 67.0 ms", p.stdout)

    def test_restore_first_same_prompt_shows_processed_near_zero(self):
        _, _, calls_save = self.run_mode("save")
        RESP["chat_prompt_n"], RESP["chat_cache_n"] = 40, 5080     # a working restore processes almost nothing
        self.addCleanup(RESP.update, {"chat_prompt_n": 5120, "chat_cache_n": 0})
        p, d, calls = self.run_mode("restore")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual([k for k, _ in calls], ["restore", "chat"])  # restore BEFORE the request
        self.assertEqual(calls[0][1], {"filename": "lbl.slot"})
        self.assertEqual(calls[1][1]["messages"][0]["content"], calls_save[0][1]["messages"][0]["content"])
        r = d["rows"][0]
        self.assertEqual((r["prompt_n"], r["cache_n"]), (40, 5080))
        self.assertIn("slot 1234567 bytes in 12.5 ms", p.stdout)

    def test_warm_two_calls_and_http_error_exit(self):
        p, d, calls = self.run_mode("warm")
        self.assertEqual([k for k, _ in calls], ["chat", "chat"])
        self.assertEqual(len(d["rows"]), 2)
        RESP["fail_slots"] = True
        self.addCleanup(RESP.update, {"fail_slots": False})
        p, d, _ = self.run_mode("save")
        self.assertEqual(p.returncode, 1)
        self.assertIn("HTTP 500 on /slots/0?action=save", p.stderr)


if __name__ == "__main__":
    unittest.main()
