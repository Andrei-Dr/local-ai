"""tests for bench/box/slotclient.py — fake llama-server (chat, /completion, /apply-template, /tokenize,
/slots/0 save/restore) on localhost; asserts call ORDER per mode, request bodies, parsed rows/lines,
HTTP-error exit; probes the urlopen timeout kwarg by exec'ing the client in-process (subprocess tests
cannot intercept it)."""
import io, json, os, subprocess, sys, tempfile, threading, unittest, urllib.parse, urllib.request
from contextlib import ExitStack, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
CLIENT = BASE / "bench" / "box" / "slotclient.py"
REG = []                                              # (kind, body) per request, cleared per run
RESP = {"chat_prompt_n": 5120, "chat_cache_n": 0, "fail_slots": False}
SID = [101, 102, 103, 104, 105]           # tokens for the rendered prompt (parse_special true path)
ETOK = [900, 901]                          # tokens for the EXT text (parse_special false path)
RTOK = [555, 556, 557]                     # tokens generated during presave
RCOV = "X" * 200                           # /completion content, longer than the 160-char cap
DEF_EXT = "\n\nNow list the three most important open risks, one line each."


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        p = urllib.parse.urlparse(self.path)
        if p.path == "/v1/chat/completions":
            kind = "chat"
        elif p.path == "/completion":
            kind = "comp"
        elif p.path == "/apply-template":
            kind = "template"
        elif p.path == "/tokenize":
            kind = "tok"
        else:
            kind = urllib.parse.parse_qs(p.query)["action"][0]
        REG.append((kind, body))
        if kind != "chat" and RESP["fail_slots"]:
            self.send_response(500)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")
            return
        if kind == "chat":
            out = {"choices": [{"message": {"content": "ok"}}], "usage": {"completion_tokens": 64},
                   "timings": {"prompt_n": RESP["chat_prompt_n"], "cache_n": RESP["chat_cache_n"],
                               "prompt_per_second": 900.0, "predicted_per_second": 31.5}}
        elif kind == "template":
            out = {"prompt": "TT %d" % len(body["messages"][0]["content"])}
        elif kind == "tok":
            out = {"tokens": SID if body["content"].startswith("TT ") else ETOK}
        elif kind == "comp":
            out = {"content": RCOV, "tokens": RTOK, "stopped": True,
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

    def run_mode(self, mode, label="lbl", out=None, **extra):
        tmp = tempfile.TemporaryDirectory()
        o = out or tmp.name
        env = dict(os.environ, URL=f"http://127.0.0.1:{self.server.server_address[1]}", OUT=o,
                   MODE=mode, PROMPT_FILE=self.docfile.name, REPS="2")
        env.update(extra)
        start = len(REG)
        p = subprocess.run([sys.executable, str(CLIENT), label], env=env, capture_output=True, text=True, timeout=60)
        d = json.loads((Path(o) / f"{label}.slot.json").read_text()) if p.returncode == 0 else None
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

    def test_SLOT_env_is_the_shared_slot_filename_across_labels(self):
        p, _, calls = self.run_mode("save", SLOT="shared.slot")                     # save row: ctx1_c16384_save
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(calls[1][1], {"filename": "shared.slot"})
        p, _, calls = self.run_mode("restore", label="lbl_r", SLOT="shared.slot")   # restore row: different LABEL, same file
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(calls[0][1], {"filename": "shared.slot"})                   # restore BEFORE the request
        p, _, calls = self.run_mode("save")                                          # unset => LABEL.slot as today
        self.assertEqual(calls[1][1], {"filename": "lbl.slot"})

    def run_cold_probe(self, extra=None):
        """Exec the client in-process against the fake server with urlopen wrapped; return observed timeouts."""
        observed = []
        real = urllib.request.urlopen
        tmp = tempfile.mkdtemp()

        def probe(req, *a, **kw):
            observed.append(kw.get("timeout"))
            return real(req, *a, **kw)

        env = {"URL": f"http://127.0.0.1:{self.server.server_address[1]}", "OUT": tmp, "MODE": "cold",
               "PROMPT_FILE": self.docfile.name, "REPS": "2"}
        env.update(extra or {})
        snap, snap_argv = dict(os.environ), sys.argv
        try:
            os.environ.update(env)
            urllib.request.urlopen = probe
            with ExitStack() as st:
                st.callback(setattr, urllib.request, "urlopen", real)
                st.enter_context(redirect_stdout(io.StringIO()))
                sys.argv = ["slotclient.py", "lbl"]
                exec(compile(Path(CLIENT).read_text(encoding="utf-8"), str(CLIENT), "exec"),
                     {"__name__": "__main__", "__file__": str(CLIENT)})
        finally:
            os.environ.clear(); os.environ.update(snap); sys.argv = snap_argv
            urllib.request.urlopen = real
        return observed

    # ------------------------- brief 17: token-exact presave / extend -------------------------
    def test_presave_tokenizes_completes_saves_sidecar(self):
        tmp = tempfile.TemporaryDirectory()
        p, d, calls = self.run_mode("presave", out=tmp.name)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual([k for k, _ in calls], ["template", "tok", "comp", "save"])
        self.assertEqual(calls[0][1], {"messages": [{"role": "user", "content": self.content}],
                                       "chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(calls[1][1], {"content": "TT %d" % len(self.content), "add_special": False,
                                       "parse_special": True})
        self.assertEqual(calls[2][1], {"prompt": SID, "n_predict": 1, "temperature": 0,
                                       "cache_prompt": True, "return_tokens": True})
        self.assertEqual(calls[3][1], {"filename": "lbl.slot"})
        r = d["rows"][0]
        self.assertEqual((r["prompt_n"], r["prefill_tps"], r["decode_tps"]), (5120, 900.0, 31.5))
        self.assertEqual(r["base_n"], len(SID + RTOK))
        self.assertEqual((r["slot_bytes"], r["slot_ms"]), (1234567, 67.0))
        sc = json.loads((Path(tmp.name) / "lbl.slot.ids.json").read_text())
        self.assertEqual(sc["ids"], SID + RTOK)
        self.assertIn("slot 1234567 bytes in 67.0 ms", p.stdout)

    def test_presave_PRE_GEN_env_sets_n_predict(self):
        p, _, calls = self.run_mode("presave", PRE_GEN="64")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual([k for k, _ in calls], ["template", "tok", "comp", "save"])
        self.assertEqual(calls[2][1]["n_predict"], 64)

    def test_extend_missing_sidecar_exits_before_any_http(self):
        tmp = tempfile.TemporaryDirectory()
        start = len(REG)
        p, d, calls = self.run_mode("extend", out=tmp.name)
        self.assertEqual(p.returncode, 1)
        self.assertEqual(calls, [])
        self.assertIn("sidecar", p.stderr)
        self.assertIsNone(d)
        self.assertEqual(REG[start:], [])

    def test_extend_restores_extends_and_reports_row(self):
        tmp = tempfile.TemporaryDirectory()
        p, _, _ = self.run_mode("presave", out=tmp.name, SLOT="shared.slot")   # sidecar is keyed by SLOT
        self.assertEqual(p.returncode, 0, p.stderr)
        p, d, calls = self.run_mode("extend", label="lbl_x", out=tmp.name, SLOT="shared.slot")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual([k for k, _ in calls], ["restore", "tok", "comp"])
        self.assertEqual(calls[0][1], {"filename": "shared.slot"})      # same shared-slot contract as brief 13
        self.assertEqual(calls[1][1], {"content": DEF_EXT, "add_special": False, "parse_special": True})
        self.assertEqual(calls[2][1], {"prompt": SID + RTOK + ETOK, "n_predict": 64, "temperature": 0,
                                       "cache_prompt": True})
        r = d["rows"][0]
        self.assertEqual((r["base_n"], r["ext_n"], r["drop"]), (8, 2, 0))
        self.assertEqual(r["reply"], "X" * 160)
        self.assertEqual((r["slot_bytes"], r["slot_ms"]), (1234567, 12.5))

    def test_extend_DROP_removes_last_sidecar_id_only(self):
        tmp = tempfile.TemporaryDirectory()
        self.run_mode("presave", out=tmp.name, SLOT="shared.slot")
        p, d, calls = self.run_mode("extend", label="lbl_x", out=tmp.name, SLOT="shared.slot",
                                    DROP="1", EXT="TAIL")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(calls[1][1]["content"], "TAIL")
        self.assertEqual(calls[2][1]["prompt"], (SID + RTOK)[:-1] + ETOK)
        r = d["rows"][0]
        self.assertEqual((r["drop"], r["base_n"], r["ext_n"]), (1, 8, 2))

    def test_TIMEOUT_env_controls_the_urlopen_kwarg(self):
        self.assertEqual(self.run_cold_probe(), [3600])                 # unset => 1 h default
        self.assertEqual(self.run_cold_probe({"TIMEOUT": "21600"}), [21600])   # 6 h for deep prefill


if __name__ == "__main__":
    unittest.main()
