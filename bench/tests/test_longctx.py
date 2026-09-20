"""tests for bench/qual/longctx.py — fake llama-server (/tokenize, /v1/chat/completions, /slots/0) like
test_slotclient; client via subprocess except the TIMEOUT-probe test (exec in-process with urlopen
wrapped). The fake server answers in `oracle` mode by READING the request (document + question in the same
body) and replying with the retrieved number / the variables holding the asked value => a working client
scores 100% end to end."""
import io, json, os, re, subprocess, sys, tempfile, threading, unittest, urllib.parse, urllib.request
from contextlib import ExitStack, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
TOOL = BASE / "bench" / "qual" / "longctx.py"
sys.path.insert(0, str(BASE / "bench" / "qual"))
import longctx

REG = []
RESP = {"override": None, "fail_all": False, "cache_zero": False, "restore_fail": False}


def oracle(content):
    doc, q = content.split("\n\n", 1)
    m = re.search(r"magic number for (\w+)\?", q)
    if m:
        mm = re.search(rf"The special magic number for {m.group(1)} is (\d+)\.", doc)
        return f"It is {mm.group(1)}." if mm else "?"
    m = re.search(r"value (\d+)\.", q)
    if m:
        val = int(m.group(1))
        held, names = [], set()
        vals = {}
        for lhs, rhs in re.findall(r"VAR ([A-Z]{3}) = (VAR [A-Z]{3}|[\d]+)\.", doc):
            vals[lhs] = vals[rhs[4:]] if rhs.startswith("VAR ") else int(rhs)
        return " ".join(v + "," for v, x in vals.items() if x == val) or "none"
    return "hm"


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        path = urllib.parse.urlparse(self.path).path
        if path == "/tokenize":
            out = {"toks": [0] * (len(body.get("content", "")) // 4)}
        elif path == "/v1/chat/completions":
            REG.append(("chat", body))
            if RESP["fail_all"]:
                self.send_response(400)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")
                return
            content = body["messages"][0]["content"]
            txt = RESP["override"] if RESP["override"] is not None else oracle(content)
            pn = len(content) // 4
            cn = 0 if RESP["cache_zero"] else pn
            out = {"choices": [{"message": {"content": txt}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": pn},
                   "timings": {"prompt_n": pn, "cache_n": cn, "prompt_per_second": 40.0,
                               "predicted_per_second": 30.0}}
        else:                                                   # /slots/0?action=...
            action = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)["action"][0]
            REG.append((action, body))
            if action == "restore" and RESP["restore_fail"]:
                self.send_response(500)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")
                return
            out = {"success": True, "n_saved": 1, "n_written": 10}
        b = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


class LQ(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        REG.clear()
        for k in RESP:
            RESP[k] = type(RESP[k])()

    def cli(self, label="lb", *args, env=None):
        tmp = tempfile.mkdtemp()
        e = dict(os.environ)
        if env:
            e.update(env)
        p = subprocess.run([sys.executable, str(TOOL), label, "--url", self.url, "--out", tmp, *map(str, args)],
                           capture_output=True, text=True, timeout=120, env=e)
        return p, tmp

    def test_plan_units(self):
        a = longctx.plan(1, 16384, 4, 1)
        b = longctx.plan(1, 16384, 4, 1)
        self.assertEqual(a, b)                                   # pure function of (seed, depth)
        c = longctx.plan(1, 32768, 4, 1)
        self.assertNotEqual(a["nums"], c["nums"])                # depth-keyed
        self.assertNotEqual(a["chains"], c["chains"])
        self.assertEqual(longctx.plan(1, 9, 1, 0)["positions"], [0.5])
        self.assertEqual(longctx.plan(1, 9, 5, 0)["positions"], [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertGreater(longctx.plan(1, 9, 5, 0)["positions"][2],
                           longctx.plan(1, 9, 5, 0)["positions"][1])          # monotone, ends at the ends
        p = longctx.plan(1, 9, 3, 3)
        self.assertEqual(len(set(p["keys"])), 3)
        names = [v for ch in p["chains"] for v in ch["vars"]]
        self.assertEqual(len(set(names)), 12)
        doc1 = longctx.document(dict(p, tokens=180), 4.0, 0)
        doc2 = longctx.document(dict(longctx.plan(1, 9, 3, 3), tokens=180), 4.0, 0)
        self.assertEqual(doc1, doc2)

    def test_scoring_units(self):
        self.assertEqual(longctx.score_needle("It is 1234567.", "1234567"), 1.0)
        self.assertEqual(longctx.score_needle("value 912345670 here", "1234567"), 0.0)   # longer digit run
        self.assertEqual(longctx.score_needle("123456", "1234567"), 0.0)
        names = ["AAA", "BBB", "CCC", "DDD"]
        self.assertEqual(longctx.score_vt("AAA and BBB hold it", names, ["ZZZ"]), 0.5)
        self.assertEqual(longctx.score_vt("AAA ZZZ", names, ["ZZZ"]), 0.0)               # foreign name => 0

    def test_one_doc_many_questions_cache_prompt_and_oracle(self):
        p, tmp = self.cli("lb", "--depths", 256, "--needles", 2, "--chains", 1)
        self.assertEqual(p.returncode, 0, p.stderr)
        chats = [b for k, b in REG if k == "chat"]
        self.assertEqual(len(chats), 3)
        docs = {c.split("\n\n", 1)[0] for (_, c) in [(0, b["messages"][0]["content"]) for b in chats]}
        self.assertEqual(len(docs), 1)                           # ONE document, many questions
        for b in chats:
            self.assertTrue(b.get("cache_prompt"))
            self.assertEqual(b["temperature"], 0)
            self.assertFalse(b["chat_template_kwargs"]["enable_thinking"])
            self.assertEqual(b["max_tokens"], 48)
        js = [json.loads(l) for l in (Path(tmp) / "lb.longctx.jsonl").read_text().splitlines()]
        self.assertEqual([r["kind"] for r in js], ["needle", "needle", "vt"])
        self.assertEqual([r["position"] for r in js], [0.0, 1.0, None])
        s = json.loads((Path(tmp) / "lb.longctx.summary.json").read_text())
        self.assertEqual(s["depths"]["256"]["needle_pct"], 100.0)
        self.assertEqual(s["depths"]["256"]["vt_mean"], 1.0)
        self.assertNotIn("prefix_reused", s["depths"]["256"])    # ratio ~1 => no flag
        self.assertIn("0.0-0.2", s["needle_pct_by_position"])
        self.assertIn("0.8-1.0", s["needle_pct_by_position"])

    def test_overflow_shrinks_three_times_then_marks_and_moves_on(self):
        RESP["fail_all"] = True
        p, tmp = self.cli("ov", "--depths", 256, "--needles", 1, "--chains", 0)
        self.assertEqual(p.returncode, 0, p.stderr + p.stdout)
        self.assertEqual(len([k for k, _ in REG if k == "chat"]), 1 + longctx.MAX_SHRINKS)
        self.assertIn("OVERFLOW depth 256", p.stdout)
        self.assertFalse((Path(tmp) / "ov.longctx.jsonl").exists())
        self.assertIn("depths", json.loads((Path(tmp) / "ov.longctx.summary.json").read_text()))

    def test_resume_skips_finished_questions(self):
        p, tmp = self.cli("rs", "--depths", 256, "--needles", 2, "--chains", 0)
        self.assertEqual(p.returncode, 0, p.stderr)
        n = len([k for k, _ in REG if k == "chat"])
        self.assertEqual(n, 2)
        REG.clear()
        p2 = subprocess.run([sys.executable, str(TOOL), "rs", "--url", self.url, "--out", tmp,
                             "--depths", "256", "--needles", "2", "--chains", "0"],
                            capture_output=True, text=True, timeout=120)
        self.assertEqual(p2.returncode, 0, p2.stderr)
        self.assertEqual([k for k, _ in REG if k == "chat"], [])
        self.assertEqual(len((Path(tmp) / "rs.longctx.jsonl").read_text().splitlines()), 2)

    def test_slot_ops_save_once_restore_failure_nonfatal(self):
        p, tmp = self.cli("sl", "--depths", 256, "--needles", 2, "--chains", 0, "--slot-save")
        self.assertEqual(p.returncode, 0, p.stderr)
        saves = [b for k, b in REG if k == "save"]
        self.assertEqual(saves, [{"filename": "sl_d256.slot"}])  # ONE save for the depth
        REG.clear()
        RESP["restore_fail"] = True
        p, tmp = self.cli("sr", "--depths", 256, "--needles", 1, "--chains", 0, "--slot-restore")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("restore ignored", p.stdout)
        self.assertEqual(len([k for k, _ in REG if k == "chat"]), 1)

    def test_prefix_not_reused_flag(self):
        RESP["cache_zero"] = True
        p, tmp = self.cli("pz", "--depths", 256, "--needles", 2, "--chains", 0)
        self.assertEqual(p.returncode, 0, p.stderr)
        s = json.loads((Path(tmp) / "pz.longctx.summary.json").read_text())
        self.assertIs(s["depths"]["256"]["prefix_reused"], False)
        self.assertIn("PREFIX NOT REUSED", p.stdout)

    def test_reuse_guard_skips_the_rest_of_a_deep_depth(self):
        RESP["cache_zero"] = True                      # the server never re-uses the document prefix
        p, tmp = self.cli("rg", "--depths", "256,512", "--needles", 5, "--chains", 0, "--reuse-guard", 512)
        self.assertEqual(p.returncode, 0, p.stderr)
        per = {}
        for _, body in [r for r in REG if r[0] == "chat"]:
            n = len(body["messages"][0]["content"])
            per[n > 1500] = per.get(n > 1500, 0) + 1
        self.assertEqual(per[False], 5)                # depth 256 is below the guard: all 5 questions asked
        self.assertEqual(per[True], 2)                 # depth 512: first question + the one that exposes no re-use, then stop
        self.assertIn("skipping the rest of depth 512", p.stdout)

    def test_TIMEOUT_env_reaches_urlopen(self):
        observed = []
        real = urllib.request.urlopen

        def probe(req, *a, **kw):
            observed.append(kw.get("timeout"))
            return real(req, *a, **kw)

        snap, snap_argv = dict(__import__("os").environ), sys.argv
        tmp = tempfile.mkdtemp()
        try:
            for val, want in ((None, 21600), ("60", 60)):
                observed.clear()
                if val is None:
                    __import__("os").environ.pop("TIMEOUT", None)
                else:
                    __import__("os").environ["TIMEOUT"] = val
                sys.argv = ["longctx.py", "tt", "--url", self.url, "--out", tmp, "--depths", "128",
                            "--needles", "1", "--chains", "0"]
                urllib.request.urlopen = probe
                with ExitStack() as st:
                    st.callback(setattr, urllib.request, "urlopen", real)
                    st.enter_context(redirect_stdout(io.StringIO()))
                    exec(compile(TOOL.read_text(encoding="utf-8"), str(TOOL), "exec"),
                         {"__name__": "__main__", "__file__": str(TOOL)})
                self.assertTrue(observed and all(t == want for t in observed), (want, observed))
        finally:
            __import__("os").environ.clear(); __import__("os").environ.update(snap)
            sys.argv = snap_argv
            urllib.request.urlopen = real


import os


if __name__ == "__main__":
    unittest.main()
