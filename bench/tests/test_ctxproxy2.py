"""tests for bench/box/ctxproxy2.py. Part 1 exercises the EntryStore (slotlib + exact-ids sidecar)
directly. Part 2 runs the real proxy against a fake upstream http.server (ephemeral ports, requests
recorded, scripted /apply-template, /tokenize, /completion stream + non-stream, /slots) — the fake
tokenizer deliberately tokenizes DIFFERENTLY from any naive re-render, so a proxy that re-tokenizes
history fails the byte-identity assertions."""
import json, os, sys, tempfile, threading, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "bench" / "box"))
import ctxproxy2


def put(store, msgs, ids, slot_bytes=1000):
    key, ent, _ = store.begin(msgs)
    store.sl.file_for(key).write_bytes(b"SLOTDATA" * 8)
    st = store.finish(key, ids, slot_bytes=slot_bytes)
    return key, st


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.st = ctxproxy2.EntryStore(self.d, 10 * 2**30, "m1", "f16/f16")

    def test_reserve_commit_lookup_returns_ids(self):
        msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        key, st = put(self.st, msgs, [1, 2, 3, 4])
        self.assertEqual(st, "ok")
        ids, k, got = self.st.lookup([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                                      {"role": "user", "content": "more"}])
        self.assertEqual((ids, k, got), ([1, 2, 3, 4], 2, key))
        side = Path(self.st.sidecar(key))
        self.assertEqual(json.loads(side.read_text())["ids"], [1, 2, 3, 4])

    def test_longest_prefix_wins(self):
        m2 = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        m4 = m2 + [{"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]
        put(self.st, m2, [1, 2])
        put(self.st, m4, [1, 2, 3, 4, 5])
        ids, k, _ = self.st.lookup(m4 + [{"role": "user", "content": "e"}])
        self.assertEqual((ids, k), ([1, 2, 3, 4, 5], 4))
        ids, k, _ = self.st.lookup(m2 + [{"role": "user", "content": "changed"}])
        self.assertEqual((ids, k), ([1, 2], 2))

    def test_corrupt_sidecar_dropped_and_healed(self):
        msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        key, _ = put(self.st, msgs, [1, 2])
        Path(self.st.sidecar(key)).write_text("{ not json")
        ids, k, _ = self.st.lookup(msgs)
        self.assertEqual((ids, k), (None, 0))
        self.assertNotIn(key, self.st.sl.index["entries"])             # healed: gone from index
        self.assertFalse(Path(self.st.sidecar(key)).exists())          # sidecar removed
        self.assertFalse(self.st.sl.file_for(key).exists())            # slot file removed with it

    def test_abort_removes_sidecar(self):
        key = self.st.sl.key_for("m1", "f16/f16", [{"role": "user", "content": "x"}])
        self.st.sl.reserve("m1", "f16/f16", [{"role": "user", "content": "x"}], 1)
        Path(self.st.sidecar(key)).write_text("{}")
        self.st.abort(key)
        self.assertFalse(Path(self.st.sidecar(key)).exists())

    def test_eviction_removes_slot_and_sidecar(self):
        small = ctxproxy2.EntryStore(self.d + "/b", 1500, "m1", "f16/f16")
        mA = [{"role": "user", "content": "a" * 10}, {"role": "assistant", "content": "ra"}]
        mB = [{"role": "user", "content": "b" * 10}, {"role": "assistant", "content": "rb"}]
        kA, _ = put(small, mA, [1], slot_bytes=1200)
        kB, st = put(small, mB, [2], slot_bytes=1200)                    # over budget -> A evicted
        self.assertEqual(st, "ok")
        self.assertFalse(small.sl.file_for(kA).exists())
        self.assertFalse(Path(small.sidecar(kA)).exists())
        self.assertTrue(Path(small.sidecar(kB)).exists())

    def test_finish_non_ok_clears_sidecar(self):
        huge = ctxproxy2.EntryStore(self.d + "/c", 100, "m1", "f16/f16")
        ent = huge.sl.reserve("m1", "f16/f16", [{"role": "user", "content": "q"}], 1)
        key = ent["key"]
        st = huge.finish(key, [1], slot_bytes=10_000)                    # bigger than whole budget
        self.assertEqual(st, "evicted")
        self.assertFalse(Path(huge.sidecar(key)).exists())


# ------------------------------------------------------------------ part 2: fake upstream + proxy
import http.client, http.server, json as _json, socket, threading, time, urllib.error, urllib.request, zlib

STAR = None


def TOK_DEFAULT(content):
    if content == "":
        return []
    ids = [len(content)]
    ids += [int(zlib.crc32(content[i:i + 6].encode()) % 9000) + 10 for i in range(0, len(content), 6)]
    return ids + [7]                          # 7 == end-of-turn token id appended by the tokenizer


class Fake(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self, daemon=True)
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), self.H)
        self.port = self.srv.server_address[1]
        self.tmpl = lambda msgs: "".join("[ROLE:%s]%s@@@" % (m["role"], m["content"]) for m in msgs)
        self.tok = TOK_DEFAULT
        self.comp = {"content": "Hello there", "tokens": [7], "tokens_predicted": 1,
                     "tokens_evaluated": 120, "stop_type": "eos",
                     "timings": {"prompt_n": 120, "cache_n": 100, "predicted_per_second": 30.0}}
        self.stream_events = [{"content": "Hel"}, {"content": "lo", "tokens": [7],
                               "timings": {"prompt_n": 5, "cache_n": 300}}]
        self.mode = "nonstream"
        self.restore_fails = False
        self.save_fails = False
        self.health_body = b"OK-health"
        self.root = None                                # slots save writes files HERE (proxy root)
        self.requests = []
        self.counts = {"restore": 0, "save": 0}
        self.start()

    def run(self):
        self.srv.serve_forever()

    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *a):
            pass

        def _j(self, obj, code=200):
            b = _json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        def do_GET(self):
            F = STAR
            F.requests.append(("GET", self.path, None))
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(F.health_body)))
                self.end_headers()
                self.wfile.write(F.health_body)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            F = STAR
            body = self._body()
            path = self.path
            try:
                j = _json.loads(body) if body else {}
            except ValueError:
                j = {"RAW": body}
            F.requests.append(("POST", path, j))
            if path == "/apply-template":
                return self._j({"prompt": F.tmpl(j["messages"])})
            if path == "/tokenize":
                return self._j({"tokens": F.tok(j["content"])})
            if path == "/completion":
                if F.mode == "stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for ev in F.stream_events:
                        self.wfile.write(b"data: " + _json.dumps(ev).encode() + b"\n\n")
                        self.wfile.flush()
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    return
                return self._j(F.comp)
            if path.startswith("/slots/0"):
                if "restore" in path:
                    F.counts["restore"] += 1
                    if F.restore_fails:
                        return self._j({"error": "restore boom"}, 500)
                    return self._j({"timings": {"restore_ms": 5.0}})
                F.counts["save"] += 1
                if F.save_fails:
                    return self._j({"error": "save boom"}, 500)
                if F.root:                              # real llama-server CREATES the slot file
                    open(os.path.join(F.root, j.get("filename", "x.slot")), "wb").write(b"D" * 4096)
                return self._j({"n_written": 4096, "timings": {"save_ms": 2.0}})
            self._j({"echoed": j}, 201)


class ProxyCase(unittest.TestCase):
    def setUp(self):
        global STAR
        STAR = Fake()
        self.fake = STAR
        self.d = tempfile.mkdtemp()
        self.fake.root = self.d
        self.logf = os.path.join(self.d, "proxy.jsonl")
        self.srv = ctxproxy2.serve("127.0.0.1:0", "http://127.0.0.1:%d" % self.fake.port,
                                   self.d, 4, "m1", "f16/f16", 1, self.logf)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.pport = self.srv.server_address[1]
        self.msgs1 = [{"role": "user", "content": "HELLO-WORLD-ONE"}]

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.fake.srv.shutdown()
        self.fake.srv.server_close()
        STAR = None

    # helpers ---------------------------------------------------------------------
    def post(self, msgs, **kw):
        body = {"messages": msgs}
        body.update(kw)
        req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % self.pport,
                                     _json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, _json.loads(r.read()), dict(r.headers)
        except urllib.error.HTTPError as e:
            return e.code, e.read(), dict(e.headers)

    def completions_sent(self):
        return [j for m, pth, j in self.fake.requests if pth == "/completion"]

    def logs(self):
        return [_json.loads(l) for l in open(self.logf)]

    def sidecar_files(self):
        return list(Path(self.d).glob("*.ids.json"))

    # flows -----------------------------------------------------------------------
    def test_miss_then_completion_carries_ids_and_saves(self):
        code, obj, hdr = self.post(self.msgs1)
        self.assertEqual(code, 200)
        self.assertEqual(obj["object"], "chat.completion")
        self.assertEqual(obj["choices"][0]["message"]["content"], "Hello there")
        self.assertEqual(obj["choices"][0]["finish_reason"], "stop")
        self.assertIn("timings", obj)
        self.assertEqual(hdr["X-Ctx-Cache"], "miss")
        sent = self.completions_sent()[0]
        self.assertEqual(sent["cache_prompt"], True)
        self.assertEqual(sent["return_tokens"], True)
        full = self.fake.tmpl(self.msgs1)
        self.assertEqual(sent["prompt"], self.fake.tok(full))
        sc = self.sidecar_files()
        self.assertEqual(len(sc), 1)
        self.assertEqual(_json.loads(sc[0].read_text())["ids"], self.fake.tok(full) + [7])
        self.assertEqual(self.logs()[-1]["hit"], False)

    def seed_conv(self, msgs, ids):
        key, _ent, _ = ctxproxy2.STORE.begin(msgs)
        open(os.path.join(self.d, "%s.slot" % key), "wb").write(b"D" * 4096)
        self.assertEqual(ctxproxy2.STORE.finish(key, ids, slot_bytes=4096), "ok")
        return key

    def test_hit_restores_extends_and_never_retokens_history(self):
        self.post(self.msgs1)
        stored = _json.loads(self.sidecar_files()[0].read_text())["ids"]
        reply = self.fake.comp["content"]
        self.fake.requests.clear()
        msgs2 = self.msgs1 + [{"role": "assistant", "content": reply},
                              {"role": "user", "content": "SECOND-QUESTION"}]
        code, obj, hdr = self.post(msgs2)
        self.assertEqual(code, 200)
        # NO restore on this hop: turn 1's save left EXACTLY this state resident server-side
        self.assertEqual(self.fake.counts["restore"], 0)
        self.assertIn("hit; restored=%d" % len(stored), hdr["X-Ctx-Cache"])
        conv2 = self.msgs1 + [{"role": "assistant", "content": reply}]
        ctxproxy2.LOADED["key"] = "some-other-key"      # force eviction of server state
        code, obj, hdr = self.post(msgs2)
        self.assertEqual(self.fake.counts["restore"], 1)
        self.assertIn("hit; restored=%d" % len(stored), hdr["X-Ctx-Cache"])
        sent = self.completions_sent()[-1]
        prompt = sent["prompt"]
        self.assertEqual(prompt[:len(stored)], stored)              # byte-identical history ids
        full2 = self.fake.tmpl(msgs2)
        self.assertNotEqual(prompt[:12], self.fake.tok(full2)[:12])  # NOT a re-tokenized render
        self.assertEqual(prompt[len(stored):][-1] if len(prompt) > len(stored) else None,
                         prompt[-1])
        self.assertGreater(len(prompt), len(stored))
        self.assertNotIn(stored[-1], prompt[len(stored):-1])         # dup eot only ever the final tok

    def test_duplicated_end_of_turn_dropped(self):
        self.post(self.msgs1)
        stored = _json.loads(self.sidecar_files()[0].read_text())["ids"]
        msgs2 = self.msgs1 + [{"role": "assistant", "content": self.fake.comp["content"]},
                              {"role": "user", "content": "X"}]
        self.post(msgs2)
        sent = self.completions_sent()[-1]["prompt"]
        tail_after_hist = sent[len(stored):]
        self.assertNotIn(7, tail_after_hist[:-1] if len(tail_after_hist) > 1 else tail_after_hist)

    def test_assistant_content_not_found_falls_back_to_miss(self):
        self.post(self.msgs1)
        self.fake.tmpl = lambda msgs: "COMPLETELY-DIFFERENT-RENDER"   # reply content absent
        msgs2 = self.msgs1 + [{"role": "assistant", "content": "Hello there"},
                              {"role": "user", "content": "again"}]
        code, obj, hdr = self.post(msgs2)
        self.assertEqual(code, 200)
        self.assertEqual(hdr["X-Ctx-Cache"], "miss")
        self.assertEqual(self.fake.counts["restore"], 0)
        sent = self.completions_sent()[-1]
        self.assertEqual(sent["prompt"], self.fake.tok("COMPLETELY-DIFFERENT-RENDER"))

    def test_loaded_key_short_circuits_second_restore(self):
        conv = self.msgs1 + [{"role": "assistant", "content": "Hello there"}]
        seeded = self.seed_conv(conv, self.fake.tok(self.fake.tmpl(self.msgs1)) + [7])
        msgs2 = conv + [{"role": "user", "content": "again"}]
        self.srv.shutdown(); self.srv.server_close()                 # high min-chars: no save churn
        self.srv = ctxproxy2.serve("127.0.0.1:%d" % self.pport,
                                   "http://127.0.0.1:%d" % self.fake.port,
                                   self.d, 4, "m1", "f16/f16", 10 ** 9, self.logf)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.post(msgs2)
        self.post(msgs2)                                             # same prefix key, reload skipped
        self.assertEqual(self.fake.counts["restore"], 1)

    def test_restore_failure_falls_back_cold_but_200(self):
        conv = self.msgs1 + [{"role": "assistant", "content": "Hello there"}]
        self.seed_conv(conv, self.fake.tok(self.fake.tmpl(self.msgs1)) + [7])
        msgs2 = conv + [{"role": "user", "content": "again"}]
        self.fake.restore_fails = True
        code, obj, hdr = self.post(msgs2)
        self.assertEqual(code, 200)
        self.assertEqual(hdr["X-Ctx-Cache"], "miss")
        sent = self.completions_sent()[-1]
        self.assertEqual(sent["prompt"], self.fake.tok(self.fake.tmpl(msgs2)))
        self.assertIn("restore_failed", self.logs()[-1])

    def test_save_failure_aborts_entry_reply_intact(self):
        n_before = len(self.sidecar_files())
        self.fake.save_fails = True
        code, obj, hdr = self.post(self.msgs1)
        self.assertEqual(code, 200)
        self.assertEqual(obj["choices"][0]["message"]["content"], "Hello there")
        self.assertEqual(len(self.sidecar_files()), n_before)
        self.assertEqual([e for e in ctxproxy2.STORE.sl.index["entries"].values()
                          if e["state"] != "ready"], [])
        self.assertIn("save_failed", self.logs()[-1])

    def test_below_min_chars_stores_nothing(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.srv = ctxproxy2.serve("127.0.0.1:%d" % self.pport, "http://127.0.0.1:%d" % self.fake.port,
                                   self.d, 4, "m1", "f16/f16", 10 ** 9, self.logf)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        code, obj, hdr = self.post(self.msgs1)
        self.assertEqual(code, 200)
        self.assertEqual(self.sidecar_files(), [])
        self.assertEqual(ctxproxy2.STORE.sl.index["entries"], {})

    def test_stream_chunks_order_role_done(self):
        self.fake.mode = "stream"
        conn = http.client.HTTPConnection("127.0.0.1", self.pport, timeout=30)
        body = _json.dumps({"messages": self.msgs1, "stream": True})
        conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
        r = conn.getresponse()
        self.assertEqual(r.status, 200)
        events = []
        acc = b""
        while True:
            line = r.fp.readline()
            if not line:
                break
            if line.startswith(b"data: "):
                events.append(line[6:].strip())
            if line.strip() == b"data: [DONE]":
                break
        texts, first_obj = [], None
        for ev in events:
            if ev == b"[DONE]":
                continue
            d = _json.loads(ev)
            self.assertEqual(d["object"], "chat.completion.chunk")
            delta = d["choices"][0]["delta"]
            if first_obj is None:
                first_obj = delta
            if "content" in delta:
                texts.append(delta["content"])
        self.assertEqual(first_obj.get("role"), "assistant")
        self.assertEqual("".join(texts), "Hello")
        fins = [_json.loads(e)["choices"][0]["finish_reason"] for e in events
                if e != b"[DONE]" and _json.loads(e)["choices"][0].get("finish_reason")]
        self.assertEqual(fins[-1], "stop")
        conn.close()

    def test_relay_get_byte_identical(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.pport, timeout=30)
        conn.request("GET", "/health")
        r = conn.getresponse()
        self.assertEqual(r.read(), b"OK-health")
        self.assertEqual(r.status, 200)

    def test_upstream_down_502(self):
        self.fake.srv.shutdown()
        self.fake.srv.server_close()
        code, body, hdr = self.post(self.msgs1)
        self.assertEqual(code, 502)
        self.assertIn(b"error", body if isinstance(body, bytes) else _json.dumps(body).encode())

    def test_log_line_fields(self):
        self.post(self.msgs1)
        rec = self.logs()[-1]
        for f in ("ts", "k", "n_msgs", "stored", "suffix", "prompt_n", "cache_n", "save_ms",
                  "save_bytes", "evicted"):
            self.assertIn(f, rec)
        self.assertEqual(rec["prompt_n"], 120)


if __name__ == "__main__":
    unittest.main()
