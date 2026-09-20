#!/usr/bin/env python3
"""ctxproxy2.py — the TOKEN-EXACT context proxy (replaces the brief-16 proxy design; slotlib stays).

Measured on the box: a slot restores in ~0.5 s and the server re-uses ALL of it (131k: 6.6 s instead of
35 min) — but only when the new request's TOKEN IDS start with EXACTLY the saved ids (recurrent hybrid:
any mismatch inside the saved range forces full re-prefill, and re-rendering history through the chat
template does NOT reproduce the model's own generated ids). So the proxy never re-tokenizes history:
each stored conversation keeps its exact ids in a sidecar, and only the NEW suffix is tokenized.

Entries live in slotlib (content-addressed over the message prefix, LRU budget) plus a sidecar
<key>.ids.json {"ids": [...], "n_msgs": k} — k messages INCLUDING the assistant reply that ended the
stored exchange. A missing/corrupt sidecar demotes the entry exactly like a missing slot file (dropped +
healed). Evictions and aborts take the sidecar with them (slotlib on_evict hook wired here — the store
wrapper lives in THIS file; slotlib was not touched).

Request flow for POST /v1/chat/completions under ONE lock (incl. the post-answer save):
  lookup longest prefix -> stored ids; render the full prompt via upstream /apply-template (string only,
  no tokenize); k == 0 -> ids = tokenize(full); k > 0 -> suffix = full[right-most occurrence of message
  k-1 content:], ids = stored + tokenize(suffix) minus a leading duplicate of the stored end-of-turn id;
  restore (once per loaded key) then POST /completion with the id array (cache_prompt true,
  return_tokens true, stream passthrough). Non-stream answers an OpenAI chat.completion object (timings
  passed through, stop_type -> finish_reason); stream relays OpenAI chunk events built line-by-line from
  the upstream SSE (for line in resp — never a sized read; that was the brief-16 bug). Worth storing =
  canonical JSON length >= --min-chars; save + sidecar AFTER the client's last byte, still under the
  lock; a failed save aborts the entry (sidecar removed) but never touches the reply.
Everything else is relayed unchanged; upstream down => 502 one-line JSON. X-Ctx-Cache header states
hit/miss. One JSONL line per request (ts, k/n, id counts, prompt_n/cache_n, restore/save ms, bytes,
evicted keys).

usage: ctxproxy2.py --listen 127.0.0.1:8090 --upstream http://127.0.0.1:8099 --root /ai/bench/slots2
                    [--budget-gib 14] [--model-id M] [--kv q4_0/q4_0] [--min-chars 4000] [--log FILE]
"""
import argparse, hashlib, json, os, re, socket, sys, threading, time, urllib.error, urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slotlib

TOOL_CALL_END_SENTINEL_NONE = None   # (special tokens never appear in this file; they come from the server)


class EntryStore:
    """slotlib wrapper adding the exact-ids sidecar. Slotlib decides WHAT prefixes are stored; this
    decides what their token ids were. Corrupt sidecar == corrupt slot file: dropped and healed."""

    def __init__(self, root, budget_bytes, model_id, kv):
        self.sl = slotlib.Store(root, budget_bytes)
        self.model_id, self.kv = model_id, kv
        self.sl.on_evict.append(self._drop_sidecars)

    def sidecar(self, key):
        return self.sl.root / ("%s.ids.json" % key)

    def _drop_sidecars(self, keys):
        for k in keys:
            try:
                self.sidecar(k).unlink()
            except OSError:
                pass

    def _read_ids(self, key):
        try:
            d = json.loads(self.sidecar(key).read_text())
            ids = d["ids"]
            if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
                raise ValueError("ids")
            return ids
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def lookup(self, messages):
        """-> (ids, k, key) for the longest stored message prefix whose slot file AND sidecar are intact;
        a ready entry with either missing is healed (dropped + files/sidecar removed) and the scan
        continues to shorter prefixes (mirrors slotlib.lookup's heal semantics)."""
        for k in range(len(messages), 0, -1):
            key = self.sl.key_for(self.model_id, self.kv, messages[:k])
            ent = self.sl.index["entries"].get(key)
            if not ent or ent.get("state") != "ready":
                continue
            ids = None
            if self.sl.file_for(key).exists():
                ids = self._read_ids(key)
            if ids is None:
                self.sl.abort(key)
                self._drop_sidecars([key])
                continue
            return ids, k, key
        return None, 0, None

    def begin(self, messages_full, evict_extra=0):
        """Evict for room, then reserve over the FULL conversation (through the assistant reply).
        -> (key, entry) or (None, None) callers must still write the slot + sidecar and finish()."""
        evicted = self.sl.evict_to_budget(evict_extra)
        key = self.sl.key_for(self.model_id, self.kv, messages_full)
        ent = self.sl.reserve(self.model_id, self.kv, messages_full, len(messages_full))
        return key, ent, evicted

    def finish(self, key, ids, slot_bytes=None):
        """Write the sidecar atomically BEFORE commit (an entry is electable only with intact ids)."""
        p = self.sidecar(key)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps({"ids": ids}))
        os.replace(tmp, p)
        st, _ = self.sl.commit(key, bytes=slot_bytes)
        if st != "ok":
            try:
                p.unlink()
            except OSError:
                pass
        return st

    def abort(self, key):
        self.sl.abort(key)
        self._drop_sidecars([key])

    def touch(self, key):
        self.sl.touch(key)

    def gc(self):
        return self.sl.gc()


def _load_cfg(a):
    global CFG, STORE, LOCK, UP, MODEL_ID, MIN_CHARS, LOGP, SEQ, LOADED
    CFG = a
    STORE = EntryStore(a.root, int(a.budget_gib * 2**30), a.model_id, a.kv)
    LOCK = threading.Lock()
    UP = a.upstream.rstrip("/")
    MODEL_ID = a.model_id
    MIN_CHARS = a.min_chars
    LOGP = a.log
    SEQ = 0
    LOADED = {"key": None}


def _api(path, payload, stream=False, timeout=600):
    req = urllib.request.Request(UP + path, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def _api_json(path, payload, timeout=600):
    with _api(path, payload, timeout=timeout) as r:
        return json.loads(r.read())


class Handler(__import__("http.server", fromlist=["BaseHTTPRequestHandler"]).BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    # ---------- plumbing ---------------------------------------------------------
    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _send(self, code, body, ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _relay(self):
        body = self._body()
        try:
            req = urllib.request.Request(UP + self.path, data=body or None, method=self.command,
                                         headers={"Content-Type": self.headers.get("Content-Type",
                                                                                    "application/json")}
                                         if body else {})
            with urllib.request.urlopen(req, timeout=600) as r:
                self._send(r.status, r.read(), r.headers.get("Content-Type", "application/json"),
                           {"X-Ctx-Cache": "relay"})
        except urllib.error.HTTPError as e:
            self._send(e.code, e.read(), "application/json", {"X-Ctx-Cache": "relay"})
        except (urllib.error.URLError, OSError):
            self._send(502, b'{"error": "upstream unreachable"}')

    do_GET = do_PUT = do_DELETE = do_HEAD = do_OPTIONS = _relay

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._chat()
        else:
            self._relay()

    # ---------- the chat flow ----------------------------------------------------
    def _chat(self):
        global SEQ
        raw = self._body()
        try:
            req_j = json.loads(raw)
        except ValueError:
            return self._send(400, b'{"error": "bad json"}')
        msgs = req_j.get("messages") or []
        stream = bool(req_j.get("stream"))
        t_start = time.monotonic()
        rec = {"ts": round(time.time(), 3), "n_msgs": len(msgs), "k": 0, "stored": 0, "suffix": 0,
               "prompt_n": None, "cache_n": None, "restore_ms": None, "save_ms": None,
               "save_bytes": None, "evicted": [], "hit": False}
        with LOCK:
            ids_stored, k, key = STORE.lookup(msgs)
            rec["k"] = k
            rec["stored"] = len(ids_stored or [])
            tmpl = {"messages": msgs}
            if "chat_template_kwargs" in req_j:
                tmpl["chat_template_kwargs"] = req_j["chat_template_kwargs"]
            try:
                full = _api_json("/apply-template", tmpl)["prompt"]
            except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
                rec["error"] = "apply-template: %s" % e
                return self._log(rec, 502, b'{"error": "upstream apply-template failed"}')
            toks_full = None

            def full_ids():
                nonlocal toks_full
                if toks_full is None:
                    toks_full = _api_json("/tokenize", {"content": full, "add_special": False,
                                                        "parse_special": True})["tokens"]
                return toks_full

            ids = None
            if ids_stored is not None:
                prev = msgs[k - 1].get("content", "") if isinstance(msgs[k - 1], dict) else ""
                pos = full.rfind(prev) if prev else -1
                if pos < 0:
                    ids_stored = None                      # content not located: treat as miss
                    rec["k"] = k = 0
                else:
                    try:
                        sfx = _api_json("/tokenize", {"content": full[pos + len(prev):],
                                                      "add_special": False,
                                                      "parse_special": True})["tokens"]
                    except (urllib.error.URLError, OSError, ValueError, KeyError):
                        sfx = None
                        ids_stored = None
                        rec["k"] = k = 0
                        rec["suffix_tokenize_failed"] = True
                    if sfx is not None:
                        if sfx and ids_stored and sfx[0] == ids_stored[-1]:
                            sfx = sfx[1:]                  # end-of-turn token already generated+stored
                        rec["suffix"] = len(sfx)
                        ids = ids_stored + sfx
            if ids is None:
                try:
                    ids = full_ids()
                except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
                    return self._log(rec, 502, ('{"error": "tokenize failed: %s"}' % e)[:400].encode())
            hit = ids_stored is not None
            if hit and LOADED["key"] != key:
                try:
                    r = _api_json("/slots/0?action=restore", {"filename": "%s.slot" % key})
                    rec["restore_ms"] = (r.get("timings") or {}).get("restore_ms")
                    STORE.touch(key)
                    LOADED["key"] = key
                except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
                    rec["restore_failed"] = str(e)[:120]
                    hit = False
                    ids = full_ids()
            rec["hit"] = hit
            payload = {"prompt": ids, "n_predict": req_j.get("max_tokens", -1),
                       "cache_prompt": True, "return_tokens": True, "stream": stream}
            for p in ("temperature", "top_p", "stop"):
                if p in req_j:
                    payload[p] = req_j[p]
            try:
                ureq = urllib.request.Request(UP + "/completion", json.dumps(payload).encode(),
                                              {"Content-Type": "application/json"})
                uresp = urllib.request.urlopen(ureq, timeout=900)
            except urllib.error.HTTPError as e:
                return self._log(rec, e.code, e.read())
            except (urllib.error.URLError, OSError):
                return self._log(rec, 502, b'{"error": "upstream unreachable"}')
            SEQ += 1
            cid = "ctx2-%d" % SEQ
            created = int(time.time())
            hdr = {"X-Ctx-Cache": ("hit; restored=%d" % rec["stored"]) if hit else "miss"}
            gen_tokens, reply, timings, stop_type = [], "", None, None
            if not stream:
                with uresp:
                    try:
                        cj = json.loads(uresp.read())
                    except ValueError:
                        return self._log(rec, 502, b'{"error": "bad upstream json"}')
                reply = cj.get("content", "")
                gen_tokens = cj.get("tokens") or []
                timings = cj.get("timings")
                stop_type = cj.get("stop_type")
                rec["prompt_n"] = (timings or {}).get("prompt_n")
                rec["cache_n"] = (timings or {}).get("cache_n")
                obj = {"id": cid, "object": "chat.completion", "created": created, "model": MODEL_ID,
                       "choices": [{"index": 0, "finish_reason":
                                    "length" if stop_type == "limit" else "stop",
                                    "message": {"role": "assistant", "content": reply}}],
                       "usage": {"prompt_tokens": len(ids), "completion_tokens": len(gen_tokens),
                                 "total_tokens": len(ids) + len(gen_tokens)},
                       "timings": timings or {}}
                self._send_close_delimited(200, json.dumps(obj).encode(), hdr)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                for hk, hv in hdr.items():
                    self.send_header(hk, hv)
                self.end_headers()
                first = True
                try:
                    for rawline in uresp:                       # LINE iteration, never sized read
                        ln = rawline.decode("utf-8", "replace").strip()
                        if not ln.startswith("data: "):
                            continue
                        d = ln[6:]
                        if d == "[DONE]":
                            break
                        try:
                            ev = json.loads(d)
                        except ValueError:
                            continue
                        if ev.get("content"):
                            reply += ev["content"]
                        gen_tokens += ev.get("tokens") or []
                        if ev.get("stop_type"):
                            stop_type = ev["stop_type"]
                        t = ev.get("timings")
                        if t:
                            timings = t
                        delta = ({"role": "assistant"} if first else {})
                        if ev.get("content"):
                            delta["content"] = ev["content"]
                        ch = {"id": cid, "object": "chat.completion.chunk", "created": created,
                              "model": MODEL_ID,
                              "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                        self.wfile.write(b"data: " + json.dumps(ch).encode() + b"\n\n")
                        self.wfile.flush()
                        first = False
                except (ConnectionError, BrokenPipeError):
                    pass
                fin = "length" if stop_type == "limit" else "stop"
                last = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": fin}]}
                self.wfile.write(b"data: " + json.dumps(last).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                if timings:
                    rec["prompt_n"] = timings.get("prompt_n")
                    rec["cache_n"] = timings.get("cache_n")
                rec["stream_done"] = True
            # ---- store the grown conversation AFTER the client has every byte -------------
            convo = msgs + ([{"role": "assistant", "content": reply}] if reply else [])
            if reply and len(slotlib.canonical_json(convo)) >= MIN_CHARS:
                key2, _ent, evicted = STORE.begin(convo)
                rec["evicted"] = evicted
                try:
                    sr = _api_json("/slots/0?action=save", {"filename": "%s.slot" % key2})
                    nb = sr.get("n_written")
                    rec["save_ms"] = (sr.get("timings") or {}).get("save_ms")
                    rec["save_bytes"] = nb
                    st = STORE.finish(key2, list(ids) + list(gen_tokens), slot_bytes=nb)
                    if st == "ok":
                        LOADED["key"] = key2
                    else:
                        STORE.abort(key2)
                except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
                    rec["save_failed"] = str(e)[:120]
                    STORE.abort(key2)
        rec["elapsed_s"] = round(time.monotonic() - t_start, 3)
        self._append_log(rec)
        try:                                        # barrier: client sees EOF only AFTER the log+save
            self.wfile.flush()
        except OSError:
            pass

    def _send_close_delimited(self, code, body, hdr):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Connection", "close")
        for k, v in (hdr or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()          # bytes are the client's; the CLOSE after the log append is the barrier

    def _log(self, rec, code, body):
        rec["error_short"] = body[:80].decode("utf-8", "replace")
        rec["elapsed_s"] = round(time.monotonic(), 3)
        self._send_close_delimited(code, body, None)
        self._append_log(rec)
        return None

    @staticmethod
    def _append_log(rec):
        if LOGP:
            with open(LOGP, "a") as f:
                f.write(json.dumps(rec) + "\n")


def serve(listen, upstream, root, budget_gib, model_id, kv, min_chars, log):
    class NS:
        pass
    ns = NS()
    ns.root, ns.budget_gib, ns.model_id, ns.kv, ns.min_chars, ns.log, ns.upstream = \
        root, budget_gib, model_id, kv, min_chars, log, upstream
    _load_cfg(ns)
    host, _, port = listen.partition(":")
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer((host, int(port or 0)), Handler)
    return srv


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ctxproxy2.py")
    ap.add_argument("--listen", default="127.0.0.1:8090")
    ap.add_argument("--upstream", default="http://127.0.0.1:8099")
    ap.add_argument("--root", required=True)
    ap.add_argument("--budget-gib", type=float, default=14.0)
    ap.add_argument("--model-id", default="q36-iq2m")
    ap.add_argument("--kv", default="f16/f16")
    ap.add_argument("--min-chars", type=int, default=4000)
    ap.add_argument("--log")
    a = ap.parse_args(argv)
    host = a.listen.partition(":")[0]
    port = a.listen.partition(":")[2]
    srv = serve(a.listen, a.upstream, a.root, a.budget_gib, a.model_id, a.kv, a.min_chars, a.log)
    print("ctxproxy2 listening on %s:%d -> %s root %s (min-chars %d)"
          % (host, srv.server_address[1], a.upstream, a.root, a.min_chars), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
