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
    print("ctxproxy2 part1: store ready at %s (request flow lands in part 2)" % a.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
