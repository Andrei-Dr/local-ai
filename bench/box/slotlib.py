#!/usr/bin/env python3
"""slotlib.py — the store layer behind ctxproxy.py: an LRU, content-addressed KV-slot store on disk.

A "slot" is a llama.cpp slot-save file: the model state after prefilling some message prefix, restorable
with one call instead of re-prefilling (30 s at 8k, minutes at 32k+). Content addressing means the SAME
(model, kv-config, messages-prefix) always maps to the SAME key, so slots survive restarts and dedupe by
construction; longest-prefix lookup turns "does any stored prefix of this request exist?" into a hash
lookup per prefix length. Slot files sit next to index.json (NO subdir). This module knows NOTHING about
HTTP — the store primitive stands alone (ctxproxy drives it; tests drive it directly).

key formula (the ctxproxy test-suite dependency): key = sha256(canonical_json({"model": model_id, "kv":
kv, "messages": messages[:k]})).hexdigest(), canonical_json = json.dumps(obj, sort_keys=True,
separators=(",", ":")).

index.json layout: {"entries": {key: {key, file, bytes, n_tokens, n_msgs, model, kv, state, created,
last_used, hits}}} — state "ready" (committed, visible, evictable) or "writing" (reserved, NOT visible to
lookup, NEVER evicted; a writing entry older than 6 h is stale and gc drops it). Writes are atomic:
index.json.tmp + os.replace, so a crash mid-write leaves the previous index intact.

CLI: slotlib.py ROOT list | gc | evict BYTES | rm KEY
"""
import json, os, sys, time
from hashlib import sha256
from pathlib import Path

WRITING_STALE_S = 6 * 3600
KV = "f16"                                                 # process-wide identity of THIS deployment's KV


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


class Store:
    def __init__(self, root, budget_bytes, clock=time.monotonic):
        """Loads root/index.json (missing = new store; corrupt = quarantined aside as
        index.json.corrupt-<ts>, data files deliberately KEPT for a human gc pass — silently deleting a
        terabyte because one JSON byte rotted is worse than stranded bytes)."""
        self.root, self.budget, self.clock = Path(root), int(budget_bytes), clock
        self.on_evict = []                               # callables(keys) informed when files are deleted
        self.root.mkdir(parents=True, exist_ok=True)
        self.jp = self.root / "index.json"
        self.index = {"entries": {}}
        if self.jp.exists():
            try:
                self.index = json.loads(self.jp.read_text())
                if not isinstance(self.index, dict) or not isinstance(self.index.get("entries"), dict):
                    raise ValueError("shape")
            except (ValueError, OSError) as e:
                q = self.root / f"index.json.corrupt-{int(time.time())}"
                os.replace(self.jp, q)
                self.index = {"entries": {}}
                print(f"SLOTLIB: index corrupt ({e}) — quarantined to {q.name}; slot files kept (run gc)")

    # ---- identity --------------------------------------------------------------
    def key_for(self, model_id, kv, messages):
        return sha256(canonical_json({"model": model_id, "kv": kv, "messages": messages}).encode()).hexdigest()

    def file_for(self, key):
        return self.root / f"{key}.slot"

    # ---- write path ------------------------------------------------------------
    def reserve(self, model_id, kv, messages, n_msgs):
        """Register state="writing" (invisible, unelectable for eviction) BEFORE the server saves; returns
        the entry. Caller fills the file then commit()s (or abort())."""
        key = self.key_for(model_id, kv, messages)
        now = self.clock()
        ent = {"key": key, "file": f"{key}.slot", "bytes": 0, "n_tokens": 0, "n_msgs": n_msgs,
               "model": model_id, "kv": kv, "state": "writing", "created": now, "last_used": now, "hits": 0}
        self.index["entries"][key] = ent
        self._persist()
        return ent

    def commit(self, key, bytes=None, n_tokens=None):
        """Flip the entry to ready with its real sizes; an entry bigger than the WHOLE budget is evicted
        immediately (returns ("evicted", None)), otherwise returns ("ok", entry)."""
        ent = self.index["entries"].get(key)
        if ent is None:
            return ("missing", None)
        ent["state"] = "ready"
        if bytes is not None:
            ent["bytes"] = int(bytes)
        if n_tokens is not None:
            ent["n_tokens"] = int(n_tokens)
        if ent["bytes"] > self.budget:
            self._drop(key, unlink=True)
            self._persist()
            for f in self.on_evict:
                f([key])
            return ("evicted", None)
        self._enforce_best_effort()
        self._persist()
        return ("ok", ent)

    def abort(self, key):
        """Roll back a reservation whose save never happened / died: entry and half-written file go."""
        if self._drop(key, unlink=True):
            self._persist()

    def touch(self, key):
        ent = self.index["entries"].get(key)
        if ent:
            ent["last_used"] = self.clock()
            ent["hits"] = ent.get("hits", 0) + 1
            self._persist()
        return ent

    # ---- read path ---------------------------------------------------------------
    def lookup(self, model_id, kv, messages):
        """LONGEST stored ready prefix: (entry, k) for the largest k<=len(messages) whose key is ready AND
        whose file exists, else (None, 0). Ready entry with MISSING file => evicted from the index as
        corruption (heal) and the scan continues to shorter prefixes."""
        for k in range(len(messages), 0, -1):
            key = self.key_for(model_id, kv, messages[:k])
            ent = self.index["entries"].get(key)
            if not ent or ent.get("state") != "ready":
                continue
            if not self.file_for(key).exists():
                self._drop(key, unlink=False)
                self._persist()
                continue
            return ent, k
        return None, 0

    # ---- maintenance -------------------------------------------------------------
    def victims(self):
        ready = [e for e in self.index["entries"].values() if e.get("state") == "ready"]
        return sorted(ready, key=lambda e: (e.get("last_used", 0), e.get("hits", 0), e.get("created", 0)))

    def evict_to_budget(self, extra=0):
        """Delete LRU-first until sum(ready bytes)+extra <= budget; return evicted keys. Ties: fewer hits
        first, then OLDER created. "writing" entries are never picked."""
        out = []
        while self.total_ready_bytes() + extra > self.budget:
            vic = self.victims()
            if not vic:
                break
            self._drop(vic[0]["key"], unlink=True)
            out.append(vic[0]["key"])
        if out:
            self._persist()
            for f in self.on_evict:
                f(out)
        return out

    def gc(self):
        """Reconcile index with root/: delete *.slot with no entry, drop entries whose file vanished, drop
        stale "writing" entries (> WRITING_STALE_S since created, strictly). Safe to run anytime."""
        now = self.clock()
        changed = False
        for key, ent in list(self.index["entries"].items()):
            if ent.get("state") != "ready":
                if now - ent.get("created", now) > WRITING_STALE_S:
                    self._drop(key, unlink=True)
                    changed = True
            elif not self.file_for(key).exists():
                self._drop(key, unlink=False)
                changed = True
        indexed = {e["file"] for e in self.index["entries"].values()}
        for p in self.root.glob("*.slot"):
            if p.name not in indexed:
                p.unlink()
                changed = True
        if changed:
            self._persist()

    def total_ready_bytes(self):
        return sum(e.get("bytes", 0) for e in self.index["entries"].values() if e.get("state") == "ready")

    # ---- internals ----------------------------------------------------------------
    def _drop(self, key, unlink=False):
        ent = self.index["entries"].pop(key, None)
        if unlink and ent:
            try:
                self.file_for(key).unlink()
            except FileNotFoundError:
                pass
        return ent is not None

    def _enforce_best_effort(self):
        out = []
        while self.total_ready_bytes() > self.budget:
            vic = self.victims()
            if not vic:
                break
            self._drop(vic[0]["key"], unlink=True)
            out.append(vic[0]["key"])
        if out:                                        # same contract as evict_to_budget: deletions
            for f in self.on_evict:                    # must reach the on_evict hooks (sidecars etc.)
                f(out)

    def _persist(self):
        tmp = self.root / "index.json.tmp"
        tmp.write_text(json.dumps(self.index))
        os.replace(tmp, self.jp)                                 # atomic swap; failure => old index intact


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__.split("CLI:")[1].strip())
    store = Store(sys.argv[1], budget_bytes=int(os.environ.get("SLOT_BUDGET_BYTES", 4 * 1024**3)))
    cmd = sys.argv[2]
    if cmd == "list":
        for e in reversed(store.victims()):
            print(f"{e['state']:7} hits {e['hits']:4} used {e['last_used']:.0f} {e['bytes']:>12} B "
                  f"msgs {e['n_msgs']:3} tok {e['n_tokens']:>9} {e['model']} {e['kv']} {e['key']}")
        print(f"TOTAL_READY {store.total_ready_bytes()} B / budget {store.budget}")
    elif cmd == "gc":
        store.gc()
        print(f"GC_OK entries {len(store.index['entries'])}")
    elif cmd == "evict":
        store.budget = int(sys.argv[3])
        keys = store.evict_to_budget()
        print(f"EVICTED {keys}")
    elif cmd == "rm":
        store.abort(sys.argv[3])
        print("RM_DONE")
    else:
        sys.exit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
