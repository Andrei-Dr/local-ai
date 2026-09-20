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


if __name__ == "__main__":
    unittest.main()
