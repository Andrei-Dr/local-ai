"""tests for bench/box/slotlib.py — Store in a temp root with an injected clock; no sleeping, no real
files besides the store itself. Crash simulation patches os.replace."""
import json, os, time, unittest
from pathlib import Path
from unittest import mock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "bench" / "box"))
import slotlib


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Case(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.clock = Clock()
        self.store = slotlib.Store(self.tmp, budget_bytes=1000, clock=self.clock)

    def put(self, msgs, n_msgs=None, size=100, model="m1", kv="f16", tokens=10):
        k = self.store.key_for(model, kv, msgs)
        e = self.store.reserve(model, kv, msgs, n_msgs if n_msgs is not None else len(msgs))
        f = Path(self.store.root) / e["file"]
        f.write_bytes(b"x" * size)
        self.store.commit(k, bytes=size, n_tokens=tokens)
        return k

    def test_key_stable_and_sensitive(self):
        k1 = self.store.key_for("m1", "f16", [{"role": "user", "content": "a"}])
        self.assertEqual(k1, self.store.key_for("m1", "f16", [{"role": "user", "content": "a"}]))
        self.assertEqual(len(k1), 64)
        self.assertNotEqual(k1, self.store.key_for("m2", "f16", [{"role": "user", "content": "a"}]))
        self.assertNotEqual(k1, self.store.key_for("m1", "q8_0", [{"role": "user", "content": "a"}]))
        ab = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        ba = [{"role": "assistant", "content": "b"}, {"role": "user", "content": "a"}]
        self.assertNotEqual(self.store.key_for("m1", "f16", ab), self.store.key_for("m1", "f16", ba))
        # canonical JSON: key order inside a message does not matter
        self.assertEqual(self.store.key_for("m1", "f16", [{"role": "u", "content": "x"}]),
                         self.store.key_for("m1", "f16", [{"content": "x", "role": "u"}]))

    def test_lookup_longest_prefix_and_self(self):
        a, b, c = [{"m": 1}], [{"m": 2}], [{"m": 3}]
        kab = self.put(a + b, size=110)         # a 2-message prefix slot
        kabc = self.put(a + b + c, size=110)    # the 3-message slot
        e, k = self.store.lookup("m1", "f16", a + b + c + [{"m": 4}])
        self.assertEqual((e["key"], k), (kabc, 3))     # LONGEST stored prefix wins (3 beats 2)
        e, k = self.store.lookup("m1", "f16", a + b)
        self.assertEqual((e["key"], k), (kab, 2))      # exact-prefix hit
        e, k = self.store.lookup("m1", "f16", a + [{"z": 9}])   # diverges at msg 2, no prefix stored
        self.assertIsNone(e)
        self.assertEqual(k, 0)
        e, k = self.store.lookup("m2", "f16", a + b)   # different model => miss
        self.assertIsNone(e)

    def test_missing_file_heals_index(self):
        k = self.put([{"m": 1}], size=50)
        (Path(self.tmp) / f"{k}.slot").unlink()
        e, kk = self.store.lookup("m1", "f16", [{"m": 1}])
        self.assertIsNone(e)
        self.assertNotIn(k, self.store.index["entries"])       # healed

    def test_writing_entries_invisible_unevictable(self):
        self.store.reserve("m1", "f16", [{"m": 1}], 1)
        e, k = self.store.lookup("m1", "f16", [{"m": 1}])
        self.assertIsNone(e)
        old = self.put([{"m": 9}], size=900)                   # older + huge
        self.clock.t += 1
        self.store.reserve("m1", "f16", [{"m": 2}], 1)         # newest is a writing entry
        ev = self.store.evict_to_budget()                      # 900 + 0 + extra(0) <= 1000 => nothing evicted
        self.assertEqual(ev, [])
        ev2 = self.store.evict_to_budget(extra=500)            # now over budget
        self.assertIn(old, ev2)                                # ready victim; the writing entries survive
        self.assertTrue(any(x["state"] == "writing" for x in self.store.index["entries"].values()))

    def test_lru_order_and_tiebreaks(self):
        a = self.put([{"a": 1}])
        self.clock.t += 1
        b = self.put([{"b": 1}])
        self.clock.t += 1
        c = self.put([{"c": 1}])                                 # c newest by last_used
        self.clock.t += 1
        self.store.touch(a)                                      # a now NEWER than b
        for k in (a, b, c):
            self.store.index["entries"][k]["bytes"] = 400        # over budget => eviction picks victims
        out = self.store.evict_to_budget()
        self.assertEqual(out, [b])                               # b alone is the oldest last_used
        # equal last_used: FEWER HITS loses first (then older created)
        d = self.put([{"d": 1}])
        e = self.put([{"e": 1}])
        self.store.index["entries"][d]["bytes"], self.store.index["entries"][e]["bytes"] = 300, 150
        self.store.index["entries"][d]["last_used"] = self.store.index["entries"][e]["last_used"] = 1.0
        self.store.index["entries"][d]["created"] -= 10        # d older created
        self.store.index["entries"][e]["hits"] += 5
        out = self.store.evict_to_budget()
        self.assertIn(d, out)
        self.assertNotIn(e, out)

    def test_oversize_entry_evicted_after_commit(self):
        k = self.put([{"big": 1}], size=1500)
        self.assertNotIn(k, self.store.index["entries"])
        self.assertFalse((Path(self.tmp) / f"{k}.slot").exists())

    def test_gc(self):
        k = self.put([{"a": 1}], size=10)
        (Path(self.tmp) / "orphan.slot").write_bytes(b"zz")
        dead = self.put([{"gone": 1}], size=10)
        (Path(self.tmp) / f"{dead}.slot").unlink()
        wk = self.store.key_for("m1", "f16", [{"stale": 1}])
        self.store.reserve("m1", "f16", [{"stale": 1}], 1)
        self.store.index["entries"][wk]["created"] = self.clock.t - 6 * 3600 - 1
        self.store.gc()
        self.assertFalse((Path(self.tmp) / "orphan.slot").exists())
        self.assertNotIn(dead, self.store.index["entries"])
        self.assertNotIn(wk, self.store.index["entries"])      # stale writer dropped
        self.assertIn(k, self.store.index["entries"])
        self.store.index["entries"][list(self.store.index["entries"])[0]]  # surviving entry intact
        self.assertEqual(list(self.store.index["entries"].values())[0]["state"], "ready")
        # 6 h boundary exactly: NOT stale
        w2 = self.store.reserve("m1", "f16", [{"w": 2}], 1)
        self.store.index["entries"][w2["key"]]["created"] = self.clock.t - 6 * 3600
        self.store.gc()
        self.assertIn(w2["key"], self.store.index["entries"])

    def test_atomic_write_survives_replace_failure(self):
        k = self.put([{"a": 1}], size=10)
        good = json.loads((Path(self.tmp) / "index.json").read_text())
        with mock.patch("os.replace", side_effect=OSError("boom")):
            self.assertRaises(OSError, self.store.reserve, "m1", "f16", [{"b": 2}], 1)
        self.assertEqual(json.loads((Path(self.tmp) / "index.json").read_text()), good)  # old index intact
        st2 = slotlib.Store(self.tmp, 1000, clock=self.clock)
        self.assertIn(k, st2.index["entries"])

    def test_corrupt_index_quarantined_data_kept(self):
        k = self.put([{"a": 1}], size=10)
        slotf = Path(self.tmp) / f"{k}.slot"
        (Path(self.tmp) / "index.json").write_text("{ not json")
        st2 = slotlib.Store(self.tmp, 1000, clock=self.clock)
        self.assertEqual(st2.index["entries"], {})
        q = [p for p in Path(self.tmp).iterdir() if p.name.startswith("index.json.corrupt-")]
        self.assertEqual(len(q), 1)
        self.assertTrue(slotf.exists())                        # data untouched (gc not automatic)
        self.assertIn(k + ".slot", [p.name for p in slotf.parent.iterdir()])

    def test_persistence_roundtrip_totals(self):
        k = self.put([{"a": 1}], size=100, tokens=33)
        st2 = slotlib.Store(self.tmp, 1000, clock=self.clock)
        ent = st2.index["entries"][k]
        self.assertEqual(ent["n_tokens"], 33) and self.assertEqual(ent["state"], "ready")
        self.assertEqual(st2.total_ready_bytes(), 100)
        self.store.abort(k)
        self.assertFalse((Path(self.tmp) / f"{k}.slot").exists())
        self.assertNotIn(k, self.store.index["entries"])


if __name__ == "__main__":
    unittest.main()
