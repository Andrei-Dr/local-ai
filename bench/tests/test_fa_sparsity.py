"""offline tests for fa_sparsity: q4_0 / q8_0 row decoding, strided K views, bounds are upper bounds, a peaked head is seen."""
import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "box"))
import fa_sparsity as fs


def q4_0_block(d, nibbles):
    lo, hi = np.array(nibbles[:16], dtype=np.uint8), np.array(nibbles[16:], dtype=np.uint8)
    return np.concatenate([np.array([d], dtype=np.float16).view(np.uint8), lo | (hi << 4)])


class Decode(unittest.TestCase):
    def test_q4_0_low_nibbles_first(self):
        nib = list(range(16)) + [15 - i for i in range(16)]
        row = q4_0_block(0.5, nib)
        x = fs.dequant_rows(row[None, :], "q4_0", 32)[0]
        np.testing.assert_allclose(x, (np.array(nib, dtype=np.float32) - 8) * 0.5)

    def test_q8_0(self):
        q = np.arange(-16, 16, dtype=np.int8)
        row = np.concatenate([np.array([0.25], dtype=np.float16).view(np.uint8), q.view(np.uint8)])
        np.testing.assert_allclose(fs.dequant_rows(row[None, :], "q8_0", 32)[0], q.astype(np.float32) * 0.25)

    def test_strided_view_is_position_major(self):
        D, n_kv, n_hkv = 32, 3, 2
        rows = {(h, p): q4_0_block(1.0, [(h * 7 + p + i) % 16 for i in range(32)]) for h in range(n_hkv) for p in range(n_kv)}
        raw = np.concatenate([rows[(h, p)] for p in range(n_kv) for h in range(n_hkv)])     # cache layout: position, then head
        path = Path(self.id() + ".bin"); raw.tofile(path)
        try:
            k = fs.load_k(str(path), {"type": "q4_0", "ne": [D, n_kv, n_hkv, 1], "nb": [18, 36, 18, 108]}, D)
        finally:
            path.unlink()
        self.assertEqual(k.shape, (n_hkv, n_kv, D))
        np.testing.assert_allclose(k[1, 2], fs.dequant_rows(rows[(1, 2)][None, :], "q4_0", D)[0])


class Analysis(unittest.TestCase):
    def test_hadamard_is_orthonormal(self):
        x = np.random.default_rng(0).normal(size=(5, 64)).astype(np.float32)
        np.testing.assert_allclose(fs.hadamard(fs.hadamard(x)), x, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(fs.hadamard(x), axis=1), np.linalg.norm(x, axis=1), rtol=1e-5)

    def test_bounds_hold_and_a_peaked_group_is_skippable(self):
        rng = np.random.default_rng(1)
        D, T = 64, 12
        k = rng.normal(size=(1, T * fs.TILE, D)).astype(np.float32) * 0.1
        q = rng.normal(size=(2, D)).astype(np.float32)
        k[0, 5] = q[0] * 8; k[0, 6] = q[1] * 8                 # one hot key per head, both inside tile 0
        r = fs.analyze(q, k, T * fs.TILE, 1.0)[0]              # analyze() asserts every bound is an upper bound
        self.assertGreater(r["oracle"][1e-8], 0.8)
        self.assertGreater(r["mass_top1pct"], 0.99)
        self.assertGreater(r["bounds"]["ball"][1e-8], 0.5)

    def test_a_diffuse_head_blocks_the_whole_group(self):
        rng = np.random.default_rng(2)
        D, T = 64, 8
        k = rng.normal(size=(1, T * fs.TILE, D)).astype(np.float32) * 0.1
        q = rng.normal(size=(2, D)).astype(np.float32); q[1] *= 0.01            # head 1 attends almost uniformly
        k[0, 5] = q[0] * 8
        r = fs.analyze(q, k, T * fs.TILE, 1.0)[0]
        self.assertEqual(r["oracle"][1e-8], 0.0)
        self.assertGreater(r["oracle_per_head"][1e-8], 0.3)


if __name__ == "__main__":
    unittest.main()
