"""ent1.py MODEL.gguf [MB_PER_TENSOR] -- how much LOSSLESS compression is left in the quantized expert weights.

Andrei 09-23: the expert path is bandwidth-bound (RAM -> CPU on decode, RAM -> PCIe -> GPU on prefill) while compute has slack;
could a lossless codec on top of the quant blocks buy bandwidth? This measures the ceiling. Per expert tensor kind (gate / up /
down / gate_up) and quant type, on a block-aligned random sample from every 5th layer:
  - code entropy: order-0 H(code) and order-1 H(code | previous code in the block) vs the stored code bits (Q2_K 2, Q3_K 3, Q4_K 4)
  - side data (sub-block scales, super-block d/dmin): byte / uint16 order-0 entropy
  - est = (H1 codes + H scales + raw d) / stored block bits   <- an entropy coder's ideal, no model cost
  - lzma -9 and zlib -9 on the same bytes (what a real general-purpose codec gets)
Read: a kind is WORTH A LOOK when min(est, lzma) < 0.85 of its stored size; otherwise the codec cannot pay for its decode cost.
Caveat: the d/dmin uint16 entropy is a plug-in estimate on ~50k samples of a 65,536-symbol alphabet (biased low); it is ~5% of the
bytes, so the bias moves est by < 1 pt."""
import lzma, os, re, sys, zlib
import numpy as np
import gguf

QK = 256
BLOCK = {"Q2_K": 84, "Q3_K": 110, "Q4_K": 144}
CODE_BITS = {"Q2_K": 2, "Q3_K": 3, "Q4_K": 4}


def H(counts):
    c = np.asarray(counts, dtype=np.float64)
    c = c[c > 0]
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def codes_and_side(qt, b):
    """b: (nb, block_bytes) uint8 -> codes (nb, 256) uint8, scales (nb, k) uint8, d (nb, m) uint16."""
    nb = b.shape[0]
    if qt == "Q2_K":
        scales, qs, d = b[:, 0:16], b[:, 16:80].reshape(nb, 2, 1, 32), b[:, 80:84]
        codes = np.concatenate([(qs >> (2 * j)) & 3 for j in range(4)], axis=2)          # (nb, 2, 4, 32): n*128 + j*32 + l
        return codes.reshape(nb, QK), scales, d.copy().view(np.uint16)
    if qt == "Q3_K":
        hmask, qs, scales, d = b[:, 0:32], b[:, 32:96].reshape(nb, 2, 1, 32), b[:, 96:108], b[:, 108:110]
        low = np.concatenate([(qs >> (2 * j)) & 3 for j in range(4)], axis=2)            # (nb, 2, 4, 32)
        hb = np.stack([np.stack([(hmask >> (4 * n + j)) & 1 for j in range(4)], axis=1) for n in range(2)], axis=1)
        return (low | (hb << 2)).reshape(nb, QK), scales, d.copy().view(np.uint16)
    if qt == "Q4_K":
        d, scales, qs = b[:, 0:4], b[:, 4:16], b[:, 16:144].reshape(nb, 4, 1, 32)
        codes = np.concatenate([qs & 15, qs >> 4], axis=2)                               # (nb, 4, 2, 32): j*64 + h*32 + l
        return codes.reshape(nb, QK), scales, d.copy().view(np.uint16)
    return None


def measure(qt, raw):
    bb = BLOCK[qt]
    b = raw.reshape(-1, bb)
    codes, scales, d = codes_and_side(qt, b)
    k = 1 << CODE_BITS[qt]
    h0 = H(np.bincount(codes.ravel(), minlength=k))
    pairs = codes[:, :-1].astype(np.int32) * k + codes[:, 1:]
    joint = np.bincount(pairs.ravel(), minlength=k * k).reshape(k, k)
    h1 = H(joint.ravel()) - H(joint.sum(axis=1))                                          # H(X_i, X_{i-1}) - H(X_{i-1})
    hs = H(np.bincount(scales.ravel(), minlength=256))
    hd = H(np.bincount(d.ravel(), minlength=65536))
    ideal_bits = QK * min(h0, h1) + scales.shape[1] * hs + d.shape[1] * hd
    est = ideal_bits / (bb * 8)
    data = raw.tobytes()
    return dict(h0=h0, h1=h1, hs=hs, hd=hd, est=est,
                lzma=len(lzma.compress(data, preset=9)) / len(data), zlib=len(zlib.compress(data, 9)) / len(data))


def main():
    path = sys.argv[1]
    mb = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    rd = gguf.GGUFReader(path)
    rng = np.random.default_rng(0)
    rows = {}
    for t in rd.tensors:
        m = re.match(r"blk\.(\d+)\.(ffn_(?:gate_up|gate|up|down)_exps)\.weight$", t.name)
        if not m or int(m.group(1)) % 5 != 0:
            continue
        qt = t.tensor_type.name
        if qt not in BLOCK:
            print(f"  skip {t.name}: {qt} has no code layout here", flush=True)
            continue
        raw = np.asarray(t.data).reshape(-1).view(np.uint8)
        bb = BLOCK[qt]
        nblk = raw.size // bb
        want = max(1, int(mb * 2**20) // bb)
        chunk = max(1, want // 8)
        grid = np.arange(0, nblk - chunk + 1, chunk)                                      # non-overlapping, block-aligned chunks
        starts = np.sort(rng.choice(grid, size=min(8, grid.size), replace=False))
        sample = np.concatenate([raw[s * bb:(s + chunk) * bb] for s in starts])
        r = measure(qt, sample)
        rows.setdefault((m.group(2), qt), []).append(r)
        print(f"  {t.name:32s} {qt}  H0 {r['h0']:.3f}  H1 {r['h1']:.3f} / {CODE_BITS[qt]} bits | scales {r['hs']:.2f}/8 | d {r['hd']:.1f}/16"
              f" | est {r['est']:.3f}  lzma {r['lzma']:.3f}  zlib {r['zlib']:.3f}", flush=True)
    print("--- SUMMARY (fraction of stored size; mean over sampled layers)")
    worth = False
    for (kind, qt), rs in sorted(rows.items()):
        f = {k: sum(r[k] for r in rs) / len(rs) for k in rs[0]}
        best = min(f["est"], f["lzma"])
        w = best < 0.85
        worth |= w
        print(f"  {kind:16s} {qt}  codes H1 {f['h1']:.3f}/{CODE_BITS[qt]}  est {f['est']:.3f}  lzma {f['lzma']:.3f}  zlib {f['zlib']:.3f}"
              f"  -> {'WORTH A LOOK' if w else 'DEAD'} ({len(rs)} layers)")
    print(f"VERDICT: {'a lossless codec has room on at least one kind' if worth else 'no kind below 0.85: lossless expert compression is DEAD'}")
    print("ENT1_DONE")


if __name__ == "__main__":
    main()
