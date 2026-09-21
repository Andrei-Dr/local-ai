#!/usr/bin/env python3
"""fa_sparsity.py DUMP_DIR: how sparse is real long-context attention, and which per-tile bound can PROVE it?

Input = the files written by the fa-dump diagnostic patch (GGML_CUDA_FA_DUMP): per attention node and decode step a Q
(F32 [D, 1, n_head]), the mask (F16, one row) and, at the last step, the K cache view (q4_0 / q8_0 / f16 [D, n_kv, n_head_kv],
any strides). K is stored ROTATED (llama.cpp attention rotation) and Q arrives rotated to match, so q.k is what the kernel sees.

Per node and step, per KV-head GROUP (the n_head / n_head_kv query heads a GQA kernel serves in one walk), with
rel = exp(logit - max logit of that query head) and tiles of TILE positions:
  oracle   share of tiles whose largest rel over all positions and all heads of the group is < thr: the ceiling for ANY
           tile-skipping scheme, for thr in THRS. Also the share of softmax mass in the top 1% / 5% of tiles.
  bounds   share of tiles a bound PROVES skippable at thr, against a cheap lower bound of the max logit (first tile + last two
           tiles computed exactly):  minmax = sum_c max(q_c*kmin_c, q_c*kmax_c) (Quest) on the stored channels;
           ball = q.centroid + |q|*radius (rotation invariant);  minmax_h = minmax after a plain Sylvester-Hadamard transform
           of q and k (tells whether un-rotating the cache would tighten the channel bound).
Prints one line per (node, step, group) and a summary; numpy only. Exit 2 on unreadable input."""
import glob, json, math, os, sys
import numpy as np

TILE = 128
THRS = (1e-8, 1e-6, 1e-4)


def dequant_rows(raw, typ, D):
    """raw: uint8 [..., row_bytes] -> float32 [..., D]. q4_0: 18-byte blocks (f16 d, 16 bytes: low nibbles = elements 0..15,
    high nibbles = 16..31, value = (nibble - 8) * d). q8_0: 34-byte blocks (f16 d, 32 int8)."""
    lead = raw.shape[:-1]
    if typ == "f16":
        return raw.view(np.float16).astype(np.float32).reshape(*lead, D)
    if typ == "f32":
        return raw.view(np.float32).reshape(*lead, D)
    nb = D // 32
    if typ == "q4_0":
        b = raw.reshape(*lead, nb, 18)
        d = b[..., :2].copy().view(np.float16).astype(np.float32)            # [..., nb, 1]
        qs = b[..., 2:]
        vals = np.concatenate([qs & 0x0F, qs >> 4], axis=-1).astype(np.float32) - 8.0
        return (vals * d).reshape(*lead, D)
    if typ == "q8_0":
        b = raw.reshape(*lead, nb, 34)
        d = b[..., :2].copy().view(np.float16).astype(np.float32)
        return (b[..., 2:].view(np.int8).astype(np.float32) * d).reshape(*lead, D)
    raise ValueError("unsupported K type %s" % typ)


def row_bytes(typ, D):
    return {"f16": 2 * D, "f32": 4 * D, "q4_0": D // 32 * 18, "q8_0": D // 32 * 34}[typ]


def load_k(path, m, D):
    """K view [D, n_kv, n_head_kv] with arbitrary strides -> float32 [n_head_kv, n_kv, D]."""
    raw = np.fromfile(path, dtype=np.uint8)
    n_kv, n_hkv = m["ne"][1], m["ne"][2]
    rb = row_bytes(m["type"], D)
    off = (np.arange(n_hkv)[:, None] * m["nb"][2] + np.arange(n_kv)[None, :] * m["nb"][1])[..., None] + np.arange(rb)
    return dequant_rows(raw[off], m["type"], D)


def hadamard(x):
    """plain Sylvester-Hadamard over the last axis (power-of-2 length), orthonormal."""
    n = x.shape[-1]
    y = x.astype(np.float32).copy()
    h = 1
    while h < n:
        y = y.reshape(*y.shape[:-1], n // (2 * h), 2, h)
        y = np.concatenate([y[..., 0, :] + y[..., 1, :], y[..., 0, :] - y[..., 1, :]], axis=-1).reshape(*y.shape[:-3], n)
        h *= 2
    return y / math.sqrt(n)


def tile_bounds(q, kt):
    """q [H, D] (already scaled), kt [T, TILE, D] -> upper bounds [H, T] on q.k over each tile: (minmax, ball)."""
    kmin, kmax = kt.min(axis=1), kt.max(axis=1)                                  # [T, D]
    minmax = np.maximum(q[:, None, :] * kmin[None], q[:, None, :] * kmax[None]).sum(-1)
    cen = kt.mean(axis=1)
    rad = np.sqrt(((kt - cen[:, None, :]) ** 2).sum(-1)).max(axis=1)             # [T]
    ball = q @ cen.T + np.linalg.norm(q, axis=1)[:, None] * rad[None]
    return minmax, ball


def analyze(q, k, n_valid, scale):
    """q [n_head, D], k [n_head_kv, n_kv, D]; positions >= n_valid are masked. -> list of per-group dicts."""
    n_head, D = q.shape
    n_hkv = k.shape[0]
    g = n_head // n_hkv
    T = n_valid // TILE                      # whole tiles only; the ragged tail is always computed exactly
    out = []
    for kv in range(n_hkv):
        qs = q[kv * g:(kv + 1) * g] * scale
        kk = k[kv, :T * TILE]
        logits = qs @ kk.T                                                        # [g, T*TILE]
        mx = logits.max(axis=1, keepdims=True)
        rel = np.exp(logits - mx).reshape(g, T, TILE)
        tile_rel = rel.max(axis=2)                                                # [g, T]
        grp = tile_rel.max(axis=0)                                                # [T]: the group needs the tile if ANY head does
        mass = rel.sum(axis=2) / rel.sum(axis=(1, 2), keepdims=True).reshape(g, 1)
        order = np.sort(mass, axis=1)[:, ::-1]
        r = {"kv_head": kv, "tiles": T,
             "oracle": {t: float((grp < t).mean()) for t in THRS},
             "oracle_per_head": {t: float((tile_rel < t).mean()) for t in THRS},
             "mass_top1pct": float(order[:, :max(1, T // 100)].sum(axis=1).mean()),
             "mass_top5pct": float(order[:, :max(1, T // 20)].sum(axis=1).mean())}
        kt = kk.reshape(T, TILE, D)
        exact = [0, T - 2, T - 1] if T >= 3 else list(range(T))
        m_lb = logits.reshape(g, T, TILE)[:, exact].max(axis=(1, 2))              # [g] lower bound of each head's max logit
        bounds = dict(zip(("minmax", "ball"), tile_bounds(qs, kt)))
        bounds["minmax_h"] = tile_bounds(hadamard(qs), hadamard(kt))[0]
        r["bounds"] = {}
        for name, ub in bounds.items():
            assert (ub + 1e-3 >= logits.reshape(g, T, TILE).max(axis=2)).all(), "bound %s is not an upper bound" % name
            gap = ub - m_lb[:, None]                                              # log of the largest possible rel weight
            r["bounds"][name] = {t: float((gap.max(axis=0) < math.log(t)).mean()) for t in THRS}
        out.append(r)
    return out


def main():
    d = sys.argv[1]
    metas = sorted(glob.glob(os.path.join(d, "*.json")))
    if not metas:
        print("fa_sparsity: no dumps in %s" % d); sys.exit(2)
    ks, rows = {}, []
    for mp in metas:                                   # K is written at the last step only; one per node
        m = json.load(open(mp))
        if m["has_k"]:
            ks[m["node"]] = load_k(mp[:-5] + ".k.bin", m["k"], m["q"]["ne"][0])
    for mp in metas:
        m = json.load(open(mp)); node = m["node"]
        if node not in ks:
            continue
        D, n_head = m["q"]["ne"][0], m["q"]["ne"][2]
        q = np.fromfile(mp[:-5] + ".q.bin", dtype=np.float32).reshape(n_head, D)
        mask = np.fromfile(mp[:-5] + ".mask.bin", dtype=np.float16)
        n_valid = int((mask[: m["k"]["ne"][1]] > -1e4).sum()) if mask.size else m["k"]["ne"][1]
        for r in analyze(q, ks[node], min(n_valid, ks[node].shape[1]), m["scale"]):
            rows.append(r)
            fmt = lambda dct: " ".join("%.0e:%5.1f%%" % (t, 100 * v) for t, v in dct.items())
            print("%-14s s%-3d kv%d tiles %4d | oracle %s | per-head %s | mass top1%% %.3f top5%% %.3f | minmax %s | ball %s | minmax_h %s"
                  % (node, m["step"], r["kv_head"], r["tiles"], fmt(r["oracle"]), fmt(r["oracle_per_head"]), r["mass_top1pct"],
                     r["mass_top5pct"], fmt(r["bounds"]["minmax"]), fmt(r["bounds"]["ball"]), fmt(r["bounds"]["minmax_h"])), flush=True)
    if rows:
        avg = lambda f: 100 * float(np.mean([f(r) for r in rows]))
        print("FA_SPARSITY n=%d | tiles skippable at 1e-8: oracle %.1f%% (per head %.1f%%) | proven: minmax %.1f%% ball %.1f%% minmax_h %.1f%% | at 1e-4: oracle %.1f%% minmax %.1f%% ball %.1f%%"
              % (len(rows), avg(lambda r: r["oracle"][1e-8]), avg(lambda r: r["oracle_per_head"][1e-8]), avg(lambda r: r["bounds"]["minmax"][1e-8]),
                 avg(lambda r: r["bounds"]["ball"][1e-8]), avg(lambda r: r["bounds"]["minmax_h"][1e-8]), avg(lambda r: r["oracle"][1e-4]),
                 avg(lambda r: r["bounds"]["minmax"][1e-4]), avg(lambda r: r["bounds"]["ball"][1e-4])))


if __name__ == "__main__":
    main()
