#!/usr/bin/env python3
"""expert_mix.py — EMULATE a hot/cold per-expert precision split inside ONE GGUF, so kld1 can price the
idea before anybody writes loader/kernel work. Runs on the box with /ai/.venv/bin/python (gguf-py + numpy
present) — the one stdlib-exception tool besides gguf_graft_mtp.py, whose reader/writer idioms this uses.

mix:  expert_mix.py --hi HI.gguf --lo LO.gguf --hot HOT.json --out OUT.gguf [--container Q8_0]
                      [--layers A-B] [--dry-run]
  HI / LO = two quantizations of the SAME model (same names + shapes, types differ). HOT.json =
  {"<layer>": [expert ids]}. Expert tensors (blk.L.ffn_{gate,up,down}_exps.weight) are dequantized from
  both files to float32, merged per expert along the LAST axis (HI for listed ids, LO elsewhere),
  requantized to --container (default Q8_0, ~free error) and written. Every other tensor is copied from
  LO byte-for-byte, type unchanged. Layers missing from HOT.json or outside --layers take LO wholesale
  (not even dequantized). Metadata copies from LO plus expert_mix.{hi_file,lo_file,hot_fraction,
  container}. One fused tensor is materialized at a time (~1.5 GiB float32 peak; never two layers).
  Prints per-layer hot counts and effective bits per expert weight =
  hot_fraction x bpw(HI) + (1 - hot_fraction) x bpw(LO). --dry-run computes all of it without writing.

hot:  expert_mix.py hot --trace F.bin [--trace G.bin ...] --frac 0.25 --out HOT.json
  stdlib only: traces in the research/scripts/moe-cache-sim/sim.py format (records of int32 layer,
  n_tokens, k, then n_tokens*k int32 expert ids); per layer take the top --frac experts by summed count
  over all traces (ties: lower id first), emit {"<layer>": [ids ascending]}.

Errors exit 2 with a message (shape mismatch, expert id >= n_expert, unsupported container, bad args).
"""
import argparse, array, gc, json, os, re, struct, sys
from collections import defaultdict

EXP_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
HAVE_DEPS = True
try:
    import numpy as np
    import gguf
    import gguf.quants  # noqa: F401
except ImportError:
    HAVE_DEPS = False


class ToolError(Exception):
    pass


def parse_layers(spec):
    if not spec:
        return None
    m = re.fullmatch(r"(\d+)-(\d+)", spec)
    if not m:
        raise ToolError("--layers wants A-B (inclusive), got %r" % spec)
    return int(m.group(1)), int(m.group(2))


# ---------------------------------------------------------------- hot mode (stdlib)
def hot_sets(trace_paths, frac):
    counts = defaultdict(lambda: defaultdict(int))     # layer -> expert -> count
    mx = {}
    for p in trace_paths:
        data = open(p, "rb").read()
        off = 0
        a = array.array("i")
        while off + 12 <= len(data):
            layer, n_tok, k = struct.unpack_from("<iii", data, off)
            off += 12
            n = n_tok * k
            if off + 4 * n > len(data):
                raise ToolError("%s: truncated record at %d" % (p, off))
            a.frombytes(data[off:off + 4 * n])
            off += 4 * n
            lc = counts[layer]
            for eid in a:
                lc[eid] += 1
                if eid > mx.get(layer, 0):
                    mx[layer] = eid
            a = array.array("i")
    out = {}
    for layer in sorted(counts):
        n_expert = mx[layer] + 1
        want = max(1, int(round(frac * n_expert)))
        order = sorted(counts[layer].items(), key=lambda kv: (-kv[1], kv[0]))
        out[str(layer)] = sorted(eid for eid, _ in order[:want])
    return out


def run_hot(argv):
    ap = argparse.ArgumentParser(prog="expert_mix.py hot")
    ap.add_argument("--trace", action="append", required=True)
    ap.add_argument("--frac", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if not (0 < a.frac <= 1):
        raise ToolError("--frac must be in (0, 1]")
    hs = hot_sets(a.trace, a.frac)
    json.dump(hs, open(a.out, "w"), indent=0)
    tot = sum(len(v) for v in hs.values())
    print("hot set: %d layers, %d expert ids total (frac %.2f)" % (len(hs), tot, a.frac))
    return 0


# ---------------------------------------------------------------- mix mode (gguf-py + numpy)
FLOAT_CONTAINERS = {"F16", "F32"}


def bpw_of(tensor):
    ne = int(tensor.n_elements) if hasattr(tensor, "n_elements") else int(np.prod(tensor.data.shape))
    return 8.0 * tensor.n_bytes / ne if ne else 0.0


def tl_n_experts(t):
    """Size of the LAST numpy axis of the element-space view (the expert axis by contract). Float and
    F16/F32-structured paths need no dequantize; block types take the traits path."""
    if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F64):
        return int(np.asarray(t.data).shape[-1])
    if t.tensor_type == gguf.GGMLQuantizationType.F16:
        return int(np.asarray(t.data.view(np.float16), dtype=np.float16).shape[-1]) \
            if t.data.dtype.names else int(np.asarray(t.data).shape[-1])
    return int(np.asarray(gguf.quants.dequantize(t.data, t.tensor_type)).shape[-1])


def elem_arr(t):
    """Element-space float32 view however gguf-py 0.19 presents the type (quant traits paths already
    return the element-shaped array; float branches a same-shaped view). Never reshape blindly."""
    return np.asarray(gguf.quants.dequantize(t.data, t.tensor_type), dtype=np.float32)


def byte_shape_of(shape, cont_t):
    fn = getattr(gguf.quants, "quant_shape_to_byte_shape", None)
    if fn is None:
        raise ToolError("installed gguf-py lacks quant_shape_to_byte_shape (need >= 0.10)")
    return tuple(fn(tuple(shape), cont_t))


def mix(args, err=sys.stderr):
    if not HAVE_DEPS:
        raise ToolError("mix needs gguf + numpy (run on the box with /ai/.venv/bin/python)")
    hotmap = {int(k): v for k, v in json.load(open(args.hot)).items()}
    rng = parse_layers(args.layers)
    cont_t = getattr(__import__("gguf").GGMLQuantizationType, args.container, None) \
        if args.container in dir(__import__("gguf").GGMLQuantizationType) else None
    if cont_t is None:
        names = [m.name for m in __import__("gguf").GGMLQuantizationType]
        raise ToolError("container %r is not a ggml type this gguf-py knows (%s)" % (args.container, names))
    rm, rl = gguf.GGUFReader(args.hi), gguf.GGUFReader(args.lo)
    arch = (rl.fields.get("general.architecture").contents()
            if "general.architecture" in rl.fields else None)
    if not arch:
        raise ToolError("LO has no general.architecture")
    hi_t = {t.name: t for t in rm.tensors}
    lo_t = {t.name: t for t in rl.tensors}
    for name, tl in lo_t.items():
        th = hi_t.get(name)
        if th is None or list(th.data.shape) != list(tl.data.shape):
            raise ToolError("HI/LO tensor mismatch: %s (%s vs %s)"
                            % (name, list(th.data.shape) if th else "ABSENT", list(tl.data.shape)))
    if args.dry_run:
        w = None
    else:
        w = gguf.GGUFWriter(args.out, arch, endianess=rm.endianess)
        for name, f in rl.fields.items():
            if name == "general.architecture" or name.startswith("GGUF."):
                continue
            vt = f.types[0]
            st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
            w.add_key_value(name, f.contents(), vt, sub_type=st)
        w.add_string("expert_mix.hi_file", os.path.basename(args.hi))
        w.add_string("expert_mix.lo_file", os.path.basename(args.lo))
        w.add_string("expert_mix.container", args.container)
    mixed_total = hot_total = expert_slots = 0
    bpw_hi = bpw_lo = 0.0
    order = [t.name for t in rl.tensors]
    for name in order:
        tl = lo_t[name]
        m = EXP_RE.match(name)
        layer = int(m.group(1)) if m else None
        in_range = m and (rng is None or (rng[0] <= layer <= rng[1]))
        ids = hotmap.get(layer, []) if in_range else []
        n_expert = tl_n_experts(tl)
        if ids and max(ids) >= n_expert:
            raise ToolError("layer %d: expert id %d >= n_expert %d" % (layer, max(ids), n_expert))
        if in_range and (ids or m):
            print("blk.%d %s: hot %d/%d" % (layer, m.group(2) if m else "?", len(ids), n_expert))
        if not in_range:                                  # identity copy from LO (even expert tensors)
            if w:
                w.add_tensor(name, tl.data, raw_shape=list(tl.data.shape), raw_dtype=tl.tensor_type)
            continue
        bh = hi_t[name]
        if not ids:                                        # layer takes LO wholesale, container-free
            if w:
                w.add_tensor(name, tl.data, raw_shape=list(tl.data.shape), raw_dtype=tl.tensor_type)
            expert_slots += n_expert
            mixed_total += 1
            continue
        if not bpw_hi:
            bpw_hi, bpw_lo = bpw_of(bh), bpw_of(tl)
        mh, ml = elem_arr(bh), elem_arr(tl)
        if mh.shape != ml.shape:
            raise ToolError("HI/LO element shape mismatch on %s: %s vs %s" % (name, mh.shape, ml.shape))
        mixed = ml.copy()
        for e in ids:                                      # expert axis = LAST numpy axis (brief contract)
            mixed[(Ellipsis, e)] = mh[(Ellipsis, e)]
        del mh, ml
        gc.collect()
        expert_slots += n_expert
        hot_total += len(ids)
        mixed_total += 1
        if w:
            if args.container in FLOAT_CONTAINERS:
                dt = np.float16 if args.container == "F16" else np.float32
                w.add_tensor(name, np.asarray(mixed, dtype=dt))
            else:
                cb = gguf.quants.quantize(np.ascontiguousarray(mixed, dtype=np.float32).ravel(), cont_t)
                w.add_tensor(name, np.frombuffer(cb, dtype=np.uint8),
                             raw_shape=list(byte_shape_of(mixed.shape, cont_t)), raw_dtype=cont_t)
                del cb
        del mixed
        gc.collect()
    if args.dry_run or w is None:
        hf = hot_total / expert_slots if expert_slots else 0.0
        eff = hf * bpw_hi + (1 - hf) * bpw_lo
        print("DRY RUN hot fraction %.4f  effective bpw %.3f  (hi %.3f lo %.3f)"
              % (hf, eff, bpw_hi, bpw_lo))
        return 0
    w.add_float32("expert_mix.hot_fraction", hot_total / expert_slots if expert_slots else 0.0)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    hf = hot_total / expert_slots if expert_slots else 0.0
    eff = hf * bpw_hi + (1 - hf) * bpw_lo
    print("mixed %d fused expert tensors, %d hot expert slots of %d (fraction %.4f)"
          % (mixed_total, hot_total, expert_slots, hf))
    print("effective bits per expert weight %.3f = %.4f x %.3f + %.4f x %.3f   out bytes %s"
          % (eff, hf, bpw_hi, 1 - hf, bpw_lo, os.path.getsize(args.out)))
    return 0


def main(argv=None, out=sys.stdout, err=sys.stderr):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "hot":
        try:
            return run_hot(argv[1:])
        except ToolError as e:
            print("ERROR: %s" % e, file=err)
            return 2
    ap = argparse.ArgumentParser(prog="expert_mix.py")
    ap.add_argument("--hi"); ap.add_argument("--lo"); ap.add_argument("--hot")
    ap.add_argument("--out"); ap.add_argument("--container", default="Q8_0")
    ap.add_argument("--layers"); ap.add_argument("--dry-run", action="store_true")
    try:
        a = ap.parse_args(argv)
        if not (a.hi and a.lo and a.hot and (a.out or a.dry_run)):
            raise ToolError("mix needs --hi --lo --hot and --out (or --dry-run)")
        return mix(a, err)
    except ToolError as e:
        print("ERROR: %s" % e, file=err)
        return 2
    except Exception as e:                                  # gguf/io errors -> same exit contract
        print("ERROR: %s" % e, file=err)
        return 2


if __name__ == "__main__":
    sys.exit(main())
