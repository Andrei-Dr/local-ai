#!/usr/bin/env python3
"""gguf_graft_mtp.py MAIN.gguf HEAD.gguf OUT.gguf [--gguf-py DIR]

Graft a standalone MTP head (the nextn block(s) of the same base model, shipped as its own GGUF with block_count = N+1
and nextn_predict_layers = 1) into the main model file, so llama.cpp runs MTP against the TARGET model
(`--spec-type draft-mtp` with no `-md`): one `output.weight` and one `token_embd` instead of two copies, and the head's
experts follow the main model's `-ot exps=CPU` placement (and its expert cache) instead of sitting in VRAM.

Tensor data is copied raw (no requantization). Metadata: everything from MAIN, except the architecture hparams
(`<arch>.*`), which come from HEAD because HEAD already describes the N+1-block model; every key whose value changes is
printed. Refuses to run when the two files disagree on architecture, vocabulary size, embedding width, or when a grafted
tensor name already exists in MAIN."""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("main"); ap.add_argument("head"); ap.add_argument("out")
    ap.add_argument("--gguf-py", default=os.environ.get("GGUF_PY", "/ai/src/llama.cpp-mainline/gguf-py"))
    a = ap.parse_args()
    sys.path.insert(0, a.gguf_py)
    import gguf

    rm, rh = gguf.GGUFReader(a.main), gguf.GGUFReader(a.head)
    val = lambda r, k: r.fields[k].contents() if k in r.fields else None
    arch = val(rm, gguf.Keys.General.ARCHITECTURE)
    if arch != val(rh, gguf.Keys.General.ARCHITECTURE):
        sys.exit(f"architecture mismatch: {arch} vs {val(rh, gguf.Keys.General.ARCHITECTURE)}")
    n_main, n_head, n_nextn = val(rm, f"{arch}.block_count"), val(rh, f"{arch}.block_count"), val(rh, f"{arch}.nextn_predict_layers")
    if not n_nextn or n_head != n_main + n_nextn:
        sys.exit(f"HEAD must describe MAIN + its nextn blocks: main block_count {n_main}, head {n_head}, nextn {n_nextn}")
    if val(rm, f"{arch}.nextn_predict_layers"):
        sys.exit("MAIN already has nextn layers")
    tm, th = {t.name: t for t in rm.tensors}, {t.name: t for t in rh.tensors}
    for name in ("token_embd.weight", "output.weight"):
        if name in tm and name in th and list(tm[name].shape) != list(th[name].shape):
            sys.exit(f"{name} shape mismatch: {list(tm[name].shape)} vs {list(th[name].shape)} (different vocabulary or width)")
    prefixes = tuple(f"blk.{i}." for i in range(n_main, n_head))
    graft = [t for t in rh.tensors if t.name.startswith(prefixes)]
    if not graft:
        sys.exit(f"HEAD has no tensors under {prefixes}")
    clash = [t.name for t in graft if t.name in tm]
    if clash:
        sys.exit(f"tensor names already in MAIN: {clash[:5]}")
    skipped = [t.name for t in rh.tensors if not t.name.startswith(prefixes)]

    w = gguf.GGUFWriter(a.out, arch, endianess=rm.endianess)
    written = set()

    def put(field):
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            return
        vt = field.types[0]
        st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(field.name, field.contents(), vt, sub_type=st)
        written.add(field.name)

    for name, f in rm.fields.items():
        if name.startswith(arch + ".") and name in rh.fields:
            if f.contents() != rh.fields[name].contents():
                print(f"  hparam {name}: {str(f.contents())[:60]} -> {str(rh.fields[name].contents())[:60]}")
            put(rh.fields[name])
        else:
            put(f)
    for name, f in rh.fields.items():
        if name.startswith(arch + ".") and name not in written:
            print(f"  hparam {name}: (absent) -> {str(f.contents())[:60]}")
            put(f)

    tensors = list(rm.tensors) + graft
    for t in tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for t in tensors:
        w.write_tensor_data(t.data, tensor_endianess=rm.endianess)
    w.close()
    gb = lambda ts: sum(t.n_bytes for t in ts) / 2**30
    print(f"grafted {len(graft)} tensors ({gb(graft):.3f} GiB) onto {len(rm.tensors)} ({gb(rm.tensors):.3f} GiB); "
          f"left out of HEAD: {skipped}; block_count {n_main} -> {n_head}, nextn {n_nextn}; wrote {a.out}")


if __name__ == "__main__":
    main()
