#!/usr/bin/env python3
"""gguf_splice.py BASE DONOR OUT REGEX

Write OUT = BASE with every tensor whose name matches REGEX taken from DONOR (its type and bytes); all other tensors and all
metadata are copied byte-for-byte from BASE. Streams tensor data (memmap), so RAM use stays small.
"""
import re
import sys

import gguf


def main() -> None:
    base_path, donor_path, out_path, pattern = sys.argv[1:5]
    rx = re.compile(pattern)
    base = gguf.GGUFReader(base_path)
    donor = {t.name: t for t in gguf.GGUFReader(donor_path).tensors}
    arch = base.fields[gguf.Keys.General.ARCHITECTURE].contents()
    w = gguf.GGUFWriter(out_path, arch=arch, endianess=base.endianess)
    for f in base.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        vt = f.types[0]
        st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(f.name, f.contents(), vt, sub_type=st)
    picked = []
    for t in base.tensors:
        src = t
        if rx.search(t.name):
            d = donor.get(t.name)
            if d is None:
                sys.exit(f"SPLICE_FAILED: {t.name} missing in donor")
            if list(d.shape) != list(t.shape):
                sys.exit(f"SPLICE_FAILED: {t.name} shape {list(d.shape)} != {list(t.shape)}")
            src = d
            picked.append(f"{t.name} {t.tensor_type.name}->{d.tensor_type.name}")
        picked_t = src
        w.add_tensor_info(t.name, picked_t.data.shape, picked_t.data.dtype, picked_t.data.nbytes, picked_t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in base.tensors:
        src = donor[t.name] if rx.search(t.name) else t
        w.write_tensor_data(src.data, tensor_endianess=base.endianess)
    w.close()
    print(f"spliced {len(picked)} tensors from donor")
    for p in picked[:6]:
        print("  " + p)


if __name__ == "__main__":
    main()
