# Usage: ~/dev/.venv/bin/python scan.py <https://huggingface.co/<repo>/resolve/main/<file>.gguf> [more URLs...]
# Range-fetches only the GGUF header (no full download) and sums tensor offsets into expert (*_exps),
# token_embd, and non-expert bytes; prints quant types and the 38 GB/s DDR4 decode ceiling. Stdlib only.
import struct, sys, urllib.request, re, json

def fetch(url, n):
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{n-1}", "User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=120) as r:
        total = int(r.headers["Content-Range"].split("/")[1])
        return r.read(), total

class Short(Exception): pass

class R:
    def __init__(s, b): s.b = b; s.o = 0
    def take(s, n):
        if s.o + n > len(s.b): raise Short()
        v = s.b[s.o:s.o+n]; s.o += n; return v
    def u32(s): return struct.unpack("<I", s.take(4))[0]
    def u64(s): return struct.unpack("<Q", s.take(8))[0]
    def str(s): return s.take(s.u64()).decode("utf-8", "replace")
    def val(s, t):
        fm = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<?",10:"<Q",11:"<q",12:"<d"}
        if t in fm:
            return struct.unpack(fm[t], s.take(struct.calcsize(fm[t])))[0]
        if t == 8: return s.str()
        if t == 9:
            et = s.u32(); n = s.u64()
            if et in fm:
                sz = struct.calcsize(fm[et]); raw = s.take(sz*n)
                return [struct.unpack_from(fm[et], raw, i*sz)[0] for i in range(min(n, 64))] if n <= 64 else f"<arr {n}>"
            out = None
            for i in range(n): s.val(et)
            return f"<arr {n}>"
        raise ValueError(t)

def scan(url):
    n = 16 << 20
    while True:
        buf, total = fetch(url, n)
        try:
            r = R(buf)
            assert r.take(4) == b"GGUF"
            ver = r.u32(); nt = r.u64(); nkv = r.u64()
            kv = {}
            for _ in range(nkv):
                k = r.str(); t = r.u32(); kv[k] = r.val(t)
            ts = []
            for _ in range(nt):
                name = r.str(); nd = r.u32(); dims = [r.u64() for _ in range(nd)]; ty = r.u32(); off = r.u64()
                ts.append((name, dims, ty, off))
            hdr_end = r.o
            break
        except Short:
            n *= 2
            if n > (512 << 20): raise
    align = kv.get("general.alignment", 32)
    data_start = (hdr_end + align - 1) // align * align
    ts.sort(key=lambda x: x[3])
    sizes = {}
    for i, (name, dims, ty, off) in enumerate(ts):
        nxt = ts[i+1][3] if i+1 < len(ts) else total - data_start
        sizes[name] = (nxt - off, dims, ty)
    return kv, sizes, total

TYPES = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",16:"IQ2_XXS",17:"IQ2_XS",18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",21:"IQ3_S",22:"IQ2_S",23:"IQ4_XS",29:"IQ1_M",30:"BF16",34:"TQ1_0",35:"TQ2_0",39:"MXFP4"}

for url in sys.argv[1:]:
    try:
        kv, sizes, total = scan(url)
    except Exception as e:
        print("FAIL", url, repr(e)); continue
    arch = kv.get("general.architecture")
    g = lambda k: kv.get(f"{arch}.{k}")
    exps = sum(s for n,(s,d,t) in sizes.items() if "_exps" in n)
    emb = sum(s for n,(s,d,t) in sizes.items() if n.startswith("token_embd") or n.startswith("per_layer_token_embd"))
    outw = sum(s for n,(s,d,t) in sizes.items() if n == "output.weight")
    nextn = sum(s for n,(s,d,t) in sizes.items() if "nextn" in n or n.startswith("mtp"))
    other = total - exps - emb
    ec, eu = g("expert_count"), g("expert_used_count")
    exp_types = {}
    for n,(s,d,t) in sizes.items():
        if "_exps" in n:
            key = re.sub(r"blk\.\d+\.", "", n) + ":" + TYPES.get(t, str(t))
            exp_types[key] = exp_types.get(key, 0) + 1
    oth_types = {}
    for n,(s,d,t) in sizes.items():
        if "_exps" not in n:
            k = TYPES.get(t, str(t)); oth_types[k] = oth_types.get(k, 0) + s
    print("====", url.split("/resolve/main/")[0].replace("https://huggingface.co/",""), "|", url.split("/")[-1])
    print(f" arch={arch} file_type={kv.get('general.file_type')} blocks={g('block_count')} n_embd={g('embedding_length')} experts={ec} used={eu} shexp_ff={g('expert_shared_feed_forward_length')} exp_ff={g('expert_feed_forward_length')} nextn={g('nextn_predict_layers')}")
    print(f" heads={g('attention.head_count')} kv_heads={g('attention.head_count_kv')} key_len={g('attention.key_length')} val_len={g('attention.value_length')} swa={g('attention.sliding_window')} key_len_swa={g('attention.key_length_swa')} full_attn_interval={g('full_attention_interval')}")
    print(f" total={total/1e9:.2f}GB  experts={exps/1e9:.2f}GB  token_embd={emb/1e9:.2f}GB  output.weight={outw/1e9:.2f}GB  non-expert-excl-embd={other/1e9:.2f}GB  nextn/mtp tensors={nextn/1e9:.2f}GB")
    if ec and eu:
        act = exps * eu / ec
        print(f" active expert bytes/token ~= {act/1e9:.3f}GB -> ceiling {38/(act/1e9):.1f} tok/s (38GB/s)")
    print(" expert tensor types:", exp_types)
    print(" non-expert bytes by type (GB):", {k: round(v/1e9,2) for k,v in oth_types.items()})
