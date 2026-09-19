import sys, os, subprocess, tempfile, numpy as np
G="~/dev/local-ai/src/llama.cpp-mainline/gguf-py"; sys.path.insert(0,G); import gguf
d=tempfile.mkdtemp(); rng=np.random.default_rng(0)
def mk(path, blocks, n_block, nextn, extra):
    w=gguf.GGUFWriter(path,"qwen35moe"); w.add_uint32("qwen35moe.block_count",n_block); w.add_uint32("qwen35moe.embedding_length",8)
    if nextn: w.add_uint32("qwen35moe.nextn_predict_layers",nextn)
    w.add_string("general.name","m"); w.add_array("tokenizer.ggml.tokens",["a","b","c"]); w.add_uint32("general.file_type", extra)
    data={}
    for n in ["token_embd.weight","output.weight"]+[f"blk.{b}.ffn_up.weight" for b in blocks]:
        data[n]=rng.standard_normal((4,8)).astype(np.float32); w.add_tensor(n,data[n])
    q=rng.integers(0,255,size=(2,18),dtype=np.uint8)  # raw Q4_0 rows: 32 weights -> 18 bytes
    qn=f"blk.{blocks[-1]}.ffn_down.weight"; w.add_tensor(qn,q,raw_shape=(2,18),raw_dtype=gguf.GGMLQuantizationType.Q4_0); data[qn]=q
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close(); return data
dm=mk(f"{d}/main.gguf",[0,1],2,0,29); dh=mk(f"{d}/head.gguf",[2],3,1,2)
r=subprocess.run([sys.executable,"~/dev/local-ai/bench/box/gguf_graft_mtp.py",f"{d}/main.gguf",f"{d}/head.gguf",f"{d}/out.gguf","--gguf-py",G],capture_output=True,text=True)
print(r.stdout, r.stderr[-400:]); assert r.returncode==0
o=gguf.GGUFReader(f"{d}/out.gguf"); names=[t.name for t in o.tensors]
assert o.fields["qwen35moe.block_count"].contents()==3 and o.fields["qwen35moe.nextn_predict_layers"].contents()==1
assert o.fields["general.file_type"].contents()==29 and o.fields["tokenizer.ggml.tokens"].contents()==["a","b","c"]
assert names.count("output.weight")==1 and "blk.2.ffn_up.weight" in names and "blk.2.ffn_down.weight" in names and len(names)==7, names
for t in o.tensors:
    src = dh[t.name] if t.name.startswith("blk.2.") else dm[t.name]
    assert np.array_equal(np.asarray(t.data).reshape(-1).view(np.uint8), src.reshape(-1).view(np.uint8)), t.name
assert [t for t in o.tensors if t.name=="blk.2.ffn_down.weight"][0].tensor_type==gguf.GGMLQuantizationType.Q4_0
r2=subprocess.run([sys.executable,"~/dev/local-ai/bench/box/gguf_graft_mtp.py",f"{d}/out.gguf",f"{d}/head.gguf",f"{d}/o2.gguf","--gguf-py",G],capture_output=True,text=True)
assert r2.returncode!=0; print("refusal ok:", r2.stderr.strip()[-80:] or r2.stdout.strip()[-80:]); print("ALL OK")
