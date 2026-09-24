# T0 worker (torchrun): one training regime of the Qwen3.6-35B-A3B-shaped student, random BF16 init from config.
# Pre-registration lives in t0.sh (the driver). This file only measures and reports.
import argparse, json, os, re, statistics, sys, threading, time
import torch
import torch.distributed as dist
import transformers
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM
from transformers.integrations import moe as tmoe

ap = argparse.ArgumentParser()
ap.add_argument("--regime", choices=("frozen", "full"), required=True)
ap.add_argument("--config", default="/w/t0/qwen36_src_config.json")
ap.add_argument("--layers", type=int, default=0, help="0 = all layers of the config (smoke runs use fewer)")
ap.add_argument("--seq", type=int, default=2048)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--experts", default="grouped_mm")
ap.add_argument("--opt", default="auto", help="auto = AdamW (frozen) / Adafactor (full)")
ap.add_argument("--train-block", type=int, default=-1, help="frozen regime: the one trainable decoder block (-1 = last)")
ap.add_argument("--mem-guard-gib", type=float, default=24.0, help="abort if host MemAvailable drops below this")
ap.add_argument("--profile", action="store_true", help="one extra profiled step after the timed steps")
ap.add_argument("--label", required=True)
ap.add_argument("--out", default="/w/t0/results")
a = ap.parse_args()

dist.init_process_group(backend="cuda:nccl,cpu:gloo")
rank, ws = dist.get_rank(), dist.get_world_size()
dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(dev)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
log = (lambda *x: print(f"[{a.label} {time.strftime('%T')}]", *x, flush=True)) if rank == 0 else (lambda *x: None)


def meminfo(key):
    for line in open("/proc/meminfo"):
        if line.startswith(key + ":"):
            return int(line.split()[1]) / 2**20  # GiB
    return float("nan")


# Host-RAM guard: the box also serves Dave's workloads; never let this job push the host into OOM.
mem = {"avail_start_gib": meminfo("MemAvailable"), "avail_min_gib": meminfo("MemAvailable")}


def guard():
    while True:
        m = meminfo("MemAvailable")
        mem["avail_min_gib"] = min(mem["avail_min_gib"], m)
        if m < a.mem_guard_gib:
            print(f"[{a.label}] MEM GUARD: MemAvailable {m:.1f} GiB < {a.mem_guard_gib} GiB, aborting", flush=True)
            os._exit(3)
        time.sleep(0.5)


threading.Thread(target=guard, daemon=True).start()

# ---- config -> model on meta ----
raw = json.load(open(a.config))["text_config"]
if a.layers:
    raw["num_hidden_layers"] = a.layers
    raw["layer_types"] = raw["layer_types"][: a.layers]
mtype = raw.pop("model_type")
cfg = transformers.CONFIG_MAPPING[mtype](**raw)


def build(c, device):
    with torch.device(device):
        return AutoModelForCausalLM.from_config(c, dtype=torch.bfloat16, experts_implementation=a.experts,
                                                attn_implementation="sdpa")


model = build(cfg, "meta")
L = cfg.num_hidden_layers
names = [n for n, _ in model.named_parameters()]
n_total = sum(p.numel() for p in model.parameters())

# ---- trainable set ----
tb = a.train_block if a.train_block >= 0 else L - 1
router_re = re.compile(r"\.mlp\.gate\.weight$")
for n, p in model.named_parameters():
    if a.regime == "full":
        p.requires_grad_(True)
    else:
        p.requires_grad_(bool(router_re.search(n)) or n.startswith(f"model.layers.{tb}."))
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_router = sum(p.numel() for n, p in model.named_parameters() if router_re.search(n))
log(f"{type(model).__name__} layers={L} params={n_total/1e9:.3f}B trainable={n_train/1e9:.4f}B "
    f"(routers {n_router/1e6:.1f}M, block {tb if a.regime == 'frozen' else '-'}), experts_impl={cfg._experts_implementation}")

# ---- FSDP2 ----
offload = CPUOffloadPolicy(pin_memory=True) if a.regime == "full" else None
kw = {"offload_policy": offload} if offload else {}
for layer in model.model.layers:
    fully_shard(layer, **kw)
fully_shard(model, **kw)

# ---- materialize + random init matched to the model's own init (per-tensor mean/std from a tiny reference) ----
t0 = time.time()
model.to_empty(device="cpu" if offload else dev)
ref_raw = dict(raw)
ref_types = ["linear_attention", "full_attention"]
ref_raw.update(num_hidden_layers=2, layer_types=ref_types, num_experts=4, num_experts_per_tok=2, vocab_size=512,
               pad_token_id=0, bos_token_id=0, eos_token_id=0)
torch.manual_seed(0)
ref = build(transformers.CONFIG_MAPPING[mtype](**ref_raw), "cpu")
ref_p = dict(ref.named_parameters())
ref_b = dict(ref.named_buffers())


def ref_name(n):
    m = re.match(r"model\.layers\.(\d+)\.(.*)", n)
    if not m:
        return n
    return f"model.layers.{ref_types.index(cfg.layer_types[int(m.group(1))])}.{m.group(2)}"


torch.manual_seed(1234 + rank)
with torch.no_grad():
    for n, p in model.named_parameters():
        r = ref_p[ref_name(n)].float()
        t = p.to_local() if isinstance(p, DTensor) else p
        mu, sd = r.mean().item(), (r.std().item() if r.numel() > 1 else 0.0)
        if sd > 0:  # generate on the GPU (CPU normal_ on 70 GB of offloaded shards takes minutes)
            t.copy_(torch.empty(t.shape, dtype=t.dtype, device=dev).normal_(mu, sd))
        else:
            t.fill_(mu)
    for mod_name, mod in model.named_modules():
        for bn, b in list(mod._buffers.items()):
            if b is None:
                continue
            full = f"{mod_name}.{bn}" if mod_name else bn
            src = ref_b[ref_name(full)]
            assert src.shape == b.shape, (full, src.shape, b.shape)
            mod._buffers[bn] = src.to(device=dev, dtype=src.dtype)  # buffers compute on GPU even with param offload
del ref, ref_p, ref_b
log(f"materialized + init in {time.time()-t0:.1f}s; host MemAvailable {meminfo('MemAvailable'):.1f} GiB")

model.train()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
params = [p for p in model.parameters() if p.requires_grad]
opt_name = a.opt if a.opt != "auto" else ("AdamW" if a.regime == "frozen" else "Adafactor-local")
# Adafactor-local: torch Adafactor over each rank's local shards (aliases of the DTensor storage). torch Adafactor on
# DTensor fails (aten.pow_ on a _NormPartial placement); per-shard factored statistics cost the same.
locs = [(p.to_local() if isinstance(p, DTensor) else p).detach() for p in params] if opt_name == "Adafactor-local" else None
opt = {"AdamW": lambda: torch.optim.AdamW(params, lr=1e-5), "Adafactor": lambda: torch.optim.Adafactor(params, lr=1e-5),
       "Adafactor-local": lambda: torch.optim.Adafactor(locs, lr=1e-5),
       "SGD": lambda: torch.optim.SGD(params, lr=1e-5)}[opt_name]()


def opt_step():
    if locs is not None:
        for l, p in zip(locs, params):
            l.grad = None if p.grad is None else (p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad).detach()
    opt.step(); opt.zero_grad(set_to_none=True)
    if locs is not None:
        for l in locs:
            l.grad = None

# ---- kernel-path record: static dispatch + runtime counters ----
mm = sys.modules[next(c for c in type(model).__mro__ if c.__module__.startswith("transformers.models.")).__module__]


def impls(f):
    out, stack, seen = set(), [f], set()
    while stack:
        g = stack.pop()
        if id(g) in seen:
            continue
        seen.add(id(g))
        if callable(g) and getattr(g, "__module__", None):
            out.add(f"{g.__module__}.{getattr(g, '__qualname__', '?')}")
        for c in getattr(g, "__closure__", None) or ():
            try:
                v = c.cell_contents
            except ValueError:
                continue
            if callable(v):
                stack.append(v)
        if getattr(g, "__wrapped__", None):
            stack.append(g.__wrapped__)
    return sorted(out)


paths = {k: impls(getattr(mm, k)) for k in ("torch_chunk_gated_delta_rule", "causal_conv1d_fn")}
gmm = {"grouped_mm": 0, "fallback": 0}
_can = tmoe._can_use_grouped_mm


def _can_counted(*x, **k):
    ok = _can(*x, **k)
    gmm["grouped_mm" if ok else "fallback"] += 1
    return ok


tmoe._can_use_grouped_mm = _can_counted

# ---- steps ----
V = cfg.vocab_size
g = torch.Generator(device="cpu").manual_seed(7 + rank)


def step():
    ids = torch.randint(0, V, (1, a.seq), generator=g).to(dev)
    torch.cuda.synchronize(); dist.barrier(); t = [time.perf_counter()]
    loss = model(input_ids=ids, labels=ids).loss
    torch.cuda.synchronize(); t.append(time.perf_counter())
    loss.backward()
    torch.cuda.synchronize(); t.append(time.perf_counter())
    # grad-norm check (untimed): T0a requires finite gradients on the trainable set
    sq = [(p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad).float().pow(2).sum() for p in params
          if p.grad is not None]
    gn = torch.stack([x.to(dev) for x in sq]).sum().double() if sq else torch.zeros((), device=dev, dtype=torch.float64)
    dist.all_reduce(gn)
    torch.cuda.synchronize(); t.append(time.perf_counter())
    opt_step()
    torch.cuda.synchronize(); dist.barrier(); t.append(time.perf_counter())
    fwd, bwd, o = t[1] - t[0], t[2] - t[1], t[4] - t[3]
    return {"loss": loss.detach().float().item(), "grad_norm": gn.sqrt().item(), "fwd": fwd, "bwd": bwd,
            "opt": o, "step": fwd + bwd + o}


rows, err, prof_top = [], None, None
try:
    for i in range(a.warmup + a.steps):
        r = step()
        r["warmup"] = i < a.warmup
        rows.append(r)
        log(f"step {i} {'warm' if r['warmup'] else 'TIME'} loss={r['loss']:.4f} gnorm={r['grad_norm']:.3e} "
            f"fwd={r['fwd']:.2f}s bwd={r['bwd']:.2f}s opt={r['opt']:.2f}s step={r['step']:.2f}s "
            f"tok/s={ws*a.seq/r['step']:.1f} peakVRAM={torch.cuda.max_memory_reserved(dev)/2**30:.1f}GiB")
    if a.profile:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            step()
        ka = sorted((e for e in prof.key_averages() if e.self_device_time_total > 0),
                    key=lambda e: -e.self_device_time_total)
        tot = sum(e.self_device_time_total for e in ka)
        prof_top = [{"name": e.key[:160], "device_ms": e.self_device_time_total / 1e3, "calls": e.count,
                     "share": e.self_device_time_total / tot} for e in ka[:40]]
        if rank == 0:
            prof.export_chrome_trace(f"{a.out}/{a.label}.trace.json")
except Exception as e:  # T0a: report the failing op precisely
    import traceback
    err = traceback.format_exc()
    print(f"[{a.label} rank{rank}] FAILED:\n{err}", flush=True)

# ---- report ----
hwm = next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmHWM")) / 2**20
mine = {"rank": rank, "vram_peak_reserved_gib": torch.cuda.max_memory_reserved(dev) / 2**30,
        "vram_peak_alloc_gib": torch.cuda.max_memory_allocated(dev) / 2**30, "host_rss_hwm_gib": hwm,
        "gpu": torch.cuda.get_device_properties(dev).gcnArchName, "pci_bus": torch.cuda.get_device_properties(dev).pci_bus_id,
        "failed": err is not None}
allr = [None] * ws
dist.all_gather_object(allr, mine)
if rank == 0:
    timed = [r for r in rows if not r["warmup"]]
    med = {k: statistics.median(r[k] for r in timed) for k in ("fwd", "bwd", "opt", "step")} if timed else {}
    res = {"label": a.label, "regime": a.regime, "world_size": ws, "seq": a.seq, "micro_batch_per_rank": 1,
           "tokens_per_step": ws * a.seq, "layers": L, "params_total": n_total, "params_trainable": n_train,
           "train_block": tb if a.regime == "frozen" else None, "optimizer": opt_name,
           "offload": bool(offload), "grad_ckpt": True, "experts_impl": cfg._experts_implementation,
           "attn_impl": cfg._attn_implementation, "gdn_paths": paths, "grouped_mm_calls": gmm,
           "torch": torch.__version__, "transformers": transformers.__version__, "hip": torch.version.hip,
           "steps": rows, "median": med, "tok_s_median": (ws * a.seq / med["step"]) if med else None,
           "step_spread_s": (max(r["step"] for r in timed) - min(r["step"] for r in timed)) if timed else None,
           "ranks": allr, "host_mem": mem | {"avail_end_gib": meminfo("MemAvailable")},
           "profile_top": prof_top, "error": err}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/{a.label}.json", "w"), indent=1)
    log("RESULT", json.dumps({k: res[k] for k in ("tok_s_median", "median", "gdn_paths", "grouped_mm_calls",
                                                  "optimizer", "error")})[:3000])
    log("RANKS", json.dumps(allr))
    if prof_top:
        for e in prof_top[:25]:
            log(f"  {e['share']*100:5.1f}% {e['device_ms']:9.1f}ms x{e['calls']:<6} {e['name']}")
dist.destroy_process_group()
sys.exit(1 if err else 0)
