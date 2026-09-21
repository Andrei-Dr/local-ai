"""mon.py start|stop LABEL -- 1 Hz GPU/PCIe/CPU/DRAM telemetry around a test window.
stop prints the summary line and writes /ai/bench/runs/LABEL.mon.json for ledger.py.
Also samples the CPU used by processes that are not the benchmark (a CI runner, a build) and flags the row."""
import json, os, re, signal, subprocess, sys, time
D = "/ai/bench/mon"
OURS = re.compile(r"^(llama-|nvidia-smi$|perf$)")
FOREIGN_FLAG_PCT = 50.0   # average over the window, in % of one core
def is_ours(comm): return bool(OURS.match(comm))
def proc_ticks():
    """pid -> (comm, utime+stime, starttime) for every process but this one."""
    out = {}
    for p in os.listdir("/proc"):
        if not p.isdigit() or int(p) == os.getpid(): continue
        try: s = open(f"/proc/{p}/stat").read()
        except OSError: continue
        comm, rest = s[s.index("(") + 1:s.rindex(")")], s[s.rindex(")") + 2:].split()
        out[int(p)] = (comm, int(rest[11]) + int(rest[12]), int(rest[19]))
    return out
def foreign_delta(a, b):
    """ticks burned between two proc_ticks() snapshots by processes that are not ours, per comm."""
    d = {}
    for pid, (comm, t, born) in b.items():
        if is_ours(comm): continue
        prev = a.get(pid)   # same pid + same start time = same process (kernel workers rename themselves per work item)
        used = t - prev[1] if prev and prev[2] == born else t
        if used > 0: d[comm] = d.get(comm, 0) + used
    return d
def foreign_summary(samples, hz):
    """samples: one foreign_delta() per second."""
    pct = [100.0 * sum(s.values()) / hz for s in samples]
    tot = {}
    for s in samples:
        for k, v in s.items(): tot[k] = tot.get(k, 0) + v
    avg = round(sum(pct) / len(pct), 1) if pct else 0.0
    return {"foreign_cpu_pct_avg": avg, "foreign_cpu_pct_max": round(max(pct, default=0.0), 1),
            "foreign_cpu_top": max(tot, key=tot.get) if tot else None, "foreign_cpu_flag": avg > FOREIGN_FLAG_PCT}
def cpu():
    v = list(map(int, open("/proc/stat").readline().split()[1:9])); return sum(v), v[3] + v[4]
if __name__ != "__main__": cmd = label = None
else:
    cmd, label = sys.argv[1], sys.argv[2]; os.makedirs(D, exist_ok=True)
    gp, pp, sp, pidf, fp = f"{D}/{label}.gpu", f"{D}/{label}.perf", f"{D}/{label}.stat", f"{D}/{label}.pids", f"{D}/{label}.foreign"
if cmd == "sample":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    prev = proc_ticks()
    with open(fp, "w") as f:
        while True:
            time.sleep(1); cur = proc_ticks()
            f.write(json.dumps(foreign_delta(prev, cur)) + "\n"); f.flush(); prev = cur
elif cmd == "start":
    a = subprocess.Popen(["nvidia-smi", "dmon", "-s", "put", "-d", "1"], stdout=open(gp, "w"), stderr=subprocess.DEVNULL)
    b = subprocess.Popen(["perf", "stat", "-a", "-x,", "-I", "1000", "-e", "uncore_imc/data_reads/,uncore_imc/data_writes/"],
                         stdout=subprocess.DEVNULL, stderr=open(pp, "w"))
    c = subprocess.Popen(["sh", "-c", "while :; do awk '/MemAvailable/{a=$2}/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int(a/1024), int((t-f)/1024)}' /proc/meminfo; sleep 1; done"],
                         stdout=open(f"{D}/{label}.mem", "w"), stderr=subprocess.DEVNULL)
    d = subprocess.Popen([sys.executable, os.path.abspath(__file__), "sample", label], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    open(pidf, "w").write(f"{a.pid} {b.pid} {d.pid} {c.pid}"); open(sp, "w").write("%d %d" % cpu())
elif cmd == "stop":
    for p in open(pidf).read().split():
        try: os.kill(int(p), signal.SIGTERM if p == open(pidf).read().split()[-1] else signal.SIGINT)
        except ProcessLookupError: pass
    time.sleep(0.5)
    t0, i0 = map(int, open(sp).read().split()); t1, i1 = cpu()
    busy = 100.0 * (1 - (i1 - i0) / max(1, t1 - t0)) * os.cpu_count()
    rows = [l.split() for l in open(gp) if l.strip() and not l.startswith("#")]
    f = lambda i: [float(r[i]) for r in rows if len(r) > i and r[i] not in ("-", "N/A")]
    pwr, temp, sm, mem, rx, tx = f(1), f(2), f(4), f(5), f(10), f(11)
    def imc(ev): return [float(l.split(",")[1]) for l in open(pp) if ev in l and l.split(",")[1].replace(".", "").isdigit()]
    rd, wr = imc("data_reads"), imc("data_writes")
    memrows = [list(map(int, l.split())) for l in open(f"{D}/{label}.mem") if len(l.split()) == 2] if os.path.exists(f"{D}/{label}.mem") else []
    m = lambda x: sum(x) / len(x) if x else 0
    mx = lambda x: max(x, default=0)
    fs = foreign_summary([json.loads(l) for l in open(fp) if l.strip()] if os.path.exists(fp) else [], os.sysconf("SC_CLK_TCK"))
    if fs["foreign_cpu_flag"]:
        print(f"    FOREIGN CPU [{label}]: avg {fs['foreign_cpu_pct_avg']:.0f}% max {fs['foreign_cpu_pct_max']:.0f}% of one core, top '{fs['foreign_cpu_top']}' -- speed numbers in this window are contaminated", flush=True)
    print(f"    telemetry[{label}]: GPU util avg {m(sm):3.0f}% max {mx(sm):3.0f}% | power avg {m(pwr):3.0f}W max {mx(pwr):3.0f}W /100W"
          f" | PCIe->GPU avg {m(rx)/1000:5.2f} max {mx(rx)/1000:5.2f} GB/s (~13 cap) | CPU busy {busy:4.0f}% of 1200% | DRAM read avg {m(rd)/1024:5.1f} max {mx(rd)/1024:5.1f} GB/s (~38 cap)", flush=True)
    r2 = lambda v, d=2: round(v, d)
    os.makedirs("/ai/bench/runs", exist_ok=True)
    json.dump({"seconds": len(rows), "gpu_util_avg": r2(m(sm), 1), "gpu_util_max": mx(sm), "gpu_mem_util_avg": r2(m(mem), 1),
               "power_w_avg": r2(m(pwr), 1), "power_w_max": mx(pwr), "power_cap_w": 100, "gpu_temp_c_max": mx(temp),
               "pcie_rx_gbs_avg": r2(m(rx) / 1000), "pcie_rx_gbs_max": r2(mx(rx) / 1000), "pcie_tx_gbs_avg": r2(m(tx) / 1000), "pcie_tx_gbs_max": r2(mx(tx) / 1000),
               "cpu_busy_pct": r2(busy, 0), "cpu_pct_total": 100 * os.cpu_count(), **fs,
               "dram_read_gbs_avg": r2(m(rd) / 1024, 1), "dram_read_gbs_max": r2(mx(rd) / 1024, 1),
               "dram_write_gbs_avg": r2(m(wr) / 1024, 1), "dram_write_gbs_max": r2(mx(wr) / 1024, 1),
               "mem_avail_mib_min": min((r[0] for r in memrows), default=None), "swap_used_mib_max": max((r[1] for r in memrows), default=None)},
              open(f"/ai/bench/runs/{label}.mon.json", "w"))
