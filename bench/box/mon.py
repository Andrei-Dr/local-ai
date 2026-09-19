"""mon.py start|stop LABEL -- 1 Hz GPU/PCIe/CPU/DRAM telemetry around a test window.
stop prints the summary line and writes /ai/bench/runs/LABEL.mon.json for ledger.py."""
import json, os, re, signal, subprocess, sys, time
cmd, label = sys.argv[1], sys.argv[2]
D = "/ai/bench/mon"; os.makedirs(D, exist_ok=True)
gp, pp, sp, pidf = f"{D}/{label}.gpu", f"{D}/{label}.perf", f"{D}/{label}.stat", f"{D}/{label}.pids"
def cpu():
    v = list(map(int, open("/proc/stat").readline().split()[1:9])); return sum(v), v[3] + v[4]
if cmd == "start":
    a = subprocess.Popen(["nvidia-smi", "dmon", "-s", "put", "-d", "1"], stdout=open(gp, "w"), stderr=subprocess.DEVNULL)
    b = subprocess.Popen(["perf", "stat", "-a", "-x,", "-I", "1000", "-e", "uncore_imc/data_reads/,uncore_imc/data_writes/"],
                         stdout=subprocess.DEVNULL, stderr=open(pp, "w"))
    open(pidf, "w").write(f"{a.pid} {b.pid}"); open(sp, "w").write("%d %d" % cpu())
else:
    for p in open(pidf).read().split():
        try: os.kill(int(p), signal.SIGINT)
        except ProcessLookupError: pass
    time.sleep(0.5)
    t0, i0 = map(int, open(sp).read().split()); t1, i1 = cpu()
    busy = 100.0 * (1 - (i1 - i0) / max(1, t1 - t0)) * os.cpu_count()
    rows = [l.split() for l in open(gp) if l.strip() and not l.startswith("#")]
    f = lambda i: [float(r[i]) for r in rows if len(r) > i and r[i] not in ("-", "N/A")]
    pwr, temp, sm, mem, rx, tx = f(1), f(2), f(4), f(5), f(10), f(11)
    def imc(ev): return [float(l.split(",")[1]) for l in open(pp) if ev in l and l.split(",")[1].replace(".", "").isdigit()]
    rd, wr = imc("data_reads"), imc("data_writes")
    m = lambda x: sum(x) / len(x) if x else 0
    mx = lambda x: max(x, default=0)
    print(f"    telemetry[{label}]: GPU util avg {m(sm):3.0f}% max {mx(sm):3.0f}% | power avg {m(pwr):3.0f}W max {mx(pwr):3.0f}W /100W"
          f" | PCIe->GPU avg {m(rx)/1000:5.2f} max {mx(rx)/1000:5.2f} GB/s (~13 cap) | CPU busy {busy:4.0f}% of 1200% | DRAM read avg {m(rd)/1024:5.1f} max {mx(rd)/1024:5.1f} GB/s (~38 cap)", flush=True)
    r2 = lambda v, d=2: round(v, d)
    os.makedirs("/ai/bench/runs", exist_ok=True)
    json.dump({"seconds": len(rows), "gpu_util_avg": r2(m(sm), 1), "gpu_util_max": mx(sm), "gpu_mem_util_avg": r2(m(mem), 1),
               "power_w_avg": r2(m(pwr), 1), "power_w_max": mx(pwr), "power_cap_w": 100, "gpu_temp_c_max": mx(temp),
               "pcie_rx_gbs_avg": r2(m(rx) / 1000), "pcie_rx_gbs_max": r2(mx(rx) / 1000), "pcie_tx_gbs_avg": r2(m(tx) / 1000), "pcie_tx_gbs_max": r2(mx(tx) / 1000),
               "cpu_busy_pct": r2(busy, 0), "cpu_pct_total": 100 * os.cpu_count(),
               "dram_read_gbs_avg": r2(m(rd) / 1024, 1), "dram_read_gbs_max": r2(mx(rd) / 1024, 1),
               "dram_write_gbs_avg": r2(m(wr) / 1024, 1), "dram_write_gbs_max": r2(mx(wr) / 1024, 1)},
              open(f"/ai/bench/runs/{label}.mon.json", "w"))
