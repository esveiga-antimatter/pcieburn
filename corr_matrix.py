#!/usr/bin/env python3
"""Cross-run correlation matrix for the pcieburn corpus.

Builds one row per run from the artifact bundle (manifest, events.csv,
nvml_trace.csv, aer_delta.txt, pcie_link_states.txt, faults.txt), then tests
the correlations the HYPOTHESES.md ledger cares about. Read-only.
"""
import csv, datetime as dt, os, re, sys
from glob import glob

RUNS = sorted(glob("/home/enjoy/antimatter/gputest/runs/2026*"))

def hms(ts):
    # events.csv: 2026-08-19T21:13:27.400Z
    return dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ")

def nvml_ts(ts):
    # nvml: 2026/08/19 21:23:53.893
    return dt.datetime.strptime(ts.strip(), "%Y/%m/%d %H:%M:%S.%f")

rows = []
for d in RUNS:
    name = os.path.basename(d)
    m = re.match(r"(\d{8}T\d{6})Z-933b21-[0-9a-f]+-cpt(\w+?)-(.+)$", name)
    if not m:
        continue
    r = dict(ts=m.group(1), node=m.group(2), arm=m.group(3), dir=d)

    # ---- manifest ----
    man = open(os.path.join(d, "manifest.txt"), errors="ignore").read()
    g = lambda k: (re.search(rf"^{k}\s*:\s*(.+)$", man, re.M) or [None, ""])[1].strip()
    r["uptime"] = int(g("uptime_seconds") or 0)
    r["boot"] = g("boot_utc")
    r["verdict"] = g("verdict")
    r["args"] = g("pcieburn_args")
    pm = re.search(r"^0, [\d.]+ W, [\d.]+ W, ([\d.]+) W", man, re.M)
    r["cap"] = float(pm.group(1)) if pm else None

    # ---- workload class ----
    a = r["args"]
    r["dim"] = 8192 if "8192" in a else 2048
    r["gpc"] = 1 if "--gemms-per-coll 1" in a else 0
    r["coll4m"] = "coll-min 4M" in a
    r["notensor"] = "--no-tensor" in a
    r["lgc"] = "lgc2100" in name

    # ---- events: TTF, rates ----
    ev = os.path.join(d, "events.csv")
    r["ttf"] = None; r["gemm_s"] = None; r["coll_s"] = None
    if os.path.exists(ev):
        t0 = tkill = None; prog = []
        for row in csv.DictReader(open(ev)):
            if row["event"] == "all_ready": t0 = hms(row["timestamp"])
            elif row["event"] == "killall" and tkill is None: tkill = hms(row["timestamp"])
            elif row["event"] == "progress" and row["rank"] == "0":
                prog.append((hms(row["timestamp"]), int(row["gemms"]), int(row["colls"])))
        if t0 and tkill:
            r["ttf"] = round((tkill - t0).total_seconds(), 1)
        if len(prog) > 2:
            span = (prog[-1][0] - prog[0][0]).total_seconds()
            if span > 0:
                r["gemm_s"] = (prog[-1][1] - prog[0][1]) / span
                r["coll_s"] = (prog[-1][2] - prog[0][2]) / span

    # ---- nvml matched window t=60..93s, width==16 rows only ----
    nv = os.path.join(d, "nvml_trace.csv")
    r["pw"] = r["temp"] = r["clk"] = r["clip"] = None
    r["gpu_pw"] = {}
    if os.path.exists(nv):
        t0 = None; acc = {}; clip = tot = 0
        for line in open(nv, errors="ignore"):
            f = line.split(",")
            if len(f) < 8 or "/" not in f[0]:
                continue
            try:
                t = nvml_ts(f[0]); gpu = int(f[1]); pw = float(f[2])
                clk = float(f[3]); tp = float(f[4]); w = f[7]
            except ValueError:
                continue
            if t0 is None: t0 = t
            e = (t - t0).total_seconds()
            if e < 60 or e > 93: continue
            if "16" not in w: continue
            acc.setdefault(gpu, []).append((pw, tp, clk))
            tot += 1
            if r["cap"] and pw >= r["cap"] * 0.97: clip += 1
        if acc:
            allv = [v for g in acc.values() for v in g]
            r["pw"] = sum(v[0] for v in allv) / len(allv)
            r["temp"] = sum(v[1] for v in allv) / len(allv)
            r["clk"] = sum(v[2] for v in allv) / len(allv)
            r["clip"] = 100.0 * clip / tot if tot else None
            r["gpu_pw"] = {g: sum(v[0] for v in vs)/len(vs) for g, vs in acc.items()}

    # ---- AER nonzero rows ----
    r["aer"] = []
    ad = os.path.join(d, "aer_delta.txt")
    if os.path.exists(ad):
        for line in open(ad, errors="ignore"):
            f = line.split()
            if len(f) >= 11 and f[0] in ("dev", "rootport", "switch"):
                counts = list(map(int, f[3:11]))
                if sum(counts):
                    r["aer"].append((f[0], f[1], f[2], counts))
    r["aer_total"] = sum(sum(c) for *_, c in r["aer"])

    # ---- link states ----
    r["x0"] = []; r["degraded"] = []; r["gen1_dwell"] = {}
    ls = os.path.join(d, "pcie_link_states.txt")
    if os.path.exists(ls):
        for line in open(ls, errors="ignore"):
            f = line.split()
            if len(f) != 4: continue
            gpu, gen, w, smp = f[0], f[1], f[2], int(f[3].split("=")[1])
            if w == "x0": r["x0"].append(gpu)
            elif w != "x16" and w != "x0": r["degraded"].append((gpu, gen, w, smp))
            if gen == "gen1" and w == "x16": r["gen1_dwell"][gpu] = smp

    # ---- fatal signature ----
    ft = os.path.join(d, "faults.txt")
    r["fatal_sig"] = ""
    r["fault_port"] = ""
    if os.path.exists(ft):
        txt = open(ft, errors="ignore").read()
        sigs = []
        if "SDES" in txt: sigs.append("SDES")
        if "ERR_FATAL" in txt: sigs.append("ERR_FATAL")
        if "DPC: containment" in txt: sigs.append("DPC")
        r["fatal_sig"] = "+".join(sigs)
        dm = re.search(r"pcieport (\S+): DPC: containment", txt)
        if dm: r["fault_port"] = dm.group(1)
    r["fault"] = 1 if (r["ttf"] is not None and "LOST" in r["verdict"].upper()
                       or "LOST" in r["verdict"].upper()) else 0
    rows.append(r)

# =========================== MASTER TABLE ===========================
print("=" * 118)
print("MASTER TABLE — one row per run (power/temp/clip = t=60-93 s matched window, x16 rows only)")
print("=" * 118)
hdr = f"{'node':<7}{'arm':<34}{'up(s)':>7}{'cap':>5}{'W':>6}{'T':>5}{'clip%':>6}{'coll/s':>7}{'TTF':>7}{'AER':>8}  outcome"
print(hdr)
for r in rows:
    out = "FAULT " + (r["fatal_sig"] or "?") if r["fault"] else r["verdict"]
    if r["degraded"]: out += " [DEGRADED %s]" % ",".join(g for g, *_ in r["degraded"])
    print(f"{r['node']:<7}{r['arm'][:33]:<34}{r['uptime']:>7}"
          f"{int(r['cap'] or 0):>5}"
          f"{(r['pw'] or 0):>6.0f}{(r['temp'] or 0):>5.0f}"
          f"{(r['clip'] if r['clip'] is not None else -1):>6.1f}"
          f"{(r['coll_s'] or 0):>7.2f}"
          f"{(r['ttf'] if r['ttf'] is not None else 0):>7.1f}"
          f"{r['aer_total']:>8}  {out}")

# =========================== CORRELATIONS ===========================
def pearson(xs, ys):
    n = len(xs)
    if n < 3: return None
    mx, my = sum(xs)/n, sum(ys)/n
    sx = (sum((x-mx)**2 for x in xs))**.5
    sy = (sum((y-my)**2 for y in ys))**.5
    if sx == 0 or sy == 0: return None
    return sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / (sx*sy)

print()
print("=" * 118)
print("C1. FAULT vs MEAN POWER, within workload family (h2: necessary-not-sufficient)")
print("=" * 118)
for fam, sel in (("8192-single, full-burst", lambda r: r["dim"] == 8192 and not r["gpc"]),
                 ("8192-single, gpc1 arms", lambda r: r["dim"] == 8192 and r["gpc"]),
                 ("2048-mixed", lambda r: r["dim"] == 2048)):
    sub = [r for r in rows if sel(r) and r["pw"]]
    fs = sorted((r["pw"], r["node"], r["fault"], r["uptime"]) for r in sub)
    print(f"\n  {fam}:")
    for pw, node, f, up in fs:
        print(f"    {pw:6.0f} W  {node:<7} up={up:>6}  {'FAULT' if f else 'clean'}")
    xs = [r["pw"] for r in sub]; ys = [r["fault"] for r in sub]
    c = pearson(xs, ys)
    if c is not None:
        print(f"    point-biserial r(power, fault) = {c:+.2f}  (n={len(sub)})")

print()
print("=" * 118)
print("C2. TTF vs UPTIME across ALL faults (h4), with boot session marked")
print("=" * 118)
faults = [r for r in rows if r["fault"] and r["ttf"]]
for r in sorted(faults, key=lambda r: r["uptime"]):
    print(f"  {r['node']:<7} up={r['uptime']:>6}s  boot={r['boot'][11:16]}  cap={int(r['cap'] or 0)}"
          f"  pw={(r['pw'] or 0):>4.0f}W  TTF={r['ttf']:>6.1f}s  {r['arm'][:36]}")
xs = [r["uptime"] for r in faults]; ys = [r["ttf"] for r in faults]
c = pearson(xs, ys)
print(f"  r(uptime, TTF) across faults = {c:+.2f}  (n={len(faults)}) — NOTE: workload varies across these")

print()
print("=" * 118)
print("C3. CORRECTABLE AER: which port, per node, per boot (h10: per-boot training)")
print("=" * 118)
for r in rows:
    if not r["aer"]: continue
    dwell = max(r["gen1_dwell"].values()) if r["gen1_dwell"] else 0
    for role, gpu, bdf, c8 in r["aer"]:
        lbl = f"RxErr={c8[0]} BadTLP={c8[1]} DLLP={c8[2]} Roll={c8[3]}"
        print(f"  {r['node']:<7} boot={r['boot'][11:16]} up={r['uptime']:>6} {r['arm'][:30]:<31}"
              f" {role:<9} gpu{gpu} {bdf:<14} {lbl}  (max gen1 dwell this run: {dwell} samp)")

print()
print("=" * 118)
print("C4. VICTIM CONSISTENCY (h6: per-slot term A) — fault port per node across all faults")
print("=" * 118)
from collections import Counter
per = {}
for r in faults:
    per.setdefault(r["node"], []).append((r["fault_port"], r["x0"], r["ttf"]))
for node, lst in per.items():
    print(f"  {node}: " + "; ".join(f"{p} x0={','.join(x)} TTF={t}" for p, x, t in lst))

print()
print("=" * 118)
print("C5. SPATIAL DECOUPLING — in faulting runs, do correctables land on the dying port?")
print("=" * 118)
for r in faults:
    aer_ports = {bdf for _, _, bdf, _ in r["aer"]}
    on = r["fault_port"] in aer_ports if r["fault_port"] else None
    print(f"  {r['node']:<7} {r['arm'][:34]:<35} fault={r['fault_port']:<13}"
          f" correctables on: {sorted(aer_ports) if aer_ports else 'NONE'}"
          f"  -> same port: {on}")

print()
print("=" * 118)
print("C6. CAP-CLIPPING FRACTION vs AER count (h8) — all runs with both measures")
print("=" * 118)
sub = [r for r in rows if r["clip"] is not None]
for r in sorted(sub, key=lambda r: -(r["clip"] or 0)):
    print(f"  {r['node']:<7} {r['arm'][:34]:<35} clip={r['clip']:>5.1f}%  AER={r['aer_total']:>7}"
          f"  {'FAULT' if r['fault'] else 'clean'}")
xs = [r["clip"] for r in sub]; ys = [r["aer_total"] for r in sub]
print(f"  r(clip%, AER total) = {pearson(xs, ys):+.2f}   (n={len(sub)})")
ys2 = [r["fault"] for r in sub]
print(f"  r(clip%, fault)     = {pearson(xs, ys2):+.2f}")

print()
print("=" * 118)
print("C7. VICTIM-GPU POWER RANK — was the dying GPU an outlier beforehand? (known-negative check)")
print("=" * 118)
for r in faults:
    if not r["gpu_pw"] or not r["x0"]: continue
    for victim in r["x0"]:
        vg = int(victim.replace("gpu", ""))
        if vg in r["gpu_pw"]:
            rank = sorted(r["gpu_pw"].values(), reverse=True).index(r["gpu_pw"][vg]) + 1
            print(f"  {r['node']:<7} {r['arm'][:32]:<33} victim gpu{vg}: {r['gpu_pw'][vg]:.0f} W"
                  f" = rank {rank}/{len(r['gpu_pw'])} by power")

print()
print("=" * 118)
print("C8. GEN1 BOOT DWELL vs SUBSEQUENT ERRORS (h10) — per run, max dwell GPU vs AER")
print("=" * 118)
for r in rows:
    if not r["gen1_dwell"]: continue
    mg = max(r["gen1_dwell"], key=r["gen1_dwell"].get)
    mx = r["gen1_dwell"][mg]
    med = sorted(r["gen1_dwell"].values())[len(r["gen1_dwell"])//2]
    if mx > 3 * max(med, 1):
        print(f"  {r['node']:<7} {r['arm'][:34]:<35} {mg} dwelt {mx} samples (median {med})"
              f"  AER this run: {r['aer_total']}")
