#!/usr/bin/env python3
"""Decompose a pass into GEMM time and collective time, then model sync rate
as a function of --gemms-per-coll."""
import sys, csv, datetime as dt

path = sys.argv[1]
dim = int(sys.argv[2])
label = sys.argv[3]

rows = []
with open(path) as f:
    for r in csv.DictReader(f):
        if r["event"] != "progress" or int(r["rank"]) != 0:
            continue
        t = dt.datetime.strptime(r["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ")
        rows.append((t, int(r["gemms"]), int(r["colls"]), float(r["gflops"])))

flop_per_gemm = 2.0 * dim ** 3
pass_s, gemm_s, n = [], [], 0
for (t0, g0, c0, _), (t1, g1, c1, gf1) in zip(rows, rows[1:]):
    dtsec = (t1 - t0).total_seconds()
    dg, dc = g1 - g0, c1 - c0
    if dc != 1 or dg <= 0 or dtsec <= 0 or gf1 <= 0:
        continue
    pass_s.append(dtsec)
    gemm_s.append(dg * flop_per_gemm / (gf1 * 1e9))   # gflops is GEMM-window based
    n += 1

if not n:
    print(f"{label}: no usable passes"); sys.exit()

mp = sum(pass_s) / n
mg = sum(gemm_s) / n
gpc = (rows[-1][1] - rows[0][1]) / (rows[-1][2] - rows[0][2])

print(f"--- {label}  (matrix-dim {dim}, {n} passes)")
print(f"    GEMMs per collective (measured)   : {gpc:.0f}")
print(f"    mean pass wall time               : {mp*1000:8.1f} ms")
print(f"    mean GEMM-window time             : {mg*1000:8.1f} ms")
print(f"    implied collective + overhead     : {(mp-mg)*1000:8.1f} ms")
print(f"    per-GEMM time                     : {mg/gpc*1000:8.3f} ms")
print()
t_gemm = mg / gpc
t_coll = mp - mg
print(f"    sync rate model: colls/s = 1 / (N x {t_gemm*1000:.3f} ms + {t_coll*1000:.1f} ms)")
print(f"    {'N (--gemms-per-coll)':>22} {'pass':>9} {'colls/s':>9}   {'vs Megatron 2-3':>16}")
for N in (1, 2, 3, 4, 8, 16, 32, 94, 256, 1024, 2453):
    p = N * t_gemm + t_coll
    print(f"    {N:>22} {p*1000:8.1f}ms {1/p:9.1f}")
print(f"    ceiling as N->0 (collective-latency bound): {1/t_coll:.1f} colls/s")
