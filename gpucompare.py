#!/usr/bin/env python3
"""Rank the GPUs in a pcieburn run against each other.

Seven healthy peers define what normal is for that node, boot and arm; the
eighth is the finding. Absolute values are not comparable across arms, so
everything here is a within-run comparison. Prints the three signals that
identified a thermally marginal card on cptcor04: median temperature, power
standard deviation (a card pinned at a thermal limit stops tracking the
workload and goes flat) and median SM clock.
"""
import csv, statistics as st, sys, os

path = sys.argv[1] if len(sys.argv) > 1 else "nvml_trace.csv"
if os.path.isdir(path):
    path = os.path.join(path, "nvml_trace.csv")

per = {}
for r in csv.reader(open(path)):
    if len(r) < 8:
        continue
    try:
        g, w, c, t, u = int(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])
    except ValueError:
        continue          # header, or a [GPU is lost] row
    if u < 95:
        continue          # steady load only: ramp and teardown wash the signal out
    per.setdefault(g, {"w": [], "c": [], "t": []})
    per[g]["w"].append(w); per[g]["c"].append(c); per[g]["t"].append(t)

if not per:
    sys.exit(f"no steady-load samples in {path}")

temp = {g: st.median(v["t"]) for g, v in per.items()}
psd  = {g: st.pstdev(v["w"]) for g, v in per.items()}
clk  = {g: st.median(v["c"]) for g, v in per.items()}
rank = sorted(temp, key=lambda g: -temp[g])

print(f"{len(next(iter(per.values()))['w'])} steady-load samples per GPU\n")
print(f"{'':6}{'temp':>7}{'rank':>6}{'gap':>6}{'pwr sd':>9}{'clk':>7}   flags")
for g in sorted(per):
    peers_t = [temp[x] for x in temp if x != g]
    peers_s = [psd[x]  for x in psd  if x != g]
    peers_c = [clk[x]  for x in clk  if x != g]
    # HOT needs the GPU to stand APART, not merely to top a gradient. Chassis
    # airflow produces a smooth 5-10C spread with no step in it (a sidewall
    # position runs hot for a reason that is not a defect), and a plain
    # "N degrees over the peer median" test flags the top of any such gradient.
    # A failing card leaves a visible gap: 19C to second place on cptcor04
    # against 2C for a gradient on cptcor12.
    others = sorted((temp[x] for x in temp if x != g), reverse=True)
    gap    = temp[g] - others[0]
    spread = others[0] - others[-1]
    f = []
    if temp[g] - st.median(peers_t) >= 5 and gap >= max(3, spread / 2):
        f.append("HOT")
    if st.median(peers_s) / max(psd[g], .01) >= 3: f.append("FLAT-POWER")
    if st.median(peers_c) - clk[g] >= 200:     f.append("DOWNCLOCKED")
    print(f"  gpu{g} {temp[g]:6.0f}C{rank.index(g)+1:>5}{gap:+6.0f}{psd[g]:9.1f}{clk[g]:7.0f}   "
          f"{'  '.join(f) or '-'}")

def _hot(g):
    o = sorted((temp[x] for x in temp if x != g), reverse=True)
    return (temp[g] - st.median(o) >= 5) and (temp[g] - o[0] >= max(3, (o[0] - o[-1]) / 2))
sus = [g for g in sorted(per)
       if _hot(g)
       or (st.median([psd[x] for x in psd if x != g]) / max(psd[g], .01) >= 3)]
print()
print(f"SUSPECT: gpu{','.join(map(str,sus))}" if sus
      else "No GPU deviates from its peers on these signals.")
