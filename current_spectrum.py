#!/usr/bin/env python3
"""Current-modulation spectrum of the pcieburn waveform, from measured timings.

Measured (events.csv decomposition, rgca17 pl575):
  8192-single : t_gemm = 17.404 ms, t_coll+ovh = 112.3 ms, full burst N = 94
  2048-mixed  : t_gemm =  0.121 ms, t_coll+ovh = 249.3 ms, full burst N = 2456
Measured amplitude (NVML p2..p98, matched window):
  8192 arm : 193 W per GPU swing   2048 arm : 144 W per GPU swing
"""
ARMS = {
    "8192-single": dict(tg=0.017404, tc=0.1123, nfull=94,   swing=193),
    "2048-mixed":  dict(tg=0.000121, tc=0.2493, nfull=2456, swing=144),
}
RAIL_V = 12.0
NGPU = 8

print("=" * 74)
print("MODULATION FREQUENCY vs --gemms-per-coll  (edges/s = 2 x colls/s)")
print("=" * 74)
for name, a in ARMS.items():
    print(f"\n{name}:  full-burst N = {a['nfull']}")
    print(f"  {'N':>6} {'pass':>9} {'fundamental':>12} {'edges/s':>9} {'edges/600s run':>15}")
    for N in (1, 4, 16, 64, a["nfull"]):
        if N > a["nfull"]:
            continue
        p = N * a["tg"] + a["tc"]
        print(f"  {N:>6} {p*1000:8.1f}ms {1/p:11.2f} Hz {2/p:9.1f} {int(2/p*600):>15,}")
    lo = 1 / (a["nfull"] * a["tg"] + a["tc"])
    hi = 1 / (1 * a["tg"] + a["tc"])
    print(f"  achievable fundamental range: {lo:.2f} .. {hi:.2f} Hz  ({hi/lo:.0f}x)")

print()
print("=" * 74)
print("AMPLITUDE of the step, per GPU and coherent across 8")
print("=" * 74)
for name, a in ARMS.items():
    dI = a["swing"] / RAIL_V
    print(f"  {name:<12} {a['swing']:>4} W/GPU -> {dI:5.1f} A/GPU on 12 V"
          f" -> {dI*NGPU:6.1f} A coherent ({a['swing']*NGPU/1000:.1f} kW)")

print()
print("=" * 74)
print("EDGE SLEW: unmeasured. Spectral content of a square wave reaches")
print("~1/(pi*t_rise) regardless of how slow the repetition rate is.")
print("=" * 74)
print(f"  {'assumed t_rise':>16} {'spectral knee':>16}   {'vs VRM loop BW 10-100 kHz':>28}")
for tr in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
    knee = 1 / (3.14159 * tr)
    verdict = "inside" if 1e4 <= knee <= 1e5 else ("above" if knee > 1e5 else "below")
    print(f"  {tr*1e6:13.0f} us {knee/1e3:13.1f} kHz   {verdict:>28}")

print()
print("=" * 74)
print("HOLDING FREQUENCY CONSTANT WHILE CHANGING AMPLITUDE")
print("  pick N so that N*t_gemm + t_coll is equal across arms")
print("=" * 74)
for target in (0.25, 0.5, 1.0):
    parts = []
    for name, a in ARMS.items():
        N = (target - a["tc"]) / a["tg"]
        if 1 <= N <= a["nfull"]:
            parts.append(f"{name} N={N:.0f} (swing {a['swing']} W)")
    if parts:
        print(f"  pass = {target*1000:.0f} ms ({1/target:.1f} Hz): " + "  vs  ".join(parts))
