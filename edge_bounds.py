#!/usr/bin/env python3
"""What can and cannot be calculated about the current edge.

Measured inputs (NVML envelope, matched window, 8192-single arm):
  dP = 193 W per GPU   (LOWER bound: NVML averages over ~300-400 ms)
  12 V rail, 8 GPUs coherent
Assumed inputs are flagged ASSUMED and are the reason this is a bound, not a value.
"""
V = 12.0
dP = 193.0
N = 8
dI = dP / V
dI_coh = dI * N

print("=" * 74)
print("MEASURED / DERIVED WITHOUT ASSUMPTIONS")
print("=" * 74)
print(f"  step per GPU          {dP:6.0f} W  -> {dI:6.2f} A on 12 V")
print(f"  coherent across {N}     {dP*N:6.0f} W  -> {dI_coh:6.2f} A")

print()
print("=" * 74)
print("THRESHOLD: how fast must the edge be to excite a VRM impedance peak?")
print("  spectral knee of a square edge = 1/(pi*t_rise)")
print("=" * 74)
for fbw in (10e3, 30e3, 100e3):
    t = 1 / (3.14159 * fbw)
    print(f"  loop BW {fbw/1e3:5.0f} kHz -> edges faster than {t*1e6:6.1f} us excite it")
print("  Kernel-to-kernel handover on one stream and SM drain are both")
print("  single-digit microseconds, so the LOAD-side edge is almost certainly")
print("  inside this window. Whether the 12 V rail sees it is the open part.")

print()
print("=" * 74)
print("CAN THE RAIL HOLD FLAT? C required = dI * t / dV")
print("=" * 74)
print(f"  {'t_rise':>8} {'dV=50mV':>12} {'dV=100mV':>12} {'dV=200mV':>12}   (per GPU)")
for t in (1e-6, 1e-5, 1e-4, 1e-3):
    row = "  " + f"{t*1e6:6.0f} us"
    for dV in (0.05, 0.10, 0.20):
        C = dI * t / dV
        row += f" {C*1e6:9.0f} uF"
    print(row)
print()
print("  ASSUMED card input bulk capacitance ~1 mF -> resulting sag:")
for t in (1e-6, 1e-5, 1e-4, 1e-3):
    dV = dI * t / 1e-3
    print(f"    t_rise {t*1e6:6.0f} us -> {dV*1e3:8.1f} mV on the card's 12 V input")
print("  ASSUMED PSU control loop BW 1-10 kHz -> it only takes over after")
print("  100 us to 1 ms. Between ~10 us and ~100 us neither the card's bulk")
print("  capacitance nor the PSU loop dominates, and that is where the step lands.")

print()
print("=" * 74)
print("GROUND RETURN / COMMON-MODE SHIFT:  V = L * di/dt")
print("  This is the part that could plausibly reach the PCIe PHY.")
print("=" * 74)
print(f"  {'t_rise':>8} {'di/dt 1 GPU':>14} {'di/dt coherent':>16} {'V @1nH':>9} {'V @5nH':>9} {'V @20nH':>9}")
for t in (1e-6, 1e-5, 1e-4):
    s1 = dI / t
    sc = dI_coh / t
    row = f"  {t*1e6:6.0f} us {s1/1e6:11.2f} A/us {sc/1e6:13.2f} A/us"
    for L in (1e-9, 5e-9, 20e-9):
        row += f" {L*sc*1e3:6.1f} mV"
    print(row)
print()
print("  For scale: PCIe Gen5 at the receiver, after channel loss, has a")
print("  differential eye height in the tens of mV. A common-mode reference")
print("  shift of that order between the GPU and its root port is significant,")
print("  and any single-ended imbalance converts part of it to differential noise.")
print()
print("  This predicts exactly the observed signature: the LINK fails while")
print("  compute stays correct (faulty=0 nan=0 in all 21 runs). It also predicts")
print("  coherence matters, because ground return currents SUM -- which is")
print("  hypothesis 9 -- and that a shared upstream port is the worst place to")
print("  aggregate them, which is where rgca18's all-8 containment happened.")
