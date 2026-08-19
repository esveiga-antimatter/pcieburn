#!/usr/bin/env python3
"""What can and cannot be calculated about the current edge at a GEMM boundary.

Measured input (NVML envelope, matched window, 8192-single arm):
  dP = 193 W per GPU  -- a LOWER bound: NVML averages over ~300-400 ms
  12 V rail, 8 GPUs coherent
Everything else is flagged ASSUMED, which is why this yields bounds, not values.

THE THREE STAGES ARE DIFFERENT THINGS. An earlier version of this script treated
"VRM loop bandwidth" and "PSU loop bandwidth" as two points on one continuum and
described the card VRM as absorbing a transient the PSU would later take over.
That is wrong on both counts:

  1. Core VRM on the card (12 V -> ~1 V multiphase buck, switching 0.5-1.5 MHz,
     loop BW ~50-150 kHz) regulates CORE VOLTAGE. Its output-impedance peak
     produces core droop, which manifests as compute errors or clock instability.
     No run in this investigation has ever shown data corruption (faulty=0 nan=0
     in 21 runs), so this is probably not the failure path.
  2. A converter does not absorb an input-side transient -- it CREATES one. When
     core load steps up the VRM draws more input current, so a FASTER VRM loop
     propagates the step to the 12 V rail SOONER. What limits how fast the rail
     sees it is the card's input filter, not the loop.
  3. The PCIe PHY is not on the core rail. It runs from its own regulator off 12 V
     (or the slot's 3.3 V), so PHY supply integrity is a question about the 12 V
     input network -- whose relevant peak is an L-C ANTI-RESONANCE between input
     capacitance and cable/connector inductance, not any control-loop bandwidth.
     The system PSU loop (~1-10 kHz) governs the rail over hundreds of us and
     longer, i.e. sustained steps rather than one edge.
"""
import math

V = 12.0
dP = 193.0
N = 8
dI = dP / V
dI_coh = dI * N

print("=" * 76)
print("MEASURED / DERIVED WITHOUT ASSUMPTIONS")
print("=" * 76)
print(f"  step per GPU        {dP:6.0f} W  -> {dI:6.2f} A on 12 V")
print(f"  coherent across {N}   {dP*N:6.0f} W  -> {dI_coh:6.2f} A")

print()
print("=" * 76)
print("STAGE 2: the 12 V input network's L-C ANTI-RESONANCE")
print("  f = 1 / (2*pi*sqrt(L*C)).  This is the peak a 12 V-side edge can ring.")
print("=" * 76)
print(f"  {'L (cable+conn)':>16} {'C (card input)':>16} {'f_res':>12} {'ring period':>14}")
for L in (100e-9, 200e-9, 500e-9):
    for C in (500e-6, 1e-3, 2e-3):
        f = 1 / (2 * math.pi * math.sqrt(L * C))
        print(f"  {L*1e9:13.0f} nH {C*1e6:13.0f} uF {f/1e3:9.1f} kHz {1/f*1e6:11.1f} us")
print("  ASSUMED ranges; both need a teardown or vendor data to pin down.")
print("  Note how insensitive it is: ~5-22 kHz across the whole plausible box.")

print()
print("=" * 76)
print("WHICH EDGES EXCITE THAT ANTI-RESONANCE?")
print("  a square edge carries content to ~1/(pi*t_rise)")
print("=" * 76)
print(f"  {'edge':>10} {'knee':>12}   {'vs ~10 kHz input anti-resonance':>34}")
for t in (5e-6, 20e-6, 50e-6, 200e-6, 720e-6):
    knee = 1 / (math.pi * t)
    if knee > 3e4:   v = "well above - strongly excites"
    elif knee > 5e3: v = "near - excites"
    else:            v = "below - weak excitation"
    print(f"  {t*1e6:7.0f} us {knee/1e3:9.1f} kHz   {v:>34}")
print()
print("  The tile-wave estimate for an 8192 GEMM puts the falling edge at")
print("  ~200-720 us (see gemm_edge.py), which would only WEAKLY ring the input")
print("  network. An edge of 20-50 us would hit it hard. That is precisely what")
print("  the --gemms-per-coll 1 run measures, so run it before theorising further.")

print()
print("=" * 76)
print("SAG AT THE CARD INPUT BEFORE ANY LOOP RESPONDS: dV = dI*t/C")
print("  Pure charge transfer out of the input bulk. Applies on timescales short")
print("  compared to the PSU loop (~1-10 kHz, so below ~100 us).")
print("=" * 76)
print(f"  {'t':>8} " + " ".join(f"{c*1e6:>9.0f}uF" for c in (500e-6, 1e-3, 2e-3)))
for t in (1e-6, 10e-6, 50e-6, 100e-6):
    row = f"  {t*1e6:6.0f}us "
    for C in (500e-6, 1e-3, 2e-3):
        row += f" {dI*t/C*1e3:8.1f}mV"
    print(row)
print("  Beyond ~100 us the PSU output capacitance and then its control loop")
print("  take over, so these numbers stop being the whole story there.")

print()
print("=" * 76)
print("GROUND RETURN / COMMON-MODE SHIFT:  V = L * di/dt")
print("  The one path that plausibly reaches the PCIe PHY's reference.")
print("=" * 76)
print(f"  {'edge':>8} {'di/dt coherent':>16} {'@1nH':>9} {'@5nH':>9} {'@20nH':>9}")
for t in (5e-6, 20e-6, 50e-6, 200e-6, 720e-6):
    sc = dI_coh / t
    row = f"  {t*1e6:6.0f}us {sc/1e6:13.2f} A/us"
    for L in (1e-9, 5e-9, 20e-9):
        row += f" {L*sc*1e3:7.1f}mV"
    print(row)
print()
print("  PCIe Gen5 differential eye height at the receiver, after channel loss,")
print("  is in the tens of mV. A common-mode reference shift of that order")
print("  between a GPU and its root port is significant, and single-ended")
print("  imbalance converts part of it to differential noise.")
print()
print("  This predicts the observed signature: the LINK fails while compute stays")
print("  correct in every one of 21 runs. It predicts coherence matters, because")
print("  return currents SUM (hypothesis 9). And it predicts a shared upstream")
print("  port is the worst aggregation point -- where rgca18's all-8 containment")
print("  fired, through 1a:01.1.")
