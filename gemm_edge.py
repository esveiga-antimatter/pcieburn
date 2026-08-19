#!/usr/bin/env python3
"""Ballpark the LOAD edge at a GEMM boundary from tile-wave granularity.

Within an unpaced burst, consecutive GEMMs overlap: blocks of GEMM k+1 fill SMs
as blocks of GEMM k retire, so there is no drain. A real drain happens only at
the burst->collective boundary, and its duration is set by how coherently the
last wave of thread blocks finishes.

Measured: 8192-dim single GEMM = 17.404 ms (rgca17 pl575, events.csv).
RTX 5090 (GB202): 170 SMs.
ASSUMED: cuBLAS SGEMM tile 128x128 or 128x64; 1-2 concurrent blocks per SM.
"""
SMS = 170
MEASURED = {8192: 17.404e-3, 2048: 0.121e-3}   # s per GEMM, measured

def waves(dim, tile_m, tile_n, blocks_per_sm):
    tiles = (dim / tile_m) * (dim / tile_n)
    resident = SMS * blocks_per_sm
    return tiles, resident, tiles / resident

print("=" * 78)
print("HOW MANY TILE-WAVES IS ONE GEMM?  (wave time = kernel time / waves)")
print("=" * 78)
for dim, t in MEASURED.items():
    print(f"\n  matrix-dim {dim}, measured kernel {t*1e3:.3f} ms")
    for (tm, tn) in ((128, 128), (128, 64)):
        for bps in (1, 2):
            tiles, res, w = waves(dim, tm, tn, bps)
            wt = t / w
            print(f"    tile {tm}x{tn}, {bps} blk/SM -> {tiles:8.0f} blocks, "
                  f"{res:4.0f} resident, {w:7.2f} waves, wave = {wt*1e6:9.1f} us")

print()
print("=" * 78)
print("WHAT THAT IMPLIES FOR THE FALLING EDGE AT A BURST BOUNDARY")
print("=" * 78)
print("  Rising edge: the block scheduler dispatches thousands of blocks in")
print("  microseconds, so occupancy -- and current demand -- ramps in ~us.")
print()
print("  Falling edge: bounded below by how coherently the final wave retires.")
print("  Best case the last wave finishes together (edge ~us). Worst case the")
print("  raggedness spans a full wave. So the drain sits between a few us and")
print("  one wave time above.")
print()
print("=" * 78)
print("SPECTRAL KNEE FOR EACH CANDIDATE EDGE, vs VRM loop BW 10-100 kHz")
print("=" * 78)
print(f"  {'edge':>12} {'knee = 1/(pi*t)':>18}   {'verdict':>34}")
for t, note in ((5e-6, "coherent last wave"),
                (50e-6, "mildly ragged"),
                (200e-6, "~1/4 wave (8192)"),
                (720e-6, "one full wave (8192)")):
    knee = 1 / (3.14159 * t)
    if knee > 1e5:   v = "ABOVE the loop BW"
    elif knee > 1e4: v = "INSIDE the loop BW"
    else:            v = "BELOW the loop BW"
    print(f"  {t*1e6:9.0f} us {knee/1e3:15.1f} kHz   {v:>34}  ({note})")

print()
print("=" * 78)
print("THE MEASUREMENT THE TOOL CAN ALREADY MAKE")
print("=" * 78)
print("  GEMM-window GFLOP/s is cudaEvent-timed (~0.5 us resolution) and")
print("  averaged over thousands of GEMMs. At --gemms-per-coll 94 the GEMMs are")
print("  back-to-back and pay no ramp; at --gemms-per-coll 1 every GEMM pays a")
print("  full drain + refill. The difference in effective per-GEMM time IS the")
print("  transition cost:")
print()
t94 = MEASURED[8192]
print(f"    measured at N=94 (back-to-back): {t94*1e3:.3f} ms/GEMM")
for pct in (0.1, 0.5, 2.0, 5.0):
    t1 = t94 * (1 + pct / 100)
    print(f"    if N=1 measures {t1*1e3:.3f} ms/GEMM ({pct:>4.1f}% slower)"
          f" -> ramp+drain = {(t1-t94)*1e6:7.1f} us")
print()
print("  One 600 s run at --gemms-per-coll 1 therefore yields the edge time to")
print("  a few us, with no extra instrumentation -- and it is the same run")
print("  hypothesis 12's sweep already needs.")
