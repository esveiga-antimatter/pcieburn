#!/usr/bin/env python3
"""How many runs per arm? Worked from the observed fault behaviour."""
from math import comb, log, sqrt

print("=" * 72)
print("1. BINARY SCREEN: can n runs/arm separate 'faults' from 'never faults'?")
print("   Fisher exact, one-sided, two arms of n runs each.")
print("=" * 72)
print(f"{'n/arm':>5} {'perfect n/n vs 0/n':>20} {'one discordant':>28}")
for n in range(2, 8):
    # perfect separation: all n faults in A, none in B
    p_perfect = comb(n, n) * comb(n, 0) / comb(2 * n, n)
    # one discordant: n-1 faults in A, 0 in B  -> P(X >= n-1)
    k = n - 1
    tot = k
    p_disc = sum(comb(n, i) * comb(n, tot - i) / comb(2 * n, tot)
                 for i in range(k, min(n, tot) + 1))
    print(f"{n:>5} {'p = ' + format(p_perfect, '.4f'):>20} "
          f"{'(' + str(n-1) + '/' + str(n) + ' vs 0/' + str(n) + ')  p = ' + format(p_disc, '.4f'):>28}")

print()
print("=" * 72)
print("2. ESTIMATING one arm's per-run fault probability (Clopper-Pearson 95%)")
print("=" * 72)
def cp_lo(k, n):
    # lower bound: solve for p where P(X>=k|p)=0.025, via bisection
    if k == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        tail = sum(comb(n, i) * mid**i * (1-mid)**(n-i) for i in range(k, n+1))
        if tail < 0.025: lo = mid
        else: hi = mid
    return lo
def cp_hi(k, n):
    if k == n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        tail = sum(comb(n, i) * mid**i * (1-mid)**(n-i) for i in range(0, k+1))
        if tail > 0.025: lo = mid
        else: hi = mid
    return lo
for n in (1, 3, 5, 8):
    for k in (n, n - 1 if n > 1 else 0, 0):
        if k < 0: continue
        print(f"   {k}/{n} faults -> p in [{cp_lo(k,n):.2f}, {cp_hi(k,n):.2f}]")
    print()

print("=" * 72)
print("3. TIME-TO-FAULT (exponential hazard): precision is set by FAULT COUNT,")
print("   not run count. Relative standard error of the rate = 1/sqrt(faults).")
print("=" * 72)
for k in (1, 3, 4, 5, 8, 11, 20):
    print(f"   {k:>2} faults -> rate known to +/-{100/sqrt(k):.0f}%"
          f"{'   <- ~detects a 2x difference' if k in (8,11) else ''}")

print()
print("=" * 72)
print("4. FOR ARMS THAT COME BACK CLEAN: exposure beats repetition.")
print("   Zero faults in exposure E -> 95% upper bound on hazard = 3/E.")
print("=" * 72)
plans = [("5 x 600 s", 5*600), ("3 x 600 s", 3*600), ("3 x 3600 s", 3*3600),
         ("1 x 3600 s", 3600), ("5 x 3600 s", 5*3600)]
for name, E in plans:
    mtbf = E / 3
    print(f"   {name:<12} exposure {E:>6} s -> hazard < 1 per {mtbf:>6.0f} s of load")

print()
print("=" * 72)
print("5. OBSERVED, for calibration (cor04, 7 arms on 933b21f)")
print("=" * 72)
loads = [282, 600, 600, 214, 107, 600, 224]   # per-arm load seconds until fault or end
faults = 4
E = sum(loads)
print(f"   total load exposure {E} s = {E/60:.0f} min, {faults} faults")
print(f"   pooled hazard  = 1 fault per {E/faults:.0f} s of load ({E/faults/60:.1f} min)")
print(f"   +/-{100/sqrt(faults):.0f}% on that rate (4 faults)")
print(f"   per-600s-run fault probability ~= {1 - pow(2.718281828, -600*faults/E):.2f}")
