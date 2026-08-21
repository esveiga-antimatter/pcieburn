#!/usr/bin/env python3
"""Bayesian model comparison over the 26-run corpus.

Hazard model per run:  lambda = exp( a_node + bP*x + bU*u + bW*w )
  x = (meanW - 450)/50      (power, matched-window mean)
  u = ln(uptime/1000)       (within-boot soak)
  w = prior fault count     (wear / cumulative damage)
Sub-models zero out coefficients. Likelihood: fault at t -> ln(lam) - lam*t;
clean for T seconds -> -lam*T.  Flat priors over declared grids; marginal
likelihood = mean over grid, so extra parameters pay an automatic Occam
penalty. n is small: read Bayes factors as orders of magnitude, not truth.
"""
import math

# node, uptime_s, meanW, wear(prior faults), t_end_s, fault(0/1)
DATA = [
    # cor04 (chronological)
    ("cor04",   697, 496, 0, 282.2, 1),
    ("cor04",  1003, 437, 1, 600.0, 0),
    ("cor04",  1675, 479, 1, 600.0, 0),
    ("cor04",  2441, 396, 1, 214.5, 1),
    ("cor04",  3067, 496, 2, 106.9, 1),
    ("cor04",   216, 438, 3, 600.0, 0),
    ("cor04", 33347, 440, 3, 224.2, 1),
    ("cor04",   687, 284, 4, 600.0, 0),
    ("cor04",  2165, 292, 4, 600.0, 0),
    ("cor04",  4241, 541, 4,  95.1, 1),
    # rgca17 — all clean
    ("rgca17",  697, 399, 0, 600.0, 0), ("rgca17", 1362, 360, 0, 600.0, 0),
    ("rgca17", 2035, 387, 0, 600.0, 0), ("rgca17", 2800, 326, 0, 600.0, 0),
    ("rgca17", 6938, 526, 0, 600.0, 0), ("rgca17",  204, 418, 0, 600.0, 0),
    ("rgca17",33335, 418, 0, 600.0, 0), ("rgca17",  682, 153, 0, 600.0, 0),
    ("rgca17", 2159, 156, 0, 600.0, 0), ("rgca17", 4235, 574, 0, 600.0, 0),
    # rgca18 (pl450-cold's post-window fault counted as fault at ~609 s)
    ("rgca18", 1132, 397, 0, 600.0, 0), ("rgca18", 1395, 359, 0, 600.0, 0),
    ("rgca18", 2068, 387, 0, 600.0, 0), ("rgca18", 2833, 318, 0, 600.0, 0),
    ("rgca18", 6971, 522, 0, 598.2, 1), ("rgca18",  222, 417, 1, 609.0, 1),
    # batch of 2026-08-19/20: up1h idle test + baseline-uncapped up700s rerun
    ("cor04",  4082, 438, 5, 600.0, 0),   # pl450-8192 @1h idle: clean
    ("cor04",   812, 494, 5,  65.6, 1),   # baseline rerun @~700s: FAULT (GPU0, ERR_FATAL)
    ("rgca17", 4078, 417, 0, 600.0, 0), ("rgca17",  807, 397, 0, 600.0, 0),
    ("rgca18", 4078, 417, 2, 600.0, 0), ("rgca18",  839, 394, 2, 600.0, 0),
    # up700s repeat 2 (2026-08-20, actual uptimes ~1500 s)
    ("cor04",  1499, 493, 6,  60.7, 1),   # GPU0 ERR_FATAL again
    ("rgca17", 1495, 402, 0, 201.4, 1),   # FIRST rgca17 fault: gpu6 ERR_FATAL
    ("rgca18", 1527, 398, 2, 600.0, 0),
]
NODES = ["cor04", "rgca17", "rgca18"]
A_GRID  = [(-14.0 + 0.5 * i) for i in range(21)]          # ln base hazard /s
BP_GRID = [0.25 * i for i in range(13)]                    # 0..3 per +50 W
BU_GRID = [-0.5 + 0.25 * i for i in range(9)]              # -0.5..1.5 per e-fold uptime
BW_GRID = [-0.5 + 0.25 * i for i in range(9)]              # -0.5..1.5 per prior fault

def covs(up, P, w):
    return ((P - 450) / 50.0, math.log(up / 1000.0), float(w))

def node_lnL(node, a, bP, bU, bW):
    s = 0.0
    for n, up, P, w, t, f in DATA:
        if n != node:
            continue
        x, u, wr = covs(up, P, w)
        lam = math.exp(a + bP * x + bU * u + bW * wr)
        s += (math.log(lam) - lam * t) if f else (-lam * t)
    return s

def lse(v):
    m = max(v)
    return m + math.log(sum(math.exp(x - m) for x in v))

def model(bPs, bUs, bWs):
    """Return (lnML, MAP betas, beta posterior weights)."""
    per_beta = []
    betas = []
    for bP in bPs:
        for bU in bUs:
            for bW in bWs:
                tot = 0.0
                for nd in NODES:
                    tot += lse([node_lnL(nd, a, bP, bU, bW) for a in A_GRID]) \
                           - math.log(len(A_GRID))
                per_beta.append(tot)
                betas.append((bP, bU, bW))
    lnML = lse(per_beta) - math.log(len(per_beta))
    mi = max(range(len(per_beta)), key=lambda i: per_beta[i])
    mx = per_beta[mi]
    wts = [math.exp(v - mx) for v in per_beta]
    tot = sum(wts)
    wts = [w / tot for w in wts]
    return lnML, betas[mi], list(zip(betas, wts))

def predict(node, up, P, w, post, dur=600.0):
    """Posterior-predictive P(fault within dur) for one cell."""
    x, u, wr = covs(up, P, w)
    acc = 0.0
    for (bP, bU, bW), wt in post:
        if wt < 1e-6:
            continue
        lls = [node_lnL(node, a, bP, bU, bW) for a in A_GRID]
        m = max(lls)
        aw = [math.exp(v - m) for v in lls]
        s = sum(aw)
        p = 0.0
        for a, w_a in zip(A_GRID, aw):
            lam = math.exp(a + bP * x + bU * u + bW * wr)
            p += (w_a / s) * (1.0 - math.exp(-lam * dur))
        acc += wt * p
    return acc

MODELS = {
    "M0 global-only (no slot term)": None,   # special-cased below
    "MA slot only":                  ([0.0], [0.0], [0.0]),
    "MA+P slot x power":             (BP_GRID, [0.0], [0.0]),
    "MA+U slot x soak":              ([0.0], BU_GRID, [0.0]),
    "MA+W slot x wear":              ([0.0], [0.0], BW_GRID),
    "MA+P+U":                        (BP_GRID, BU_GRID, [0.0]),
    "MA+P+W":                        (BP_GRID, [0.0], BW_GRID),
    "MA+P+U+W (full)":               (BP_GRID, BU_GRID, BW_GRID),
}

results = {}
for name, spec in MODELS.items():
    if spec is None:
        # one shared a for all runs, no node term
        vals = []
        for a in A_GRID:
            tot = sum(node_lnL(nd, a, 0, 0, 0) for nd in NODES)
            vals.append(tot)
        lnML = lse(vals) - math.log(len(vals))
        results[name] = (lnML, (0, 0, 0), None)
    else:
        results[name] = model(*spec)

best = max(v[0] for v in results.values())
print("=" * 86)
print("MODEL COMPARISON  (flat priors on declared grids; lnBF vs best model)")
print("=" * 86)
for name, (lnML, mapb, _) in sorted(results.items(), key=lambda kv: -kv[1][0]):
    print(f"  {name:<32} lnML={lnML:8.2f}  lnBF={lnML-best:7.2f}"
          f"   MAP: bP={mapb[0]:+.2f} bU={mapb[1]:+.2f} bW={mapb[2]:+.2f}")

print()
print("=" * 86)
print("POSTERIOR-PREDICTIVE P(fault in 600 s) FOR PLANNED CELLS, per surviving model")
print("=" * 86)
CELLS = [
    ("T1 cor04  soak-1h  pl450-8192 (up=3600,P=440,w=5)", "cor04", 3600, 440, 5),
    ("T2 rgca17 soak-1h  pl450-8192 (up=3600,P=420,w=0)", "rgca17", 3600, 420, 0),
    ("T3 cor04  baseline-rerun      (up=700, P=496,w=5)", "cor04",  700, 496, 5),
    ("T4 cor04  low-power arm       (up=3600,P=290,w=5)", "cor04", 3600, 290, 5),
    ("T5 cor04  deep-soak repeat    (up=33000,P=440,w=5)", "cor04", 33000, 440, 5),
]
names = ["MA slot only", "MA+P slot x power", "MA+U slot x soak",
         "MA+W slot x wear", "MA+P+U", "MA+P+W"]
hdr = f"  {'cell':<52}" + "".join(f"{n.split()[0]:>9}" for n in names)
print(hdr)
for label, nd, up, P, w in CELLS:
    row = f"  {label:<52}"
    for n in names:
        post = results[n][2]
        row += f"{predict(nd, up, P, w, post):>9.2f}"
    print(row)
print()
print("  Disagreement across surviving models = expected information of the run.")
