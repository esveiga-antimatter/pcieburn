# Investigation status, hypothesis ledger, and analysis discipline

This file is the fixed reference for the PCIe link-fault investigation using
`pcieburn`. It exists because a full round of measurements had to be discarded,
and because several confident conclusions along the way turned out to be
artifacts of how the data was read rather than facts about the hardware.

**If you are an agent or engineer about to analyse a run directory, read
[Analysis discipline](#analysis-discipline) first.** It is not general advice —
every item is a mistake that was actually made here, and most of them produced a
plausible-sounding wrong answer that survived several turns before being caught.

---

## Status

**Code freeze: `933b21f`** (`logging: prioritizes csv instead of stdout`).

**All measurements taken before this freeze are scratched.** Runs on different
nodes were made with three different commits (`3b31bc9`, `6cd50e6`, `5d97182`),
so no cross-node comparison from that period is valid. Those rounds did real
work — they validated the harness as a harness and produced roughly a dozen
fixes to it — but they yield no reliable analysis.

Prior indications from that period are recorded only so the same ground is not
re-covered blindly, never as evidence.

**Post-freeze measurement is under way.** Thirteen batches across three nodes are
recorded in [Post-freeze results](#post-freeze-results-933b21f), 38 runs and
**ten containments**. They refute hypothesis 1 outright, refute hypothesis 12,
demote hypothesis 4 from confirmed-causal, and establish that no single-factor
model fits. **Read the corrections subsection first** — three confident claims
from an earlier revision of this file were wrong, one of them because the harness
reported a fatal run as `clean`.

**Then read [Full-corpus re-read](#full-corpus-re-read-all-38-runs-from-raw-artifacts).**
A pass over every bundle from the raw artifacts retired four further standing
claims, resolved h13's mask prerequisite and h14's ambient confound from data
already on disk, found a **second false negative** (a run whose host died
mid-run and whose blank `verdict` field was read as clean), reclassified the
rgca18 "setup error" run as post-fault state, and surfaced a covariate this file
was not tracking at all: **cor04's kernel changed mid-campaign, and it separates
the long- and short-TTF faults perfectly.** Three defects in `corr_matrix.py` are
also documented and fixed there; figures quoted as "27 runs" or "26 runs" earlier
in this file are that script's silently truncated subset, not the corpus.

### Binary identity is carried in the run tag

The manifest records the *source* commit but nothing verifies the **binary** was
built from it — `pcieburn` is untracked, so two nodes can sit on identical commits
and run different binaries. Rather than change frozen code, both identities go in
the tag, where they land in the run directory name and the `events.csv` tag
column:

```sh
BIN=./pcieburn
TAG="$(hostname)-pl575-g$(git rev-parse --short=7 HEAD)-b$(sha256sum "$BIN" | cut -c1-8)"
```

Read the pair together:

| both match across nodes | fully comparable |
|---|---|
| git matches, binary differs | source identical, **toolchain or build path differs** |
| git differs | source differs — invalid, stop |

The middle row is why both are needed. A binary-hash mismatch does **not** by
itself mean the code differs: `nvcc -lineinfo` embeds source paths, and a
different CUDA patch level yields a different binary from identical source. The
hash is asymmetric — a match is a strong guarantee, a mismatch means investigate.

Two things to watch. Hash the binary the wrapper will actually run: it honours
`PCIEBURN_BIN`, so hashing `./pcieburn` while it executes something else puts a
truthful-looking but wrong hash in the tag. And the manifest records the driver
version but **not the CUDA toolkit version**, so a toolchain difference is
otherwise invisible — capture `nvcc --version | tail -1` per node once at setup.

---

## What survives the scratch

These come from kernel logs, PCI config registers, IPMI/Redfish, or vendor
source review. None depends on which `pcieburn` binary ran, so all of it stands.

**Platform / error handling**

- `_OSC` AER ownership differs between COR04 and RGCA nodes despite *identical*
  board (`TURIN2D24G-2L+/500W`) and BIOS version (`10.14 01/23/2025`). It is
  therefore a BIOS **setting**, not firmware or hardware. RGCA has since been set
  to OS-first.
- On the switchboard nodes, **DPC is enabled only at root ports**, never at switch
  downstream ports. Containment there is necessarily fabric-wide: one GPU's fatal
  error takes down every GPU behind that switch. Multiple GPUs dropping together
  on a switch node is expected scope, *not* an escalation.
- **Corrected (38-run re-read).** DPC events split into two register-level
  classes, and the earlier blanket claim — "every DPC event reads `ERR_FATAL
  received from <device>`, therefore a momentary power dropout is not the
  mechanism" — holds for only 7 of 10. The DPC Status register's Trigger Reason
  field, bits [2:1], separates them and the kernel prints its own decode:
  - `status:0x1f05` → reason `10`, *`ERR_FATAL` message received*, source device
    named. Six in-window events, on `00:01.1` and `1a:01.1`. For these the
    dropout argument stands: the endpoint transmitted its own escalation.
  - `status:0x1f01` → reason `00`, *unmasked uncorrectable error detected at the
    port*, decoded `severity=Uncorrectable (Fatal), type=Transaction Layer,
    (Receiver ID)`, first error `[5] SDES`, **no source device**. Three events,
    all cor04 `a0:01.1`/GPU5. The GPU transmitted nothing; the root port latched
    Surprise Down on its own detection and then logged `broken device, retraining
    non-functional downstream link at 2.5GT/s`.

  So the dropout argument must be scoped to the `ERR_FATAL` class. It says
  nothing about cor04 GPU5, where the observable is exactly "the device stopped
  transmitting."
- Error direction is consistent: corruption is detected at the *receiving* port
  with the GPU as transmitter. That implicates the GPU's TX path and the physical
  channel, not the root port's transmitter.

**Instrumentation limits**

- PSU watt sensors are unusable above idle: `PWR_*_PIN` wraps at 255 W,
  `PWR_*_POUT` saturates near 510 W, and DCMI derives from `PIN` (it printed
  `3 W` at load onset). `CUR_PSU*_IOUT` is the only in-range power channel;
  multiply by the measured 12 V rail.
- BMC `CurConsumedWatts` is a torn read — it changed in 189 of 326 samples while
  the sensors feeding it changed ~56 times.
- NVML `power.draw` refreshes only every ~0.5–1.2 s despite 100 ms polling, so
  neither it nor any BMC channel can resolve sub-millisecond transients. Claims
  about *transients* need a scope or inline analyser; claims about *level* do not.
- Sub-maximum idle link speed is **not** by itself a defect indicator. All 8 RGCA
  links idle at Gen1 with ASPM disabled on every root port, and return to Gen5
  when recently active. Only a link that differs from its peers *on the same
  machine* is interesting, and even that has not predicted a failure.

**Source-review facts (why gpu-burn differs from diag and pcieburn)**

| | gpu-burn (defaults) | DCGM `diagnostic` | pcieburn |
|---|---|---|---|
| matrix dim | 8192 (`#define SIZE`) | 2048 | 2048 |
| precision | FP32 only | half + single | half + single |
| tensor cores | off (`-tc` opt-in) | on at halfway | on at halfway |
| launch pacing | depth-2 events + `usleep` | none | none |
| reproduces? | no | yes | yes |

Kernel launch rate differs by **~65–480×** between them (roughly 42/s for
8192-dim FP32 versus 2,700–20,000/s for 2048-dim). This is the basis of
hypothesis 1.

**Pre-existing cross-tool pattern (from the runbook, independent of pcieburn)**

`diag` reproduces the fault while moving *zero* bytes between GPUs;
`pcie` / `memory_bandwidth` / `nccl-tests` saturate the bus and have never
reproduced it. PCIe payload volume is not the driver. Note also that a PCIe link
in L0 transmits continuously — utilization affects payload, not signalling — so
"only 4% of Gen5 x16" says nothing about channel stress.

---

## Post-freeze results (`933b21f`)

Seven arms × three nodes, 600 s each, on the frozen binary. **The goal here is the
platform's margin envelope, not a single guilty part.** An arm that eliminates a
candidate has narrowed the envelope, and a *clean* run that accumulated enormous
error counts is more informative than one that simply failed.

### Corrections to earlier readings of this same data

Three claims that were stated confidently in an earlier revision of this file are
false. They are recorded here because each was produced by a specific, repeatable
mistake, not by bad luck.

1. **"rgca18 logged 694,377 correctable errors and stayed clean at 450 W."** It
   did not stay clean. It took a `DPC`/`ERR_FATAL` containment on the same port at
   `01:42:16`, **8.8 s after the load phase ended and 4.2 s after the supervisor's
   `finish` event** — and 5 s after the wrapper's final `dmesg_after` snapshot at
   `01:42:11`. The harness could not have seen it; the verdict `clean` and
   `exit_code 0` are a false negative. The storm *did* escalate.
2. **"All arms ran back-to-back on a single boot per node."** True for the RGCA
   nodes (four arms on the `23:07` boot) but false for cor04, which had **four
   boot sessions** — `22:38`, `23:13`, `00:12`, `01:28` — because each of its
   faults forced a reboot.
3. **"Time-to-fault extrapolates to zero at ~85 min uptime."** Refuted directly:
   cor04 at **33,347 s** uptime faulted at 224 s, not near zero. The three points
   the line was fitted through also spanned three different boots, so it was never
   a within-boot soak trend.

### Every containment in the corpus (10, over 38 runs)

Harvested by deduping every `DPC: containment event` line across every `dmesg`
snapshot in every bundle by (node, wall-clock second, port). Ten events, nine
inside a run's own capture window plus fault 5 at +1.0 s past its window. No
others hide in any gap. Bounded, as before, by the three run tails a following
reboot put beyond `dmesg` reach.

| # | node | arm | t | port | GPU | DPC reason | signature | radius |
|---|---|---|---|---|---|---|---|---|
| 1 | cor04 | uncapped 2048 | 282.2 s | `a0:01.1` | GPU5 `a1:00.0` | `0x1f01` / `00` | `SDES`, no correctables anywhere on that port, ever | 1 GPU |
| 2 | cor04 | lgc2100 | 214.5 s | `00:01.1` | GPU0 `01:00.0` | `0x1f05` / `10` | `ERR_FATAL`; 1 correctable log line, same second; **AER delta reads all zero** | 1 GPU |
| 3 | cor04 | pl575 8192 | 106.9 s | `a0:01.1` | GPU5 `a1:00.0` | `0x1f01` / `00` | `SDES`, correctables on GPU0's port 27 s earlier | 1 GPU |
| 4 | rgca18 | pl575 8192 | 598.1 s | `1a:01.1` | GPU4 `2d:00.0` | `0x1f05` / `10` | 1836 RxErr → `ERR_FATAL`; **1.9 s before load end** | **all 8** |
| 5 | rgca18 | pl450 8192 | 608.8 s | `1a:01.1` | GPU4 `2d:00.0` | `0x1f05` / `10` | 694,377 RxErr → `ERR_FATAL`; **8.8 s after load end**, outside the window | **all 8** |
| 6 | cor04 | pl450-warm 8192 | 224.2 s | `00:01.1` | GPU0 `01:00.0` | `0x1f05` / `10` | `ERR_FATAL`, 3954 BadTLP | 1 GPU |
| 7 | cor04 | coll4M 8192 | 95.1 s | `a0:01.1` | GPU5 `a1:00.0` | `0x1f01` / `00` | `SDES`; 922 BadTLP on GPU0's port, last 19 s earlier | 1 GPU |
| 8 | cor04 | uncapped 2048 | 65.6 s | `00:01.1` | GPU0 `01:00.0` | `0x1f05` / `10` | `ERR_FATAL`, 8 correctable lines spanning 4 s | 1 GPU |
| 9 | cor04 | uncapped 2048 | 60.7 s | `00:01.1` | GPU0 `01:00.0` | `0x1f05` / `10` | `ERR_FATAL`, 15 correctable lines spanning 4 s | 1 GPU |
| 10 | rgca17 | uncapped 2048 | 201.4 s | `1a:01.1` | gpu6 `36:00.0` | `0x1f05` / `10` | `ERR_FATAL`; 1 RxErr / 1 BadTLP on `34:10.0` | **all 8** |

cor04 7, rgca18 2, rgca17 1. `faulty=0 nan=0` in **38 of 38 runs** — data
corruption has still never been observed.

**Victim alternation on cor04 is borderline and has broken.** GPU5, GPU0, GPU5,
GPU0, GPU5, GPU0, GPU0 — 5 of 6 consecutive pairs alternate, p = 0.109 against a
fair coin. Do not build on it.

**Xid 154's recovery action is not a reliable reset-versus-reboot input.** cor04
requested `0x2 (Node Reboot Required)` in all seven of its events and rgca17 in
its one, but rgca18 requested `0x2` for fault 4 and **`0x1 (GPU Reset Required)`
for fault 5** — and after fault 5 all eight GPUs stayed invisible to the OS for
9 h 15 m until a reboot (see the reclassified run below). Automation keyed on
this field would have tried a GPU reset on an unrecoverable node.

### Soak is causal: the cleanest single-variable result so far

cor04, boot `01:28`, same cap, same workload, same binary, **nothing touched in
between**:

| uptime at load start | outcome |
|---|---|
| 216 s | clean through 600 s, no fatal anywhere in the boot |
| 33,347 s (9.3 h) | **FAULT at 224 s** |

450 W was previously the only cap at which cor04 had never faulted. After nine
hours of uptime it faults at that cap in under four minutes. One variable moved.

The threshold therefore lies somewhere in `216 s < t < 33,347 s` — a 150× bracket
that needs bisection, and the shape inside it is unknown. What is now excluded is
a *linear* ramp to zero: at 9.3 h the time-to-fault was 224 s, longer than the
107 s seen at 51 min on a higher cap, so the response saturates rather than
collapsing.

### Both rgca18 faults sit on large coherent current transitions

The two rgca18 faults landed at opposite extremes of the load cycle, 11 s apart in
run time:

- **Fault 4** (`pl575`, t=598 s): the last valid NVML samples show **all eight
  GPUs pinned at 577–587 W**, i.e. clipping the cap simultaneously — roughly
  4.6 kW of coherent load at the instant of loss.
- **Fault 5** (`pl450`, t=609 s): the 8-GPU mean fell 358 → 112 → 92 → 79 → 47 →
  42 W across `01:42:10`–`01:42:15`, about **2 kW shed with the steepest step
  inside one sample**, and the containment fired one second after idle was
  reached.

The workload produces these steps coherently by construction: at `01:13:36.400`
all eight GPUs step together from ~465–506 W to ~576 W within a single 100 ms
sample. Per-GPU swing is 193 W (sd 56 W) in the 8192 arm against 144 W (sd 35 W)
in the 2048 arm, so roughly **1.5 kW of synchronised swing** across the chassis.

**Measurement limit:** NVML repeats the same power value across 3–4 consecutive
100 ms samples, so its effective update period is ~300–400 ms. It cannot resolve
the actual slew rate. The argument above rests on the coherent *step structure*
and the endpoints, not on a measured di/dt.

### What the 8192 workload actually changed

Not dispatch rate — that fell ~100× — but its interaction with the power cap:

| arm | samples within 3% of cap | clock sd |
|---|---|---|
| 2048 uncapped | **0.0%** | 229 MHz |
| 8192 pl575 | 35.4% | 197 MHz |
| 8192 pl450 | 47.1% | 298 MHz |

The 2048 workload never touches the cap; the 8192 workload spends a third to a
half of its time clipping, and clipping is *worse* at the lower cap. This is why
lowering the cap does not help and, on rgca18, coincided with a 378× rise in
correctable errors. Note that rgca17 clipped at 46.0% with **zero** errors, so
clipping alone is not sufficient — susceptibility gates it.

### rgca17 is the most degraded node, not the most robust

Its GPU6 port `34:10.0` trains to a **different state on every boot**, and its
error behaviour follows the trained state rather than the workload:

| boot | trained state | errors |
|---|---|---|
| `22:39` | stuck at **gen1 x16 for 265 s** (every other GPU: 5.5 s) before reaching gen5 | 1572 RxErr / 8018 BadTLP |
| `23:07` | clean gen5 x16 | 0 / 8 / 21 / 25 across four arms |
| `01:28` | **gen5 x8 — half width — for the entire run, twice** | zero, both runs |

At x8 it produced zero errors and two `clean` verdicts while running de-rated. So
rgca17's reputation as "the noisy node that never fails" was an artifact of one
boot's training outcome, and **its error counts are not comparable across boots.**
This is precisely the `DEGRADED` condition `linkcheck.sh` was written to detect,
and the run wrapper reported `clean` without flagging it.

### Edge rate, tested properly: refuted

Two arms were needed because the first was confounded. Both are recorded, because
the confounded one is what made the design error visible.

| arm | edges/s | mean W (t=60–93 s) | temp | PCIe | outcome |
|---|---|---|---|---|---|
| N=94, coll 128M–1G (reference) | 1.14 | 495.8 | 63.6 °C | 0.6% | **FAULT 107 s** |
| N=1, coll 128M–1G (confounded) | ~43 | 284.1 | 46.5 °C | 29.4% | clean x2, both hosts |
| **N=1, coll 4M (controlled)** | **107** | **541.4** | 65.4 °C | 0.5% | **FAULT 95 s** |

The controlled arm hit its design target — 53.7 collectives/s per rank, traffic
matched at 0.5% against 0.6% — by scaling collective size in proportion to N so
duty cycle held. Because `t_coll` scales with size, fixing duty fixes traffic
automatically and edge rate varies as 1/N.

**Result: ~90x the coherent current edges moved time-to-fault by 11%.** Power came
in 9% *above* the reference (the 4 MiB collective is partly latency-floored rather
than bandwidth-scaled, so duty overshot), which means the arm was harsher and still
did not fault meaningfully faster. Hypothesis 12 is refuted.

cor04's fault was its usual one: `a0:01.1` -> GPU5 `a1:00.0`, `SDES`,
`status:0x1f01`, GPU5 alone to x0 while the other seven held x16, and all eight
GPUs given `Xid 154` recovery action **0x2 (Node Reboot Required)** — a full reboot,
not a GPU reset. Afterwards `nvidia-smi -L` showed 7 of 8. Its tally is now GPU5 x3
(282, 107, 95 s) and GPU0 x2 (214, 224 s), still alternating with no pattern. For the
third time the correctables landed on **GPU0's** port (922 BadTLP on `00:01.1`)
while GPU5 died.

rgca17 stayed clean through the full 600 s and was **healthy afterwards** — nothing
in `dmesg` past the AER messages, `nvidia-smi -L` reporting all 8. But its
`34:10.0` did log **42 RxErr / 511 BadTLP**, the first non-zero on that port since
the `23:07` boot, at full x16 width, and episodic as always: 63 lines at t=60–120 s,
two at t=240–300, silence for the final 300 s. That cannot be attributed to edge
rate — against the confounded arm's zero, this one has 90x the edges *and* 2x the
power.

#### Why the first arm was confounded

`--gemms-per-coll` is **not** an edge-rate knob at fixed amplitude, which is how
hypothesis 12 originally specified it. Fewer GEMMs per collective means
proportionally more time *inside* the collective, so duty cycle and mean power fall
with it: at N=1 with the default collective, GEMM duty fell 94% -> 37% and mean power
494 -> 277 W. At 277 W the arm sat below any faulting power observed (437 W clean,
494 W faulted on cor04), so `clean` was explained by power and the arm said nothing
about edge rate. The fix was to scale collective size with N, holding duty.

**Do not compare the p2–p98 swing across arms with different duty cycles.** It reads
170 W at N=94 against 258 W at N=1, which looks like larger amplitude and is an
artifact: at N=94 the GPU is in the low state ~6% of the time, so NVML rarely samples
it and its ~400 ms averaging smears what it catches.

Also recorded from the confounded pair: **rgca17's gpu6 came back up at x16** —
62-sample normal ramp, full width — recovering from the x8 state it held across two
runs on the previous boot, with no intervention, and logged zero AER under 29.4% of
Gen5 x16. Its error behaviour tracks per-boot training outcome rather than workload.
Third independent vote for hypothesis 10.

### Cross-run correlation pass (27 runs, `corr_matrix.py`)

A systematic pass over every artifact bundle — one row per run (matched-window
power/temp/clipping, collective rate, TTF, per-port AER, link states) — testing
the ledger's correlations against the whole corpus at once. `corr_matrix.py`
regenerates the table. Descriptive statistics only: n is small and arms are
confounded, so the r values rank signals, they do not test them.

**Arms are not node-invariant — a topology-linked confound not previously listed.**
The byte-identical gpc1 arm (`--gemms-per-coll 1`, default 128M–1G collectives)
achieved **18.5 GB/s egress/rank on cor04 and 4.6 GB/s on rgca17 — 4.0x slower**
(21.1 vs 5.2 collectives/s, 284 W vs 153 W). Cause: an 8-way host-staged
collective funnels every rank's traffic through the switchboard's single shared
uplink, which saturates; cor04's eight independent root ports do not. The effect
vanishes for small collectives (coll4M: 54.9 vs 56.9 colls/s) and is mild in
full-burst arms (collective is ~6% of the pass). **Consequence: any arm whose
duty cycle is collective-dominated is systematically milder on RGCA nodes, and
"same arm" claims must be validated against achieved power and collective rate
per node, not per design.** rgca17's clean gpc1 runs at 153 W say nothing.

**cor04 has two distinct failure modes on its two slots — not one mechanism with
two victims.**

| slot | correctables across all runs | fault signature | faults |
|---|---|---|---|
| GPU0 `00:01.1` | BadTLP in **every** 8192 arm: 32,275 / 3,954 / 922 / 1 | `ERR_FATAL received from 01:00.0` — errors escalate to endpoint fatal | 214 s, 224 s |
| GPU5 `a0:01.1` | **zero, in all 27 runs** | `SDES` at the root port — sudden loss, no warning of any kind | 282, 107, 95 s |

The one "correctables on the dying port" case in the corpus (pl450-warm) is a
GPU0 fault — consistent with the split: GPU0 warns and then dies of its warnings;
GPU5 never warns. Any precursor-based detection can only ever cover the GPU0
class. Note also GPU0's BadTLP appears **only under the 8192 workload** and scales
*inversely* with cap (450 W: 32k/4k; 575 W: 1/922) — the term-C inverse-power
effect survives, but only within this slot.

**Hypothesis 8's clipping metric fails as a general predictor.** Across 26 runs,
r(clip%, AER) = +0.20 and r(clip%, fault) = +0.03. rgca17 ran the coll4M arm at
**100% clip (574 W pinned) and stayed clean**; lgc2100 faulted at 0% clip and
396 W. Clipping fraction is dropped as a stress metric; what remains of h8 is the
within-slot inverse-power correctable scaling above.

**No pooled uptime→TTF trend: r = +0.02 across the six faults.** The h4 evidence
remains the controlled pair (438 W clean at 216 s uptime vs 440 W fault at
33,347 s — power identical to within 2 W), not a dose-response curve. The
within-boot-family decline (282→214→107→95 s) spans four boots and four arms and
must not be read as one.

**Victim power rank is confirmed non-predictive** with the full corpus: GPU5 was
rank 1, 1, and 8 of 8 by power in its three faults; GPU0 rank 6 both times.

**Per-boot training state dominates switch-node error counts** (h10, full-corpus
check): the only anomalous gen1 dwell in all 27 runs (gpu6, 2,655 samples =
265 s) belongs to the 10,599-error run; both x8 trainings produced zero errors;
clean-x16 boots produce small episodic counts (9–553). Nothing else in the
corpus predicts rgca17's error volume.

### The idle test and the baseline re-run (wrapper v2, first `--settle` batch)

Six runs, 2026-08-19/20, all on wrapper v2 with the 30 s settle window active:

| cell | node | uptime | mean W | outcome |
|---|---|---|---|---|
| pl450-8192 after ~1 h idle | cor04 | 4082 s | 438 | clean |
| | rgca17 | 4078 s | 417 | clean |
| | rgca18 | 4078 s | 417 | clean |
| baseline-uncapped re-run @ ~700 s | cor04 | 812 s | **494** | **FAULT 65.6 s** (GPU0 `00:01.1`, ERR_FATAL) |
| | rgca17 | 807 s | 397 | clean |
| | rgca18 | 839 s | 394 | clean |

The re-run is a perfect power control: **494 W in the matched pre-fault window in
both the original and the re-run, to the watt** — same arm, same uptime (697 vs
812 s), same binary. TTF went **282.2 → 65.6 s**. Only two covariates moved: wear
(0 → 5 prior faults) and campaign time/ambient. Two honest qualifiers: the victim
was GPU0/ERR_FATAL where the original was GPU5/SDES, so the re-run sampled the
*other* failure mode; and under a constant-hazard null with mean 282 s, a draw
≤ 65.6 s has ~21% probability — direction matches wear, magnitude is one
exponential draw, not proof. The refit below is the sober version.

The idle hour produced no fault on any node and no width loss, so idle soak at
pl450 through 1 h is unremarkable — which, combined with the re-run, moves the
"uptime" story toward wear/campaign-time and away from within-boot time.

**Correction: "zero AER" in this batch was wrong, and only rgca18 was zero.**
Read straight from the three `aer_delta.txt` files:

| node | port | RxErr | BadTLP | BadDLLP | GPU Rollover |
|---|---|---|---|---|---|
| cor04 | `00:01.1` (GPU0) | 0 | **1,214** | 0 | 10 |
| rgca17 | `34:10.0` (gpu6) | **108** | **200** | 1 | 18 |
| rgca18 | — | 0 | 0 | 0 | 0 |

The idle-soak conclusion survives; the phrase "zero AER" must not be quoted from
this batch. cor04's 1,214 is also a data point in the correctable-versus-fatal
anti-correlation below.

**Wrapper v2 false positive found and fixed in v3.** All four RGCA runs in this
batch were verdict-gated `CLEAN BUT LINK DEGRADED` with a *uniform* ~35 s
below-gen5 dwell on **all eight GPUs** — that is pre-load idle at gen1 (RGCA links
idle at gen1, and switch-node prep takes ~35 s) plus the settle window's post-load
idle, both counted by v2's whole-trace dwell check. v3 computes the degradation
verdict from load-window samples only (`pcie_link_states_load.txt`; the full-trace
table keeps its old semantics). Validated against the archive: the x8 runs and the
265 s gen1 dwell still flag; the four false positives do not. **Uniform dwell
across all GPUs is the signature of the artifact — real degradation is per-link.**
The four v2 verdicts in this batch should be read as clean.

### up700s repeat 2 (wrapper v3's first batch) — rgca17's first fault

Three runs, 2026-08-20 ~02:49, `baseline-uncapped`, actual uptimes ~1500 s (the
tag says up700s; post-fault reboots shifted the starts — the manifest is
authoritative):

| node | uptime | pre-fault W | outcome |
|---|---|---|---|
| cor04 | 1499 s | 493 | **FAULT 60.7 s** — GPU0 `00:01.1`, ERR_FATAL, third GPU0 fault in a row |
| rgca17 | 1495 s | 402 | **FAULT 201.4 s** — **first rgca17 fault in 13 runs** |
| rgca18 | 1527 s | 398 | clean |

**rgca17's first fault came from exactly the slot term A identified**: `ERR_FATAL
received from 0000:36:00.0` = gpu6, the marginal `34:10.0` link that produced
every one of the node's historical correctables — contained at the shared root
port `1a:01.1`, all 8 GPUs to x0, full switch-node blast radius. Only 1 RxErr /
1 BadTLP preceded it: near-silent this time. Two standing claims retire:
"rgca17 never faults" is dead, and with it the strong reading of the slot term —
**all three nodes fault; susceptibility sets the rate and the victim slot, not
immunity.** Every fatal in the fleet has now come from a previously-identified
per-node slot (cor04 GPU0/GPU5, rgca17 gpu6, rgca18 gpu4).

Under every fitted model this cell had P(fault) ≈ 0.01–0.04 — the most surprising
single outcome of the campaign, hence the most informative. Note also both faults
landed in the same simultaneously-started batch, 2.5 min apart, at ~02:50 local:
ambient/time-of-day remains unmeasured (protocol item: BMC inlet temp is still
not being logged).

**The cor04 staircase after two repeats: 282.2 → 65.6 → 60.7 s** at matched
~494 W. Direction is wear-ward, but the refit (35 runs, 10 faults) still keeps
the wear MAP near zero — two short draws in a row have ~10% probability under
constant hazard, not yet a trend. The refit also softened the power slope
(bP back to +0.75, x2.1 per +50 W: rgca17's 402 W fault pulls it down) and
**narrowed slot-vs-no-slot to lnBF ≈ 1.5** — the slot term's strongest evidence
was rgca17's zero, now gone. slot x power remains the standing model; nothing
else earns a parameter. Updated predictive spreads keep the same test ranking:
the low-power cell (0.10 vs 0.62) and the staircase remain the two live probes.

### Bayesian model comparison (32 runs, `bayes_models.py`)

Constant-hazard survival models fitted to every usable run (7 fault events,
including rgca18's post-window fault at ~609 s; the rgca18 setup-error run is
excluded): `lambda = exp(a_node + bP*(W-450)/50 + bU*ln(up/1000) + bW*wear)`.
Flat priors over declared grids; marginal likelihoods, so every extra parameter
pays an automatic Occam penalty — this is the quantitative guard against
unfalsifiable factor-stacking: a factor stays in the model only if it buys more
likelihood than its prior spread costs.

Updated after the idle/re-run batch (32 runs, 8 fault events):

| model | lnBF vs best | MAP |
|---|---|---|
| slot x power | 0.00 | hazard **x2.7 per +50 W** |
| slot x power x soak | −0.93 | bU = 0 |
| slot x power x wear | −1.23 | bW = 0 |
| full (P+U+W) | −2.15 | |
| slot only | −3.23 | |
| slot x soak | −3.74 | |
| slot x wear | −4.45 | |
| no slot term at all | **−6.55** | |

The new batch *strengthened* slot x power (bP MAP moved 0.75 → 1.00; power is now
worth ~25x over slot-only) and pushed both soak and wear MAPs to **zero** — the
1 h-idle clean plus the 700 s-uptime fault is anti-soak as a smooth effect, and
the 1 h-idle clean at wear=5 is anti-wear as a smooth effect. Two things survive
outside the fit: the h4 controlled pair still admits a **threshold** soak in
(4082 s, 33347 s] — the model family only contained smooth log-uptime, so a
threshold was never tested — and the 65.6 s re-run is consistent with wear as a
trend that needs the staircase (below) to confirm. The plain-language summary of
32 runs: **slot x power with large (exponential) TTF dispersion explains
everything measured so far; every additional factor is currently decoration.**

Posterior-predictive P(fault in 600 s) for feasible cells, per model — the spread
across models is the expected information of running that cell:

| cell | slot | +P | +U | +W | +P+U | +P+W | spread |
|---|---|---|---|---|---|---|---|
| cor04 soak-1h pl450 (in flight) | .51 | .47 | .59 | .45 | .52 | .63 | 0.18 |
| rgca17 soak-1h | .02 | .01 | .02 | .02 | .01 | .01 | ~0 |
| **cor04 baseline-uncapped re-run @ ~700 s uptime** | .51 | .77 | .38 | .45 | .66 | .83 | **0.45** |
| **cor04 low-power arm (~290 W) @ 1 h uptime** | .51 | .09 | .59 | .45 | .13 | .16 | **0.50** |
| cor04 deep-soak repeat (~9 h) | .51 | .47 | .83 | .45 | .69 | .63 | 0.38 |

Reading: the **in-flight soak-1h run is the least discriminating cor04 cell**
(all models roughly agree it faults ~half the time) — its value is mechanism-side
(h15/h16), not model separation. The two highest-information cheap runs are the
**baseline-uncapped re-run at matched ~700 s uptime** (soak says 0.38, power+wear
says 0.83, and TTF adds resolution: soak predicts ≈282 s, wear predicts far less)
and the **low-power arm at 1 h uptime** (power models say ~0.1, soak says 0.6).
The h15/h16 2x2 is invisible to all of these models — retrain state and thermal
history are constant across the whole corpus, so their posteriors are flat and
each off-diagonal cell is worth close to a full bit. Budget note: at these
predicted probabilities one run moves lnBF by ~0.5–0.8, so resolving soak-vs-wear
to "strong" (lnBF ≈ 3) costs **4–6 targeted runs**, consistent with the
sample-size section. Caveats: constant-hazard within a run is an approximation,
power is measured rather than assigned, and lnBF differences under ~1 are noise
at these grid resolutions.

#### Mechanism candidates consistent with h13, ranked by how many confirmed facts each also tracks

1. **Stale Gen5 equalization drifting on the GPU side** — EQ coefficients are set
   at boot training and typically reused across idle↔load speed changes; the GPU
   die swings ~60 °C between idle and load while the root port barely moves, so
   the GPU TX drifts furthest from its coefficients. Tracks h13 + h15 + h10 (per-
   boot error variance *is* per-boot EQ outcome) + h4. Discriminator already
   queued: retrain-before-warm-run.
2. **GPU-local PHY supply disturbance from the card's own load transients** — the
   TX driver runs off a card-local rail; the host TX does not care about the
   card's 16 A steps. Tracks h13 + h2, explains SDES-with-no-warning, compatible
   with h12's refutation (amplitude-driven, not rate-driven). Does not explain
   soak alone.
3. **Connector/riser contact degradation on the card-TX pin group** (fretting,
   driven by thermal-cycle micro-motion) — in the CEM connector the card's TX and
   RX pairs occupy distinct pin groups, so a systematic mechanical bias degrades
   one direction fleet-wide. Tracks h13 + h14 + h16 + h7 (different connector
   systems per topology class). Testable by reseat/swap with Lane Error Status.
4. **GPU PHY junction heat as a term-B ingredient** — the GPU is the hottest
   transmitter in the system. Tracks h13 + h2; excluded as sole cause by the 9 h-
   idle fault and the 438/440 W pair.

**Downgraded by h13:** the symmetric ground-shift story (`edge_bounds.py`) — a
common-mode shift between GPU and root port displaces both directions about
equally and both are AC-coupled, so it predicts roughly symmetric errors, not
700k-to-zero.

### Corpus audit for further false negatives

All 21 post-freeze runs were audited by scanning each boot session's *last* run's
`dmesg_before`, which spans the whole boot including the gaps between runs, for
containment and endpoint-loss signatures outside any run's own capture window.

| boot session | covered | out-of-window fatals |
|---|---|---|
| rgca18 `23:07` | pl450, pl525, lgc2100 + gaps | none (and zero correctables all boot) |
| rgca17 `23:07` | pl450, pl525, lgc2100 + gaps | none; all correctables fell inside run windows |
| cor04 `23:13` | pl450, pl525 + gaps | none |
| cor04 `00:12` | — | none |
| cor04 `01:28` | pl450-cold + 9 h gap | none — that run was genuinely clean |
| rgca17 `01:28` | pl450-cold + 9 h gap | none; zero fatal *and* zero correctable across 9.3 h |
| rgca18 `01:28` | pl450-cold | **fault #5** — the one false negative in the corpus |

So **exactly one** run in the corpus was misreported. Three runs remain
unauditable: the last run on a boot that was followed by a reboot leaves its tail
beyond `dmesg_after` unrecoverable from `dmesg` (rgca17 and rgca18
`baseline-uncapped`, and rgca17 `pl575-8192`). If the journal is persistent on
these hosts, `journalctl --list-boots` and `journalctl -b -N` can still recover
those tails; otherwise treat those three `clean` verdicts as bounded only up to
their snapshot.

No errors were found in any idle gap between runs, on any node — including
rgca17's 9.3 h idle stretch. Correctable errors only ever accrue under load.

### Correction: the release edge is not a general driver

The load-release observation is scoped to rgca18 fault #5 and does not generalise.
Binning every correctable kernel line by its position relative to load end:

| run | lines | in first 90% | final 10% | after load end |
|---|---|---|---|---|
| rgca17 uncapped | 146 | 146 | 0 | 0 |
| rgca17 pl525 | 9 | 0 | 9 | 0 |
| rgca17 lgc2100 | 23 | 2 | 11 | 10 |
| rgca17 pl575-8192 | 41 | 41 | 0 | 0 |
| rgca18 pl575-8192 | 191 | 188 | 3 | 0 |
| rgca18 pl450-8192 | 616 | 487 | 118 | 11 |
| cor04 pl450-warm | 80 | 80 | 0 | 0 |

Two low-count rgca17 runs put essentially all their errors in the tail, which is
not chance at those counts, but the high-count runs put them mid-load. And **four
of the six fatals — every cor04 fault — happened mid-load with no release
involved.** Hypothesis 9 therefore rests on fault #5's timing plus fault #4's
coherent-peak coincidence, and is not yet supported as a general mechanism.

### Two harness gaps this exposed

- **`dmesg_after` is snapshotted too early.** A fault during or just after
  teardown lands outside the capture window and the run reports `clean` with
  `exit_code 0`. This produced fault #5 as a silent false negative. A settle
  delay of ≥30 s after teardown, or a second snapshot, is required — and every
  existing `clean` verdict should be re-audited against the *following* run's
  `dmesg_before`.
- **Degraded link width does not affect the verdict.** Two runs completed at
  `gen5 x8` on a known-marginal port and were reported `clean`. Width and gen
  should gate the verdict, or at minimum print a loud warning; the logic already
  exists in `linkcheck.sh`.

---

## Full-corpus re-read (all 38 runs, from raw artifacts)

Every bundle in `runs/` parsed directly from `manifest.txt`, `events.csv`,
`nvml_trace.csv`, `aer_delta.txt`, `aer_uncorrectable.csv`,
`pcie_link_states*.txt`, `pcie_link_rootports.csv`, `psu_summary.txt` and every
`dmesg` snapshot — not through the existing summary scripts. Corpus totals:
38 bundles, 3 nodes, 13 batches, 10 containments, 18,197 s of load exposure.

### Three tooling defects that changed the numbers

Found while reproducing the earlier correlation pass; all three are now fixed in
`corr_matrix.py`.

1. **Its directory regex hard-coded `933b21`**, so it silently skipped the nine
   newest runs (the `538076` and `911fc2` batches). The "27 runs" and "26 runs"
   figures quoted above are that subset, not the corpus.
2. **Its fixed t=60–93 s window blends teardown idle into any run that faulted
   before 93 s** — now three of them. This is why the two `up700s` cor04 runs
   read 184 W and 120 W on a naive re-run. Under a window that always ends ≥5 s
   before the fault they read **494 W and 495 W**, which is what the
   `baseline-uncapped` comparison actually rests on.
3. **It summed AER rows without deduping by `(role, bdf)`.** On RGCA nodes the
   shared root port `1a:01.1` is listed once per GPU, so a single `NonFatalErr`
   was counted eight times.

### Corpus integrity: two runs are not what the corpus records

**`20260819T203844Z-933b21-ad0a69-cptrgca18-…-gpc1` — the host died mid-run.**
Six live-written files end in NUL padding to an exact page boundary
(`events.csv`, `nvml_trace.csv`, `psu_current.csv`, `pcie_dmon.txt`,
`pcieburn.log`, `aer_counters.csv`) — a filesystem that recorded inode sizes and
never flushed the data. **Every post-run artifact is absent**: no
`aer_delta.txt`, `dmesg_after.txt`, `dmesg_delta.txt`, `faults.txt`,
`pcie_link_states*.txt`, `psu_summary.txt`. The manifest has no `end_utc`, no
`exit_code` and no `verdict`. Last flushed telemetry is t=+589.3 s of load; its
byte-identical twin on the same node (20:14, same arm, same binary) ran 643.5 s,
so it died between t≈589 s and load end. Both sibling runs in that batch (cor04,
rgca17) completed normally, so this was not a rack-level power event.

This is the **second false negative in the corpus and a worse one than fault 5**,
which at least left a `dmesg` tail on the next run's snapshot. rgca18's next
bundle boots at 22:41:13, so the 20:48 kernel tail is unrecoverable from the
artifact set. `journalctl --list-boots` then `journalctl -b -N` on rgca18 is the
only remaining route and is worth spending: a third rgca18 event near 600 s of
load would move the timing pattern below from three points to a signature.

**The rgca18 "setup/usage error" run is fault 5's 9-hour aftermath, not an
excluded null.** Its log says `no CUDA devices found (probe returned -1)`, but
the manifest's own `nvidia-smi -L` snapshot carries **eight** lines of `Unable to
determine the device handle for gpu <bdf>: Unknown Error`, and its
`dmesg_before` carries the 01:42:16 containment, `Xid 79` on all eight GPUs and
`AER: device recovery failed`. It is the **same boot** as the run that took
fault 5. So it is positive evidence that a switch-node containment removes all
eight GPUs from the OS for the remainder of the boot — here 9 h 15 m — and it is
where the `0x1 (GPU Reset Required)` trap above was found. The Bayesian fit
should stop excluding it as a setup error and start counting it as post-fault
state; rgca18's wear should also be 2, not 1, since fault 5 is currently
uncounted.

### Provenance holds

One driver (`580.178.04`) on all 38 runs, one binary hash per node throughout
(`692bd6` / `fd4431` / `ad0a69`), and all 13 batches launched on a single commit.
One covariate is **not** clean and is not currently a recorded run variable — see
the kernel confound below.

### h13 resolved on fleet hardware: the zero is not a mask artifact

The ledger held h13 pending a `CEMsk` read on a fleet GPU. The kernel already
printed the register, every time a GPU endpoint logged. Correctable Error mask
bit 0 is RxErr, bit 6 BadTLP, bit 7 BadDLLP, bit 12 Replay Timer Timeout:

| device | slot | status | mask | masked |
|---|---|---|---|---|
| `nvidia 0000:01:00.0` | cor04 GPU0 | `0x00001100` | `0x00001000` | ReplayTimerTimeout only |
| `nvidia 0000:2d:00.0` | rgca18 gpu4 | `0x00001100` | `0x00001000` | ReplayTimerTimeout only |
| `nvidia 0000:36:00.0` | rgca17 gpu6 | `0x00001100` | `0x00001000` | ReplayTimerTimeout only |
| `pcieport 0000:2b:10.0` | rgca18 leaf | `0x000000c1` | `0x00001000` | ReplayTimerTimeout only |
| `pcieport 0000:34:10.0` | rgca17 leaf | `0x000010c1` | `0x00001000` | ReplayTimerTimeout only |
| `pcieport 0000:00:01.1` | cor04 GPU0 root | `0x00000040` | `0x00001000` | ReplayTimerTimeout only |

This is a **stronger** argument than the counter totals it replaces. The status
register latches regardless of the mask, so the GPU rows are not merely
un-incremented counters: the GPU's own hardware recorded bits 8 and 12 set and
bits 0, 6 and 7 **clear** at the moment it reported, while its upstream receiver
recorded exactly the opposite. RxErr and BadTLP are unmasked on all three
implicated GPU endpoints, on fleet hardware, driver 580, OS-first. The pending
`lspci` check is no longer a prerequisite for h13.

Corpus-wide, deduped by port: **697,979 RxErr / 94,989 BadTLP / 6,275 BadDLLP**
on upstream receivers against **zero of all three** on GPU endpoints, which
logged 988 Rollover — their own replay timer, i.e. them retransmitting. Only
four ports out of forty have ever logged a correctable error, and cor04
`a0:01.1` remains at zero across all 38 runs despite three fatals.

One reading caution: the `a0:01.1` line showing `mask=0x00000000` belongs to an
**Uncorrectable** error block, so it is the uncorrectable mask register, not a
correctable-mask anomaly. Nothing masked there either, which is why the
classifier below works.

### `aer_uncorrectable.csv` is a working classifier, not a blind spot

The analysis-discipline entry saying this file cannot detect the thing is wrong.
It separates the two fatal classes perfectly, with no false positives across the
29 non-fatal runs:

| class | runs | data rows each | content |
|---|---|---|---|
| reason `00` — SDES at the port | 3 | **70** | `a0:01.1 rootport fatal SDES` + `TOTAL_ERR_FATAL`, latched ~1.5 s after DPC |
| reason `10` — `ERR_FATAL` received | 6 | 0 | header only |
| clean | 29 | 0 | header only |

Fisher exact on 3/3 versus 0/6 gives p = 0.012. The emptiness is diagnostic, not
a sampling failure: in the `ERR_FATAL` class the root port never records an
uncorrectable error *of its own*, so there is nothing to poll. A non-empty file
is a positive identification of the silent GPU5 mode — the only mode that leaves
no other trace.

### Precursors exist for one class only, and the lead time is seconds

Port-matched: for each containment, were there correctable log lines on *the same
port* beforehand?

| # | node | port | same-port lines | first | last | largest interior gap |
|---|---|---|---|---|---|---|
| 1 | cor04 | `a0:01.1` | **0** | — | — | — |
| 3 | cor04 | `a0:01.1` | **0** | — | — | — |
| 7 | cor04 | `a0:01.1` | **0** | — | — | — |
| 2 | cor04 | `00:01.1` | 1 | 0 s | 0 s | — |
| 8 | cor04 | `00:01.1` | 8 | 4 s | 0 s | 4 s |
| 9 | cor04 | `00:01.1` | 15 | 4 s | 0 s | 3 s |
| 10 | rgca17 | `1a:01.1` | 14 | 0 s | 0 s | — |
| 6 | cor04 | `00:01.1` | 169 | 103 s | 0 s | 31 s |
| 4 | rgca18 | `1a:01.1` | 199 | 201 s | 0 s | **102 s** |

Three consequences. **The silent class is unmonitorable** — `a0:01.1` has logged
zero correctables in all 38 runs and produced three fatals from perfect silence,
so no AER-rate alarm can ever see it; its only positive signal is the
`aer_uncorrectable.csv` latch, after the fact. **The warning class warns late** —
all six `ERR_FATAL` events have same-port correctables ending in the *same
second* as the containment, and in three of six the entire precursor sequence is
4 s long. Detection is feasible; prevention is not. **The quiet-gap lesson
sharpens** — fault 4 went 102 s with nothing on the root port before resuming and
dying in the same second, so the earlier "80 s" figure should read 102 s on
`1a:01.1`.

### Correctable volume and fatal hazard are anti-correlated on the same slot

Holding node, cap, matrix dimension and arm constant on cor04's warning slot:

| run | when | uptime | prior fatals | mean W | BadTLP on `00:01.1` | outcome |
|---|---|---|---|---|---|---|
| pl450-8192 cold | 08-19 01:31 | 216 s | 3 | 438 | 32,275 | clean |
| pl450-warm | 08-19 10:43 | 33,347 s | 3 | 440 | 3,954 | **fault 224 s** |
| pl450-8192 up1h | 08-19 23:49 | 4,082 s | 5 | 437 | 1,214 | clean |

The first two are h4's controlled pair — same boot, same cap, same arm, power
matched to 2 W. The new observation is that the correctable rate fell **8×**
across that pair *and* the fatal appeared; by the third run it is 27× below the
first. Over the same interval cor04's time-to-fault on the matched uncapped arm
fell 4.6×. This upgrades "correctables and escalation are separate processes"
from an absence of correlation to a **measured anti-correlation**, and is a
second independent reason not to build an alarm on AER rate.

### rgca18's three terminal events all landed within 20 s of 600 s of load

Read together with the truncated run above, rgca18 has had exactly three
terminal events:

| event | t of load | nominal load end | note |
|---|---|---|---|
| host crash, 08-19 20:38 | ≥ 589.3 s | ~641 s (gpc1 arm) | bound is the last flushed page, so the true time is later |
| fault 4 | 598.1 s | 600 s | 1.9 s before load end |
| fault 5 | 608.8 s | 600 s | 8.8 s after load end |

Under a uniform hazard across a 600 s run, one event landing in a 20 s window at
the end has probability 0.033; all three has probability **3.7 × 10⁻⁵**. For
contrast, cor04's seven faults sit at 10, 11, 16, 18, 36, 37 and 47% of their
load phases, and rgca17's single fault at 34%.

Recognising 598.1 s as the load boundary unifies faults 4 and 5 under one timing
pattern rather than treating one as a coherent-current-peak coincidence and the
other as a release event. But the more important consequence is a design defect:
**every rgca18 run in the corpus is censored at almost exactly the point where
its hazard concentrates**, so its eight `clean` verdicts carry far less
information than their count suggests.

*Discriminating test.* "Release edge" and "~600 s of cumulative exposure" are
indistinguishable while every run is 600 s. One **1200 s** run on rgca18
separates them: a fault near 600 s means cumulative exposure with a knee at ten
minutes; near 1200 s means the release transition; no fault falsifies both. This
is cheaper and cleaner than h9's N × 100 s design, which moves teardown count and
per-run exposure together. It also tests a competing explanation worth keeping in
view — that the wrapper's own post-load config-space sweep across ~17 devices is
the provocation.

### The RGCA fabric is a three-tier cascade; both victims sit on its deepest branch

`pcie_link_rootports.csv` is byte-identical on rgca17 and rgca18 and shows far
more structure than "all eight GPUs behind one root port": three switch tiers,
with three GPUs 3 hops from the root port and four at 5 hops behind a third-tier
switch.

| GPU endpoint | hops | path from root port to leaf downstream port |
|---|---|---|
| `1f:00.0` | 3 | `1a:01.1 → 1c:00.0 → 1e:00.0` |
| `20:00.0` | 3 | `1a:01.1 → 1c:00.0 → 1e:10.0` |
| `23:00.0` | 3 | `1a:01.1 → 1c:04.0 → 22:00.0` |
| `3b:00.0` | 3 | `1a:01.1 → 1c:0c.0 → 27:10.0` |
| `2c:00.0` | 5 | `1a:01.1 → 1c:0c.0 → 27:00.0 → 29:00.0 → 2b:00.0` |
| **`2d:00.0`** | 5 | `1a:01.1 → 1c:0c.0 → 27:00.0 → 29:00.0 → 2b:10.0` — **rgca18 victim** |
| `30:00.0` | 5 | `1a:01.1 → 1c:0c.0 → 27:00.0 → 29:04.0 → 2f:00.0` |
| **`36:00.0`** | 5 | `1a:01.1 → 1c:0c.0 → 27:00.0 → 29:0c.0 → 34:10.0` — **rgca17 victim** |

Both susceptible slots are 5 hops deep, behind third-tier switch `29`, on a leaf
downstream port ending `:10.0` — a set with exactly two members of eight, so the
null probability of both landing there is (2/8)² = 0.06. **The feature was
identified after seeing which slots failed; treat that number as a prompt, not a
test.**

The mechanism it does *not* support matters as much. No intermediate hop (`1c`,
`27`, `29`) ever logged an error in 38 runs, and every count sits on the final
GPU↔leaf link, so jitter is not accumulating down the cascade — each hop is an
independently retimed link. Depth is therefore a proxy for **physical placement
on the switchboard** (trace length, connector position, airflow), which is the
kind of physical referent term A has been missing. It also revises the h6 note
that these are "different physical slots on identical boards": they are
**homologous branch positions**, which turns the swap test into a matched
within-branch comparison available on *both* nodes.

### The susceptible slot is never the node's thermal extreme

Mean temperature rank per slot across every run with all eight GPUs in the
matched window, rank 1 = hottest. Victim *power* rank was already known to be
non-predictive; temperature and thermal swing had not been checked, and they are
the natural candidates for a fretting or PHY-heat story.

| node | hottest three slots (mean T-rank) | susceptible slot | its T-rank | its ΔT-rank | coldest |
|---|---|---|---|---|---|
| cor04 | gpu5 (1.31), gpu1 (2.38), gpu4 (2.92) | gpu5 · gpu0 | 1.31 · 4.00 | 1 · 7 | gpu7 (7.85) |
| rgca17 | gpu1 (1.64), gpu3 (1.91), gpu6 (3.36) | gpu6 | 3.36 | 1 | gpu4 (6.73) |
| rgca18 | gpu1 (1.45), gpu0 (1.55), gpu4 (3.00) | gpu4 | 3.00 | 2 | gpu7 (7.82) |

cor04's gpu5 *is* the hottest slot in 12 of 13 runs — a genuine, stable per-slot
asymmetry worth recording. But the second- and third-hottest slots on that node
have never failed; cor04 gpu0 has failed four times from rank 4.00 with the
second-**lowest** thermal swing on the node; and both RGCA victims sit mid-pack
behind two hotter slots that have never produced a single error. Victim
temperature ranks across the nine in-window faults are 1, 5, 1, 3, 4, 5, 4, 3, 1
— mean 2.9 against a null of 4.5, entirely carried by cor04 gpu5.

**Junction temperature and thermal cycle depth are therefore excluded as the
slot-selection term.** This does not touch h16's claim about cycle count driving
degradation over time; it removes the version of h16 that would predict *which*
slot degrades, and it removes ranked mechanism 4 ("GPU PHY junction heat as a
term-B ingredient") from the slot-selection role.

### PCIe payload is exonerated harder than the "2.9% of Gen5 x16" figure implies

h5's discriminating test is "128M vs 4G at fixed compute; if TTF is unchanged,
PCIe traffic is exonerated outright." Computed per run from
`coll_bytes_pcie_link` in `events.csv`:

| arm | node | GB/s per rank | % Gen5 x16 | mean W | outcome |
|---|---|---|---|---|---|
| 8192 gpc1 128M–1G | cor04 | **37.16** | **59.0** | 287 | clean, 0 AER |
| 8192 gpc1 128M–1G | cor04 | **37.10** | **58.9** | 286 | clean, 0 AER |
| 8192 gpc1 128M–1G | rgca17/18 | 9.26–9.29 | 14.7 | 148–152 | clean ×4 |
| 2048 uncapped | cor04 | 4.30–4.60 | 6.8–7.3 | 494–495 | fatal ×3 |
| 8192 single full burst | cor04 | 0.92–1.01 | 1.5–1.6 | 440–503 | fatal ×2 |
| 8192 gpc1 coll4M | cor04 | 0.81 | 1.3 | 545 | fatal, 95 s |

cor04 — the node that faults once per 667 s of load — ran **59% of Gen5 x16 for
606 and 607 s with zero AER counts and no fault, twice**. Across 37 runs
r(egress, fault) = **−0.18**; faulting runs span 0.81–4.60 GB/s per rank, clean
runs 0.83–37.16.

Honest limit: the high-traffic arms are also the low-power ones, so this shows
traffic cannot *substitute* for power, not that traffic is irrelevant at fixed
power. Closing h5 outright needs an arm holding ~495 W **and** ~37 GB/s, which
the current knobs cannot express — raising collective share lowers duty cycle and
therefore power. That is a harness design problem, not another run.

### Ambient is measurable retroactively, and is not the mediator of anything

Every bundle contains ~30 s of pre-load NVML at idle; the coldest idle GPU
reading is a usable proxy for thermal state at load start. Across 36 runs it
spans 19–29 °C and is mostly a per-node offset, not a diurnal signal.

| statistic | value | reading |
|---|---|---|
| r(ambient, fault) | +0.06 | n = 36; no association |
| r(ambient, ln TTF) over faults | −0.27 | n = 9; weak, wrong magnitude |
| mean, faulting runs | 21.3 °C | 0.2 °C apart |
| mean, clean runs | 21.1 °C | |
| the two 02:50 faults | 22 / 20 °C | time-of-day not implicated |
| across the power-matched staircase | 20 → 21 → 22 °C | 2 °C total, against a 4.6× TTF change |

That last row removes ambient from the covariates moving together across the
cor04 staircase, leaving two rather than three. The `ipmitool sdr` inlet read is
still worth adding for the record but is no longer blocking.

### The staircase is airtight on the load side, and newly confounded by a kernel upgrade

| run | when | mean W | mean clk | clk sd | p98 clk | ambient | uptime | prior fatals | kernel | TTF |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 08-18 22:50 | 495 | 2412 | 141 | 2692 | 20 °C | 697 s | 0 | `7.0.0-28` | 282.2 s |
| 2 | 08-20 02:01 | 494 | 2412 | 136 | 2692 | 21 °C | 812 s | 5 | `7.0.0-29` | 65.6 s |
| 3 | 08-20 02:49 | 495 | 2410 | 136 | 2692 | 22 °C | 1,499 s | 6 | `7.0.0-29` | 60.7 s |

Matched to 1 W of mean power, 2 MHz of mean clock, 5 MHz of clock standard
deviation, 0 MHz at p98 and 2 °C of ambient — and time-to-fault fell 4.6×.
Within-boot uptime is **eliminated** as the mediator: it rose across the three
while TTF fell, and pooled across all nine faults r(ln uptime, ln TTF) = **+0.35**,
the wrong sign for soak. Ambient is eliminated above. Over the fault set
r(prior fatals, ln TTF) = **−0.86** pooled and **−0.88** within cor04 — the
strongest association in the corpus — against r(power, ln TTF) = −0.21.

**But cor04's kernel went `7.0.0-28` → `7.0.0-29` between the pl450-warm run and
the first gpc1 run, while both RGCA nodes stayed on `-28` throughout, and kernel
version is not currently a recorded run variable.** Every long-TTF cor04 fault
(282, 214, 107, 224 s; mean 207) is on `-28` and every short one (95, 66, 61 s;
mean 74) is on `-29` — a *perfect* separator, and a cleaner one than prior-fault
count, which has the 224 s pl450-warm fault out of order. Mitigating evidence:
diffing the two boots' PCIe lines gives identical `_OSC` (26), `DPC` (44) and
`ASPM` (8) line counts and an identical command line apart from the image name,
so error-handling configuration did not change — but the NVIDIA kernel module was
rebuilt.

*Cheapest decisive run in the plan.* Boot cor04 on `7.0.0-28` and repeat
`baseline-uncapped` at ~700 s uptime. TTF back near 200–280 s implicates the
kernel and dissolves the wear staircase; TTF still near 60 s removes the kernel
and leaves wear versus campaign-time. One reboot, no code, more decisive than the
4–6 wear runs currently budgeted.

Statistical honesty on the staircase itself: under a constant hazard with mean
282 s, two consecutive draws both ≤ 66 s have probability 0.040. Suggestive, not
settled — and these are descriptive correlations over fault events. The censored
survival fit remains the authority on hazard, and should be re-run with a kernel
term, with the two reclassified runs above, and with rgca18's wear corrected.

### Per-node hazard, with exposure properly accounted

| node | runs | load exposure | containments | rate | 
|---|---|---|---|---|
| cor04 | 13 | 4,666 s | 7 | 1 per **667 s** |
| rgca18 | 10 | 6,041 s | 2 | 1 per 3,021 s |
| rgca17 | 13 | 7,490 s | 1 | 1 per 7,490 s |

cor04 7/13 runs fatal against rgca17 1/13 gives Fisher one-sided **p = 0.015**,
so h6 is now statistically supported rather than only descriptive; on exposure the
rate ratio is 11.2×. **No run in the corpus exceeded 642 s of load**, so every
`clean` verdict in the entire corpus is bounded at roughly one arm-length.

### Confirmed unchanged

- **Correctables accrue only under load.** 696 distinct (node, second, port)
  correctable events across every `dmesg` snapshot in the corpus; every one falls
  inside a load window, including across rgca17's 9.3 h idle stretch.
- **Cap clipping fails as a stress metric.** r(clip%, AER) = +0.18,
  r(clip%, fault) = −0.04 over 37 runs.
- **PSU envelope is not predictive.** r(peak instrumented pair, fault) = +0.27,
  r(max swing, fault) = +0.21. The pl450-warm fatal had the *lowest* current
  swing of any 8192 run (55.8 A), which also sits against an amplitude-driven
  reading of h9.
- **Per-boot training dominates rgca17's error counts (h10).** Its gpu6 logged
  9,672 counts on the boot where it dwelt 265 s at gen1 and exactly zero on both
  runs where it trained to gen5 **x8**. Note that only the three `911fc2` runs
  carry `pcie_link_states_load.txt`, so every earlier link-state reading is
  whole-trace and includes the uniform pre-load gen1 artifact — the x8 states are
  real at 6,070 and 6,058 samples; the uniform ~35 s dwells are not.
- **h3 is weaker than "actively harmful, 2/2 nodes".** The pin worked —
  `lgc2100` cut cor04's SM clock sd from 141 to 28 MHz and rgca17's from 223 to
  24 — and cut power from 495 to 395 W. But the cor04 comparison is 282.2 s at
  0 prior fatals against 214.5 s at 1 prior fatal: a 24% TTF change across
  exactly one increment of the covariate the staircase shows is worth 4.6× over
  five. The rgca17 leg is a correctable-count difference on a node whose counts
  are not comparable across boots. Read as "no benefit observed, confounded with
  wear"; the `-lgc` near-boost arm is still the right test.

---


## Third platform: intelmgx-CG480 (8x RTX PRO 6000 Blackwell Server)

**Never pool this box's numbers with cor04/RGCA data.** It differs on every axis
at once: same GB202 die but the PRO SKU (more SMs, 600 W default), dual-socket
Intel MGX host, **four PCIe switches with two GPUs each** (PIX pairs 0/1, 2/3,
4/5, 6/7 across two NUMA domains — a third topology class, blast radius 2),
**P2P enabled on all pairs**, driver 595.91.07 / CUDA 13.2 vs the fleet's 580,
and — critically — **firmware-first error handling**: every host bridge reports
`_OSC: platform does not support [AER DPC]`, GHES/APEI is on, no port has
kernel-managed DPC, and recovery would go through EDR.

**Observability consequence:** `aer_delta.txt` is structurally blind here — the
OS never receives native AER events, so zero counters mean nothing. The
observables on this box are kernel GHES/CPER (`Hardware Error` lines, rasdaemon)
and the **BMC SEL** (`ipmitool sel list` bracketed around every run). This is
also the native configuration for the fleet's original PERR/SERR symptom — the
BMC sees errors first by design — so any event here exercises the exact
misinterpretation chain the investigation started from. Its power telemetry is
healthy where the fleet's is broken (`PSU*_POWER_OUT` unsaturated at >1 kW,
`PSU*_IOUT`, and an ambient sensor `PSU1_AMB_TEMP`), and the driver exposes
`temperature.gpu.tlimit` plus `clocks_event_reasons_counters.*` — cumulative
seconds under each throttle cause, a rigorous replacement for the clip% proxy.

Baseline state found on arrival: links uniformly clean (gen5 x16, ASPM disabled,
no training anomalies), endpoint `CEMsk` all clear (spec default — nothing
masked, see h13), a leftover 430 W power limit from a prior tenant (reset and
record), and a prior workload that ended before our runs.

**BLOCKED — box is compromised (found 2026-08-20).** nvtop revealed seven
root-owned cryptominer processes (masquerading as `systemd-update-helper --coin
pearl -o hk.pearl...`, ~2122 MiB and 15–29% on every GPU) that are hidden from
`nvidia-smi --query-compute-apps` — i.e. NVML process enumeration is hooked. The
430 W cap and 429 W draw noted on arrival were this miner, not a prior tenant. All
pre-registered tests below are on hold until the box is remediated (preferably
reimaged, since a root rootkit makes any on-box reading — including a "clean" idle
— untrustworthy) and re-characterized from scratch. No pcieburn data has been
taken on this platform; none should be until then.

**Pre-registered tests and predictions** (value is in falsifying die-level
explanations; this platform has none of our marginal slots):

1. Calibration nulls — the two frozen arms (2048 mixed uncapped; 8192-single at
   600 W), 600 s. *Predict: no fault, no SEL entry, no CPER.* Any GPU→upstream
   correctable surfacing via SEL/CPER on clean pro hardware would promote h13
   from "our boards" to GB202-generic; a fault would be larger news.
2. P2P A/B — same arm with `NCCL_P2P_DISABLE=1` (fleet-matched host staging)
   vs `0` (peer traffic), plus a `--gpus 0,1` PIX-pair cell where collective
   traffic never leaves one switch. *Predict: no behavioral difference in fault
   terms (bandwidth is exonerated); byte-accounting lines invalid under P2P
   (`PCIE_HOST_STAGING_MULT` assumes staging — read as relative only).* With P2P
   off, each switch uplink carries two GPUs' staged traffic — the highest
   per-root-port load of any configuration yet run; the exoneration data says
   it still does nothing.
3. Power ladder 300/450/600 W with the T.Limit + throttle-counter sidecar.
   Deliverable: edge-vs-worst-internal-sensor delta on GB202 under our GEMM
   load, and exact seconds under `sw_power_cap`/`hw_power_brake`.
4. Boot-training loop: ~8 reboots, `linkcheck.sh --csv` each boot pre-load.
   *Predict (h10): zero training variance on this platform; variance is a
   fleet BIOS/riser property, not GB202/driver.*
5. `dcgmi diag -r 3` as the original-reproducer cross-check.

## Tenant workload profile: hivenet's vLLM serving

**Read the role of this section carefully.** The profile below was supplied by the
head engineer for the **hivenet** product — and hivenet recorded **zero servers
down across 28 days**. Meanwhile every node in the fault corpus (`cptcor04`,
`cptrgca17`, `cptrgca18`) is `tenant: fal`, whose workload is unknown to us and
deliberately stays that way: the platform has to be reliable regardless of
workload type, so this investigation does not chase external tenants' internals.

So this is **the profile of the tenant that is not failing.** An arm shaped like it
is a *negative control*, not a target:

- if a hivenet-shaped arm does **not** reproduce while the adversarial 8192 arm
  does, that is consistent with hivenet's clean record and tells us the fault
  needs something this profile lacks;
- if it **does** reproduce, then hivenet's zero downtime must come from somewhere
  else — perfVM/IOMMU, different hardware, or monitoring blindness (see
  hypothesis 11) — because the workload alone would suffice.

Either outcome is informative, which is why it is worth running. What it is *not*
is a reason to re-centre the harness on this shape. The deliberately adversarial
arms stay primary, because `fal` could be running anything and the platform must
survive it.

The numbers, for all five models hivenet serves. Megatron tensor
parallelism all-reduces the activation tensor twice per layer (post-attention,
post-MLP), per token; the message is `tokens_in_step x hidden x 2 B` and is
independent of TP degree. `vllm_shape.py` in this repo reproduces the table.

| model | TP | all-reduces/step | decode msg | decode GEMM sq-equiv | prefill msg | prefill GEMM sq-equiv |
|---|---|---|---|---|---|---|
| gemma-4-26b | 2 | 60 | 352 KiB | 633 | 88 MiB | 4020 |
| gemma-4-31b | 2 | 120 | 672 KiB | 974 | 168 MiB | 6186 |
| qwen3.6-27b | 2 | 128 | 640 KiB | 943 | 160 MiB | 5988 |
| qwen3.6-35b | 2 | 80 | 256 KiB | 512 | 64 MiB | 3251 |
| gpt-oss-20b | 1 | **0** | — | — | — | — |

Batch size is bimodal, not a single number: decode runs ~1 token per live
sequence (`max-num-seqs=64`), prefill fills toward `max-num-batched-tokens=16384`.
"GEMM sq-equiv" is the N of an NxN pcieburn GEMM with the same FLOP count as the
real `[tokens, hidden] x [hidden, hidden/TP]` projection.

**Three things this settles.**

*Our collective size is accidentally right for prefill.* `--coll-min 128M` was
chosen from `nccl-tests` convention, and real prefill all-reduces are 64–168 MiB.
The 8192 arm is a prefill-shaped workload in message size.

*But `--matrix-dim 8192` overshoots.* Real prefill GEMMs are 3251–6186
square-equivalent, so **4096 is the production-representative dimension** and 8192
is at or beyond the largest real prefill GEMM. The DCGM default of 2048 sits
between decode (512–974) and prefill.

*The real mismatch is synchronisation rate, not size.* Measured in the 8192 arm:
**94 GEMMs between consecutive collectives**. Production does **2–3** (QKV +
output projection, then up/gate + down). We are therefore 30–47x too infrequent in
how often the ranks are forced into lockstep — and lockstep is the axis hypothesis
9 is about. Decode is worse still: at 20–50 steps/s the bigger models run
2,400–12,800 all-reduces/s and move 0.42–4.19 GB/s egress/rank, against our
measured 0.50 GB/s.

**Control arms this makes available with existing flags** (regime pinned,
`--no-tensor`):

| arm | flags |
|---|---|
| prefill-faithful | `--matrix-dim 4096 --coll-min 64M --coll-max 168M --gemms-per-coll 2` |
| decode-faithful | `--matrix-dim 1024 --coll-min 256K --coll-max 1M --gemms-per-coll 2` |
| true TP2 domain | `--gpus 4,5` (the pair holding cor04's susceptible GPU5) |

The TP2 arm matters because production runs **four independent 2-GPU lockstep
domains**, not one 8-way domain. pcieburn currently makes all eight coherent,
which overstates production on the coherent-amplitude axis while understating it
on sync rate. `--gpus` reduces the domain to a faithful TP2 pair without touching
the code.

**What still cannot be modelled without unfreezing the code.** The application
team's own view is that the prefill<->decode alternation is the most relevant
feature, and pcieburn holds one matrix dimension and one collective size for a
whole run. Representing it needs a duty-cycled alternation between two
`(matrix_dim, coll_size, gemms_per_coll)` regimes. If that is added, it should be
a new optional flag whose absence leaves the existing code path bit-identical, so
the seven arms already recorded here stay comparable.

**One free observation worth more than any run we can do.** `gpt-oss-20b` is
served TP1 and performs **zero** collectives — no cross-GPU synchronisation at
all. If this fault has ever occurred on a node serving only gpt-oss, then
synchronisation is not necessary for it and hypothesis 9 is dead without spending
a single test run. Ask before building the arms above.

---

## Hypothesis ledger

Status is post-freeze evidence only. "Prior indication" columns from the
scratched rounds have been dropped where real data now exists.

| # | Hypothesis | Post-freeze status | Next discriminating test |
|---|---|---|---|
| 1 | **Kernel dispatch rate.** Small kernels impose high-frequency load modulation; VRM output impedance peaks near its loop bandwidth, and rail ripple costs PCIe eye margin. | **REFUTED.** 54 GEMM/s vs 4,490-6,004 — a ~100x reduction — produced the most destructive arm tested (cor04 107 s, rgca18's first-ever fault). Predicted direction was the opposite. | closed unless a mechanism is proposed that survives this result |
| 2 | Power level. | **Not protective on its own.** Both susceptible nodes faulted at 450 W: rgca18 at ~609 s (fault 5) and cor04 at 224 s once warm (fault 6). A lower cap also *increases* cap clipping (47% vs 35% of samples) and, on rgca18, coincided with 378x more correctable errors. Mean power is a poor summary of this workload. | hold the cap fixed and vary soak; treat clipping fraction, not mean watts, as the stress metric |
| 3 | Clock/voltage pinning (transient suppression). | **Downgraded to "no benefit observed, confounded with wear".** The pin worked (cor04 SM-clock sd 141 → 28 MHz, rgca17 223 → 24) and cut power 495 → 395 W, but the cor04 evidence is 282.2 s at 0 prior fatals vs 214.5 s at 1 prior fatal — a 24% TTF change across exactly one increment of the covariate the power-matched staircase shows is worth 4.6× over five. The rgca17 leg (0 errors at 362 W vs 21 at 326 W) is a correctable-count difference on a node whose counts are not comparable across boots. The earlier "actively harmful, 2/2 nodes" reading is withdrawn. | `-lgc` **near boost** (~2550) so only variation is removed, not level. A low lock moves level, variation and dispatch rate at once. |
| 4 | Uptime / time since cold boot. | **DEMOTED from confirmed-causal.** The controlled pair (clean at 216 s uptime vs fault at 33,347 s, 438/440 W) stands, but the follow-ups moved against a smooth soak effect: clean at 4082 s uptime (pl450-8192, wear=5), fault at 812 s uptime (baseline re-run, 494 W), and the 32-run refit puts the soak MAP at zero. What remains viable is a **threshold** in (4082 s, 33347 s] — never tested by the smooth model — or the pair's effect belongs to wear/ambient (h14). | continue the bisection from above: ~4 h and ~9 h points at pl450-8192 on cor04; read jointly with the h14 staircase |
| 5 | **Interleaving** — the question the harness was built for. | **Near-closed, and traffic is exonerated far harder than the earlier "2.9% of Gen5 x16" framing.** Computed per run from `coll_bytes_pcie_link`: cor04 — the node that faults once per 667 s of load — ran **59.0% and 58.9% of Gen5 x16 for 606 and 607 s with zero AER counts and no fault**, while faulting reliably at 1.3–7.3%. Across 37 runs r(egress, fault) = **−0.18**; faulting runs span 0.81–4.60 GB/s per rank, clean runs 0.83–37.16. | The 128M-vs-4G arm is now redundant — the gpc1 arms already moved 8–40× more bytes for nothing. What is left is the confound: the high-traffic arms are also the low-power ones, so this shows traffic cannot *substitute* for power, not that it is irrelevant at fixed power. Closing h5 outright needs an arm holding ~495 W **and** ~37 GB/s, which the current knobs cannot express (raising collective share lowers duty cycle and therefore power). That is a harness design decision, not a run. |
| 6 | Per-machine susceptibility. | **Statistically supported, not just descriptive.** cor04 7/13 runs fatal vs rgca17 1/13 gives Fisher one-sided **p = 0.015**; on load exposure the rates are 1 per 667 s (cor04), 1 per 3,021 s (rgca18) and 1 per 7,490 s (rgca17) — an 11.2× ratio. Susceptibility sets rate and victim, not immunity: every fatal has come from a previously-identified slot — cor04 GPU5 `a0:01.1` (SDES, silent, **zero correctables in all 38 runs**) + GPU0 `00:01.1` (ERR_FATAL, correctable-generating); rgca17 gpu6 `34:10.0`; rgca18 gpu4 `2b:10.0`. **Correction:** the RGCA victims are not merely "different physical slots on identical boards" — `pcie_link_rootports.csv` is byte-identical on both nodes and shows a three-tier cascade in which both victims are **homologous branch positions**: 5 hops deep, behind third-tier switch `29`, on a leaf port ending `:10.0`, a set with exactly 2 of 8 members. Null probability (2/8)² = 0.06, feature chosen post hoc. No intermediate hop ever logged an error, so depth is a proxy for physical placement on the switchboard, not for accumulated jitter. **Junction temperature is excluded as the slot term** — see the thermal-rank table: all three victims sit at or behind two hotter, error-free slots except cor04 gpu5, and cor04 gpu0 fails from T-rank 4.00 with the node's second-lowest thermal swing. | read-only `Lane Error Status` / `LnkSta2` on the four implicated links **and on their same-branch peers `2b:00.0` / `2f:00.0`**, which the cascade map turns into a matched within-branch comparison available on both RGCA nodes; then card/slot swap using the direction split (h13) to interpret. |
| 7 | Topology class (riser vs switchboard). | **Confirmed, distinct in both mechanism and blast radius.** COR04 faults are `SDES`/`ERR_FATAL` with no precursor and contain 1 GPU. RGCA's fault ran a 200 s correctable ramp first and, because all eight GPUs sit behind the single root port `1a:01.1`, containment took **all 8** down with `device recovery failed`. | never pool the classes; compare within class only |

**New this round, not previously on the ledger:**

| # | Hypothesis | Basis | Next test |
|---|---|---|---|
| 8 | **PHY/SerDes margin falls as cap clipping rises** (term C). | **Downgraded: clipping fraction fails as a general metric** — r(clip%, AER) = +0.20, r(clip%, fault) = +0.03 over 26 runs; rgca17 ran 100% clipped at 574 W clean, lgc2100 faulted at 0% clip. What survives is within-slot: cor04 GPU0's BadTLP scales inversely with cap (450 W: 32k/4k; 575 W: 1/922), and only under the 8192 workload. | If pursued, pursue it per-slot on cor04 GPU0 with cap as the only variable; do not use clipping fraction as a fleet metric. |
| 9 | **Large coherent current transitions trigger the fatal, on either edge.** | **Reframed by the 38-run re-read: the pattern is real but it is rgca18-specific and it is about elapsed load time, not necessarily the edge.** All three terminal events that node has ever had landed within 20 s of 600 s of load — fault 4 at 598.1 s (1.9 s *before* load end), fault 5 at 608.8 s, and the previously-unflagged host crash at ≥589.3 s. Under a uniform hazard over a 600 s run that is p ≈ 3.7 × 10⁻⁵ for all three. cor04's seven faults sit at 10–47% of their load phases and rgca17's at 34%, so nothing generalises off rgca18. **The important consequence is a design defect, not a mechanism: every rgca18 run in the corpus is censored at almost exactly the point where its hazard concentrates**, so its eight `clean` verdicts carry far less information than their count suggests. Amplitude also sits against this: the pl450-warm fatal had the *lowest* PSU current swing of any 8192 run (55.8 A). | **One 1200 s run on rgca18, same arm.** "Release edge" and "~600 s of cumulative exposure" are indistinguishable while every run is 600 s: a fault near 600 s means cumulative exposure with a knee at ten minutes, near 1200 s means the release transition, no fault falsifies both. Cheaper and cleaner than N × 100 s, which moves teardown count and per-run exposure together, and it simultaneously tests whether the wrapper's own post-load config-space sweep across ~17 devices is the provocation. `--rank-stagger 5` still never run. |
| 11 | **IOMMU/VFIO changes the PCIe error-handling path**, so a passthrough host survives what a baremetal host does not. | **The IOMMU limb is REFUTED by the existing corpus; only the error-path limb survives.** Every node in the fault corpus already boots `amd_iommu=on iommu=pt` and reports `iommu: Default domain type: Passthrough` with IOMMU groups populated, on all 38 runs, with no `vfio` anywhere — and faults 10 times anyway. The proposed test had therefore been running all along. Two details from the same lines: the kernel answers `AMD-Vi: Unknown option - 'on'`, so `amd_iommu=on` is a no-op token and `iommu=pt` is what took effect; and the RGCA nodes additionally carry `pci=realloc=off,hpmemsize=0,hpiosize=0` while cor04 does not — a topology-class platform difference not previously recorded, and one that bears on re-enumeration after DPC. | What is left needs `vfio-pci` binding and guest-scoped containment, not a kernel-cmdline change. **Cheaper substitute available on the MGX box:** if `_OSC` AER/DPC ownership is a BIOS setting there as it is on the fleet, toggling firmware-first ↔ OS-first on identical hardware under identical load isolates the error-handling path as a single reversible variable, without needing a guest at all. |
| 12 | **Transient count, not exposure time, drives the fault** — the rate of coherent current *edges* rather than how long the load runs. | **REFUTED, and this is the investigation's first properly controlled single-factor test.** ~90x more coherent current edges at matched duty cycle and matched PCIe traffic changed time-to-fault by 11% (107 s -> 95 s) on cor04. A transient-count mechanism predicts TTF collapsing toward ~1 s. The arm also overshot power by 9% (541 vs 496 W), so it was *harsher* than the reference and still did not fault meaningfully faster. Together with hypothesis 1 (100x *less* dispatch -> worse), the transient-rate family is now empty in both directions. | Closed. n=1 per arm and exponential TTF gives +/-100% on a single event, so the 11% itself is noise — but the *absence* of a 90x effect is not a subtle inference. Reopen only with a mechanism that survives both refutations. |
| 13 | **The failing direction is uniformly GPU→upstream (GPU TX / board RX).** The marginal element is on the GPU side of every link: its transmitter, its PHY supply, or the TX pairs of its connector path. | **Mask prerequisite RESOLVED on fleet hardware — h13 now stands on the data.** The kernel printed the register itself every time a GPU endpoint logged: `nvidia 0000:01:00.0`, `0000:2d:00.0` and `0000:36:00.0` all show `status=0x00001100 mask=0x00001000`, i.e. **only Replay Timer Timeout masked; RxErr (bit 0) and BadTLP (bit 6) unmasked and clear**, while their upstream receivers show exactly the opposite bits set. This is stronger than the counter argument it replaces: the status register latches regardless of the mask, so the GPU rows are not un-incremented counters — the GPU's own hardware recorded no receive errors at the moment it reported. Corpus-wide over all 38 runs, deduped by port: **697,979 RxErr / 94,989 BadTLP / 6,275 BadDLLP** on upstream receivers against **zero of all three** on GPU endpoints, which logged 988 Rollover (their own replay timer). Only 4 ports of 40 have ever logged a correctable error. Reading caution: the `a0:01.1` dump showing `mask=0x00000000` belongs to an *Uncorrectable* error block, so it is the uncorrectable mask register, not a correctable-mask anomaly. **The earlier "checked on the MGX PRO 6000 under driver 595: CEMsk all clear" is withdrawn** — that reading was taken on the box later found to be running a root-level rootkit, and that unit has since been swapped out. | The `lspci -vvv` `CEMsk` check is no longer a gate. Next: `Lane Error Status` (Secondary PCIe cap, and again in the 32 GT/s Physical Layer cap) on the implicated receiving ports plus their same-branch peers, to localise which lanes take the hits; then the card/slot swap, where direction sharpens the interpretation — errors following the card implicate GPU TX PHY or its supply, errors staying with the slot implicate the slot's RX path or the connector TX pairs. |
| 14 | **Cumulative wear, not within-boot soak** — repeated faults/containments progressively degrade the marginal slot, and h4's "uptime" is proxying it. | **Strongest association in the corpus, and newly confounded by a kernel upgrade.** The power-matched cor04 `baseline-uncapped` triple is matched to 1 W of mean power, 2 MHz of mean clock, 5 MHz of clock sd, 0 MHz at p98 and 2 °C of ambient, and TTF fell 282.2 → 65.6 → 60.7 s. Over the fault set r(prior fatals, ln TTF) = **−0.86** pooled, **−0.88** within cor04, against r(power, ln TTF) = −0.21. Two rival covariates are now eliminated: within-boot uptime rose across the triple while TTF fell (pooled r(ln uptime, ln TTF) = **+0.35**, wrong sign for soak), and **ambient is no longer unmeasured** — the coldest pre-load idle GPU temperature in every bundle is a usable proxy, giving r(ambient, fault) = +0.06, 21.3 °C mean on faulting runs vs 21.1 °C clean, and only 2 °C of movement across the triple. **But cor04's kernel went `7.0.0-28` → `7.0.0-29` mid-campaign while both RGCA nodes stayed on `-28`, and kernel version is not a recorded run variable.** Every long-TTF cor04 fault (282, 214, 107, 224 s) is on `-28` and every short one (95, 66, 61 s) is on `-29` — a perfect separator, cleaner than prior-fault count, which has the 224 s fault out of order. Mitigating: the two boots' `_OSC` (26), `DPC` (44) and `ASPM` (8) line counts and command lines are identical, so error-handling configuration did not change — but the NVIDIA module was rebuilt. Under a constant hazard with mean 282 s, two consecutive draws ≤ 66 s have p = 0.040. | **Cheapest decisive run in the plan: boot cor04 on `7.0.0-28` and repeat `baseline-uncapped` at ~700 s uptime.** TTF back near 200–280 s implicates the kernel and dissolves the wear staircase; still near 60 s removes the kernel and leaves wear vs campaign-time. One reboot, no code. Then re-run the survival fit with a kernel term, with the two reclassified runs, and with rgca18's wear corrected to 2. Record `uname -r` in the manifest from now on. |
| 15 | **The soak variable is idle link-state residency / training age, not time or temperature.** GPUs autonomously downtrain to gen1 while idle; hours of idle-at-gen1 (or a stale training) may be what erodes the link, and retraining may reset it. | The pl450-warm fault (TTF 224 s at 33,347 s uptime) followed **~9 h of idle** — the node was at ambient temperature at load start, so whatever "soak" is, it survives complete thermal relaxation and **cannot be junction/board temperature**. What does persist across 9 h idle: link-training age, idle gen1 residency, driver/GSP state age. | Before a warm run, force a retrain (`setpci CAP_EXP+10.w=0020:0020` on the implicated ports, confirm gen5 x16 via `linkcheck.sh`), then run the arm. TTF back at cold values → training age/residency is the mechanism and a periodic retrain is a cheap fleet mitigation. TTF still short → driver/system state age or wear (h14). |
| 16 | **Thermal-cycle depth/count, not temperature level** — each cold→hot→cold excursion is one fatigue/fretting cycle; idle time matters as *cycle depth*, not as elapsed time. | Re-explains the 9 h-idle fault as the campaign's deepest thermal relaxation followed by full reheat, and gives h14 a physical mechanism (thermo-mechanical micro-motion → fretting at the connector, whose card-TX pins sit in their own physical group → tracks h13's direction uniformity). Constraints already in hand: fault timing rules out an instantaneous dT/dt trigger, and the h12 pair disfavours per-pass micro-cycling. **Narrowed by the 38-run re-read: the slot-selecting version of h16 is excluded.** Mean per-slot temperature rank across every run with all 8 GPUs in window puts the victims at 1.31 and 4.00 (cor04 gpu5, gpu0), 3.36 (rgca17 gpu6) and 3.00 (rgca18 gpu4) — mean victim rank 2.9 against a null of 4.5, entirely carried by cor04 gpu5, while the second- and third-hottest slots on every node have never produced a single error and cor04 gpu0 fails from mid-pack with the node's second-lowest ΔT. So thermal cycle *depth* does not predict which slot degrades; it may still drive degradation over time. Run-level ΔT remains collinear with power at fixed cooling on the fleet's axial-fan cards. | The 2x2 that separates h15 from h16 on a long-idled cor04, same arm: {retrain, no-retrain} x {pre-warm 10 min light load, cold start}. **Better instrument available on the MGX box:** a passively-cooled server card with BMC fan control breaks the power/ΔT collinearity outright — hold die power fixed and move ΔT by 15–25 °C, or hold temperature fixed and move power. Also the cycle-count arm: 6 x 100 s with ~5 min cool-down gaps vs 1 x 600 s at matched cap and total load-seconds. |
| 10 | **Per-boot link training outcome sets the link's margin**, independently of workload. | rgca17 GPU6 trained to three different states across three boots (265 s gen1 dwell; clean x16; x8 twice) and its error counts followed the trained state, not the arm. | `linkcheck.sh` at every boot before any load; retrain with `setpci CAP_EXP+10.w=0020:0020` and record whether the state holds. Never compare error counts across boots without recording the trained state. |


Observed times-to-fault, post-freeze, all 38 runs. cor04: **60.7, 65.6, 95.1,
106.9, 214.5, 224.2, 282.2 s** (7). rgca18: **598.1, 608.8 s** (2, the second
outside its run's capture window), plus a host death at ≥589.3 s of load with no
verdict recorded. rgca17: **201.4 s** (1). Pre-freeze and indicative only: 77,
117, 178, 200, 209, 1556 s.

Load exposure and rate: cor04 4,666 s / 7 → 1 per 667 s; rgca18 6,041 s / 2 → 1
per 3,021 s; rgca17 7,490 s / 1 → 1 per 7,490 s. **No run in the corpus exceeded
642 s of load**, so every `clean` verdict is bounded at roughly one arm-length —
and on rgca18 that bound sits exactly where its three terminal events landed.

---

## How many runs per arm

Computed by `sample_size.py`. Companion tools: `sync_rate.py` decomposes a run's
pass into GEMM-window and collective time from `events.csv`; `current_spectrum.py`
turns that into the modulation-frequency and edge-rate table used by hypothesis 12;
`vllm_shape.py` maps a tenant serving profile onto the harness's knobs. The observed calibration is cor04 on `933b21f`:
**4 faults in 2,627 s of load = one fault per 657 s**, which makes the per-600 s-run
fault probability about **0.60** — not 1.0. That number drives everything below.

**A binary screen needs 5 runs per arm, not 3.** Fisher exact, one-sided, two arms
of n:

| n/arm | perfect separation n/n vs 0/n | one discordant run (n-1)/n vs 0/n |
|---|---|---|
| 2 | p = 0.167 | p = 0.500 |
| 3 | p = 0.050 | p = 0.200 |
| 5 | **p = 0.004** | **p = 0.024** |
| 8 | p < 0.001 | p = 0.002 |

n=3 reaches p=0.05 *only* on a perfect split, and one discordant run collapses it
to 0.20. Since the per-run probability is ~0.60 rather than ~1.0, discordance is
expected, not exceptional: a destructive arm reads as **completely clean by chance
6.4% of the time at n=3** (0.4³) versus 1.0% at n=5 (0.4⁵). Five is the smallest
design that survives one anomalous run.

**But for arms expected to come back clean, duration beats repetition.** Zero
faults in exposure E bounds the hazard at 3/E:

| plan | exposure | 95% bound on hazard |
|---|---|---|
| 3 × 600 s | 1,800 s | < 1 per 600 s |
| 5 × 600 s | 3,000 s | < 1 per 1,000 s |
| 1 × 3600 s | 3,600 s | < 1 per 1,200 s |
| 3 × 3600 s | 10,800 s | **< 1 per 3,600 s** |

Against an observed hazard of one per 657 s, `5 × 600 s` only bounds it 1.5x below
what we already see — nearly worthless as a clean-arm claim. `3 × 3600 s` bounds it
5.5x below, for fewer runs of setup overhead.

**For a rate comparison, precision is set by fault count, not run count.** The
relative standard error of an exponential hazard is 1/sqrt(faults): 4 faults gives
±50%, 8 gives ±35%, 11 gives ±30%. Detecting a 2x difference between two arms needs
roughly 8–11 faults each, which at p≈0.6 means **13–18 runs per arm**. Reserve that
for the two arms that matter most; do not attempt it fleet-wide.

**Replicates are not exchangeable, and this is the same trap as before.** Soak is
confirmed causal (hypothesis 4), and five back-to-back 600 s runs plus overhead
span roughly an hour, so replicate index is partly a soak covariate. Do **not** run
5xA then 5xB. Use randomised blocks: each block contains every arm once, in an
order that differs per block, so soak drift is spread across arms instead of
aligned with one.

**Decide the reboot policy explicitly**, because it interacts with hypothesis 10.
Rebooting between replicates resamples the link-training outcome — correct if you
want to characterise the *node*, but it adds variance. Not rebooting characterises
*this boot's configuration* with less variance and narrower validity. Note the
built-in confound: a fault forces a reboot while a clean run does not, so reboot
status correlates with outcome unless you reboot before *every* run. Rebooting
before every run and holding uptime constant (say ~10 min) fixes the soak
covariate and makes replicates genuinely exchangeable, at the cost of reboot time.

**The recommended default**, then: 5 x 600 s per screening arm in randomised
blocks, reboot before each run, uptime matched; switch any arm that comes back
clean to 3 x 3600 s before calling it clean; and spend 13–18 runs only where a rate
ratio is the actual deliverable.

---

## Experimental protocol

1. **Verify identical commit *and* binary hash on every node** before a
   comparison set. This is the failure that cost a full round. Both are carried
   in the run tag (see above); the manifest independently records the full
   commit:
   ```sh
   for d in runs/*/; do printf '%-46s %s\n' "$(basename $d)" \
     "$(sed -n '/--- git ---/{n;p;q}' $d/manifest.txt)"; done | sort -k2
   ```
   Anything that does not group by one hash is not comparable. Tags whose `-g`
   parts match but whose `-b` parts differ mean the toolchain diverged, not the
   source.
2. **Duration ≥300 s, prefer 600 s.** A 180 s run straddled a real 178 s fault,
   and a 60 s run passed 13 minutes before a 117 s fault on the same machine.
   A clean run shorter than ~2× the longest observed time-to-fault is not
   evidence of anything.
3. **Set `-pl` explicitly on every run.** An applied limit persists for the whole
   boot session, so an unset run silently inherits the previous arm's value.
4. **≥3 trials per arm.** The base rate appears to be roughly one fault per node
   per 300 s run, so a single clean run distinguishes nothing.
5. **One variable per set.** Tabulate the design matrix first and check for
   collinear or empty cells before claiming a variable is isolated.
6. **Group by topology class.** Four machines split 2/2 is two experiments of two.
7. After a fault, prefer a power cycle only when required — it also resets uptime,
   which is itself hypothesis 4's variable.
8. **Do not run an arm series back-to-back on one boot.** Doing so makes arm order
   and uptime collinear, which is what happened to the first six post-freeze arms
   and is why none of them can separate cap from soak. Either reboot between arms
   so every arm starts at a comparable uptime, or counterbalance the order across
   nodes so soak is not aligned with the variable under test.
9. **Record uptime in the tag alongside the cap.** `uptime_seconds` is in the
   manifest, but the tag is what gets read at a glance, and soak is now a primary
   variable rather than a nuisance one.
10. **Report time-to-fault, never pass/fail alone.** The clearest signal found so
    far is a *trend in time-to-fault* across three runs that were each individually
    just "a fault". Pass/fail would have discarded it.
11. **Report correctable counts and the fatal outcome separately**, while noting
    that they are *not* independent: on rgca18 a 694k-error run did escalate, just
    outside the capture window.
12. **Never trust a `clean` verdict without checking the next run's
    `dmesg_before`.** The wrapper's `dmesg_after` snapshot can precede the fault
    (this happened: fault #5 landed 5 s after the snapshot), so a fatal during or
    just after teardown is invisible. Until the wrapper adds a settle delay, audit
    every clean verdict against the following run's pre-snapshot, which spans the
    gap.
13. **Record the trained link state at every boot, before any load.** Run
    `linkcheck.sh` and store it. A marginal port can come up x8, or dwell at gen1
    for minutes, and both silently change the electrical configuration under test —
    rgca17 GPU6 did all three across three boots. Error counts from different
    boots are not comparable without this.
14. **Run arms in randomised blocks, never n-of-A then n-of-B.** Replicate index
    carries soak (hypothesis 4) and boot-session link state (hypothesis 10). See
    [How many runs per arm](#how-many-runs-per-arm) for the sizing.
15. **Log boot_utc per run and group by boot session, not by arm order.** cor04 had
    four boot sessions across seven arms because its faults forced reboots;
    assuming one boot per node produced a wrong collinearity claim.

---

## Analysis discipline

Each item below caused a wrong conclusion in this investigation that was stated
confidently and only caught later.

**Never conclude absence from a truncated or deduplicated view.** `grep ... |
sort -u | head` was used twice to conclude "there are no DPC or AER messages in
this log." There were 1365 of them in 1376 lines; the head cut fired before
reaching them. Count first (`grep -c`), then look.

**Check that your view shows every field before asserting a zero.** The AER delta
table printed 5 of 8 correctable counters. On that basis the upstream switch
fabric was declared clean when every trunk port had recorded `NonFatalErr=1`.

**Compute the null probability and then respect it.** Two faults landing in the
same 2-run bucket out of 5 has a ~1-in-10 chance of happening by accident. That
was computed correctly and then quietly leaned on anyway for several turns, until
a 96 h clean run destroyed it.

**Look for collinearity before declaring a confound broken.** Every low-uptime
run was power-capped and every high-uptime run was at ≥500 W. The confound was
announced as broken while the design matrix still had two variables moving
together and two empty cells.

**Verify samples come from the same regime before differencing them.** A 20%
throughput gap between two nodes turned out to be non-tensor FP32 on one and
TF32 tensor on the other — the tensor math mode engages at *half the run
duration*, so a short run and a long run measure different code paths.

**When two datasets differ in shape but not in value, suspect the sampler.** One
node's throughput looked oscillatory and another's flat. Both were identical
within 0.35%. The `--report-interval` gate had been throttling the event log as
well as the console, aliasing one precision out entirely — 2473 half passes and
zero single passes logged across 1800 s.

**Bucket-average before claiming a trend.** A drift check on single samples across
a sawtooth workload showed a pattern that vanished under 5-minute means.

**Establish that a measurement can detect the thing before treating its absence
as evidence — and re-check that judgement when the corpus grows.**
`aer_uncorrectable.csv` came back empty and was first explained as firmware
clearing the registers, then as a 1 Hz poll too slow to catch the bit before DPC
contains the link. **Both explanations were wrong.** Over 38 runs the file holds
70 rows in all three reason-`00`/SDES faults and zero in all six
reason-`10`/`ERR_FATAL` faults and all 29 clean runs — a perfect classifier,
p = 0.012. The emptiness is diagnostic: in the `ERR_FATAL` class the root port
never records an uncorrectable error of its own, so there is nothing to poll. The
original discipline is still right, but it cuts both ways — "this instrument
can't see it" is itself a claim that needs testing against every case, not just
the ones that came back empty.

**A zero AER counter delta on a run that ended in containment is not evidence of
no precursor.** This points the opposite way to the `dmesg`-undercount item
below, and both are true at once. Fault 2 logged a correctable line *and* an
`ERR_FATAL` on `00:01.1`, yet its `aer_delta.txt` reads **all zero** — the
post-run counter read happens after the endpoint has gone. Faults 8 and 9 show
the same shortfall in milder form (3 counts against 8 log lines; 8 against 15).
So: take magnitudes from the counters, take *existence and sequence* from the
kernel log, and never infer "no correctables preceded this" from a zero delta on
a fatal run.

**Check whether an analysis script's own filters silently drop data.**
`corr_matrix.py` hard-coded `933b21` in its run-directory regex and skipped the
nine newest runs without warning; its fixed t=60–93 s power window blended
teardown idle into every short-TTF fault, turning 494 W into 184 W; and it summed
AER rows without deduping, counting one RGCA root-port event eight times. A
script that produces a plausible table from a silently truncated corpus is more
dangerous than one that errors. Print the row count the script actually used and
compare it against `ls -d runs/2026* | wc -l` before reading any number out of it.

**Render the artifact; do not trust the exit code or file size.** A plotting
script produced correctly-autoscaled axes with no data drawn at all, because an
undefined point terminates a line in gnuplot and the per-GPU filter left every
trace as isolated points. The images looked entirely plausible until viewed.

**Do not recite tool field or flag names from memory.** `power.enforced_limit`
does not exist; it is `enforced.power.limit`. Enumerate from the tool
(`nvidia-smi --help-query-gpu`, `ipmitool sdr list`, a Redfish GET on the
collection) and quote what the target machine reports.

**A high correctable-error count is not evidence of impending failure, and a low
one is not reassurance.** The same port on the same node under the same workload
logged 694,377 `RxErr` and stayed up, then 1,836 and died. An AER-rate alarm built
on the obvious reading of those counters would have fired on the healthy run and
stayed silent on the fatal one.

**Exclude post-fault rows before averaging any NVML field.** The trace keeps
running after a rank dies, so a whole-run mean silently blends load with teardown
idle. This put one node's per-GPU power at ~400 W against ~520 W for its peers and
made the victim GPU look like a dramatic outlier; in a matched t=60-105 s window
the same GPU was +2% on power. Always window explicitly, and window identically
across runs of different length.

**A quiet period is not recovery.** Correctable storms on every node that has them
are episodic — bursts separated by long silences. rgca18 went **80 s with zero
logged errors** immediately before its fatal containment. Do not read a gap as the
problem having passed.

**`dmesg` line counts are not error counts.** `aer_ratelimit: N callbacks
suppressed` means the kernel log undercounts by orders of magnitude — 596 logged
lines against 694,377 in the sysfs counters. Take magnitudes from
`aer_delta.txt`/sysfs and use the kernel log only for sequence and timing.

**Do not expand a run-tag token yourself.** `lgctdp` was read as "no clock lock"
when it meant `nvidia-smi --lock-gpu-clocks=tdp`, i.e. clocks *were* pinned, and
the advice built on that misreading was backwards. Tags are operator shorthand,
not a schema; the manifest's `nvidia-smi -q -d CLOCK` snapshot and power-limit
table are the authority on what was actually applied.

**Check the PSU trace's own cadence before correlating with it.** `psu_current.csv`
is nominally 0.25 s but actually lands near 1 Hz — the interval is `ipmitool`
latency plus the sleep — with dozens of gaps over 1 s and several spurious >50%
current drops per run. One such bad read appeared 2.7 s before a fault and looked
like a precursor; NVML showed system power flat across the same moment. The series
is usable for envelopes, not for event correlation.

**A `clean` verdict is a claim about the capture window, not about the hardware.**
rgca18's 600 s run at 450 W reported `clean` with `exit_code 0` and then took a
fatal containment 8.8 s after the load ended — 5 s after the wrapper's last
`dmesg_after` snapshot. An entire conclusion ("massive correctable storm with no
escalation") was built on that false negative and had to be withdrawn. Bound every
clean verdict by checking the *next* run's `dmesg_before`, which covers the gap.

**A missing artifact is a louder signal than a bad one — audit for absence, not
just for content.** The corpus's second false negative was found by scanning for
NUL bytes, not by reading anything: one rgca18 bundle has six live-written files
truncated to a page boundary, no `dmesg_after.txt`, no `faults.txt`, no
`aer_delta.txt`, and a manifest with no `verdict` line at all — the host died
mid-run and every parser downstream read the blank verdict as clean. Before
analysing a batch, check that every bundle has the same file list and that
`verdict`/`exit_code` are actually present, and grep the corpus for NUL bytes.
`if [ -z "$verdict" ]` must never fall through to "clean".

**Check the trained link state before comparing error counts.** rgca17's GPU6 port
looked like the noisiest link in the fleet on one boot (8018 BadTLP, after dwelling
265 s at gen1 where every peer took 5.5 s) and produced exactly zero errors on
another — because it had trained to **x8**, half width, and stayed there for two
whole runs while reporting `clean`. The quietest node in a table can be the most
degraded one. Width and gen are part of the experimental condition, not background.

**Do not average an NVML field across a length-mismatched window.** The trace keeps
running after a rank dies, so a whole-run mean blends load with teardown idle. This
put one node's per-GPU power at ~400 W against ~520 W for its peers and made the
victim GPU look like a dramatic outlier; in a matched t=60-105 s window it was +2%.

**Check what else a knob moves before calling it a single-factor lever.**
`--gemms-per-coll` was specified as an edge-rate knob at fixed amplitude and is
actually an edge-rate *and duty-cycle* knob: at N=1 the GEMM duty fell 94% -> 37%
and mean power 494 -> 277 W, so the arm could not test what it was built for. Write
down every quantity a knob changes — rate, amplitude, duty cycle, mean power,
traffic — and confirm from the artifacts which of them actually moved, before
reading the outcome as evidence about the intended factor.

**Do not infer an edge rate or slew from NVML.** `power.draw` repeats across 3–4
consecutive 100 ms polls, so the effective update period is ~300–400 ms and Nyquist
is ~1.5 Hz. The current edge that hypothesis 12 concerns is expected in the
microsecond range — we are blind to it by a factor of roughly 10⁴. The 193 W
amplitude figure is a valid *envelope* (a sub-Hz quantity NVML can resolve) and a
**lower bound** on the true step, since any faster excursion inside the averaging
window is smoothed away. `dmon` and the BMC are slower still. Measuring the edge
needs a current probe on the 12 V cables; nothing in the artifact bundle can
substitute.

**Know your sampler's real resolution before claiming a transient.** NVML repeats
the same `power.draw` value across 3-4 consecutive 100 ms polls, so its effective
update period is ~300-400 ms and it cannot resolve slew rate. Any di/dt argument
from this data rests on the coherent step structure and the endpoints, never on a
measured rate of change. Separately, `psu_current.csv` is nominally 0.25 s but
lands near 1 Hz — `ipmitool` latency plus the sleep — with several spurious >50%
current drops per run; one such bad read appeared 2.7 s before a fault and looked
like a precursor while NVML showed system power flat. Envelopes only, never event
correlation.

**A quiet period is not recovery.** Correctable storms are episodic on every node
that has them. rgca18 went 80 s with zero logged errors immediately before fault
#4. Do not read a gap as the problem having passed.

**`dmesg` line counts are not error counts.** `aer_ratelimit: N callbacks
suppressed` means the kernel log undercounts by orders of magnitude — 596 logged
lines against 694,377 in the sysfs counters. Take magnitudes from
`aer_delta.txt`/sysfs; use the kernel log for sequence and timing only.

**Do not expand a run-tag token yourself.** `lgctdp` was read as "no clock lock"
when it meant `nvidia-smi --lock-gpu-clocks=tdp` — clocks *were* pinned, and the
advice built on that misreading was backwards. Tags are operator shorthand, not a
schema; the manifest's `nvidia-smi -q -d CLOCK` snapshot and power-limit table are
the authority on what was applied.

**Check boot_utc before treating uptime as a within-session trend.** A monotonic
rise in uptime across a run series looked like continuous soak; the runs spanned
four boots, and the fitted threshold extrapolated from them was refuted by the
first point outside the fitted range.

**"No servers were down" is not "no faults occurred."** On baremetal this fault
produces `Xid 154 — Node Reboot Required` and the node needs rebooting, so it
registers as downtime. Under GPU passthrough, containment fires at the *host* root
port and can take out the guest while the host stays up — invisible to any monitor
keyed on host availability. Ask for the fault signature, not the availability
figure.

**A `vfio-pci` host emits no `Xid` lines at all.** With GPUs bound to VFIO for
passthrough there is no NVIDIA driver loaded for them on the host, so `Xid`, `NVRM`
and "GPU has fallen off the bus" cannot appear in the host log by construction —
only `pcieport ... DPC: containment event` and AER lines do. A fleet-wide scrape for
`Xid` will return zero from every passthrough host whether or not the fault fired.
Search for the `pcieport` signatures instead.

**Check which tenant or population a supplied workload profile describes before
treating it as representative.** A detailed vLLM serving profile arrived and was
initially written up as "production fidelity" for the failing nodes. It was the
profile of the tenant with *zero* incidents, while every node in the fault corpus
belongs to a different tenant whose workload is unknown. That inverts the profile's
role from target to negative control, and inverts what a reproduction on it would
mean.

**State plainly which of your own claims a new result withdraws.** Several
conclusions here were superseded, and the value of the ledger depends on old
claims being retired explicitly rather than left standing alongside their
replacements.
