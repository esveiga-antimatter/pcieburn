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

**Post-freeze measurement is under way.** Seven arms x three nodes on `933b21f`
are recorded in [Post-freeze results](#post-freeze-results-933b21f), with six
faults. They refute hypothesis 1 outright, confirm hypothesis 4 (soak) as causal
on a single controlled comparison, and establish that no single-factor model fits.
**Read the corrections subsection first** — three confident claims from an earlier
revision of this file were wrong, one of them because the harness reported a fatal
run as `clean`.

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
- Every DPC event observed reads `ERR_FATAL received from <device>`. A device that
  had browned out or vanished cannot transmit a fatal error message upstream.
  **A momentary power dropout is not the mechanism.**
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

### The six faults

| # | node | arm | t | port | GPU | signature | radius |
|---|---|---|---|---|---|---|---|
| 1 | cor04 | uncapped 2048 | 282 s | `a0:01.1` | GPU5 `a1:00.0` | `SDES`, no correctables | 1 GPU |
| 2 | cor04 | lgc2100 | 214 s | `00:01.1` | GPU0 `01:00.0` | `ERR_FATAL`, zero AER counters | 1 GPU |
| 3 | cor04 | pl575 8192 | 107 s | `a0:01.1` | GPU5 `a1:00.0` | `SDES`, correctables on GPU0's port | 1 GPU |
| 4 | rgca18 | pl575 8192 | 598 s | `1a:01.1` | GPU4 `2d:00.0` | 1836 RxErr → Rollover → `ERR_FATAL` | **all 8** |
| 5 | rgca18 | pl450 8192 | ~609 s | `1a:01.1` | GPU4 `2d:00.0` | 694,377 RxErr → `ERR_FATAL` | **all 8** |
| 6 | cor04 | pl450-warm 8192 | 224 s | `00:01.1` | GPU0 `01:00.0` | `ERR_FATAL`, 3954 BadTLP | 1 GPU |

cor04 4, rgca18 2, rgca17 0. cor04 alternates between its two implicated ports
with no discernible pattern (GPU5, GPU0, GPU5, GPU0). `faulty=0 nan=0` in every
run — data corruption has still never been observed.

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
| 3 | Clock/voltage pinning (transient suppression). | **Actively harmful, 2/2 nodes.** `lgc2100` did worse than the free-clock arm at *higher* power on both cor04 (398 W faulted vs 437 W clean) and rgca17 (326 W, 21 errors vs 362 W, 0). Folded into term C. | `-lgc` **near boost** (~2550) so only variation is removed, not level. A low lock moves level, variation and dispatch rate at once. |
| 4 | Uptime / time since cold boot. | **CONFIRMED CAUSAL.** cor04, one boot, one cap, one workload, nothing touched: clean at 216 s uptime, FAULT at 224 s when started at 33,347 s uptime. The linear-collapse model is refuted — at 9.3 h the time-to-fault was *longer* than at 51 min on a higher cap, so the response saturates. | bisect `216 s < t < 33,347 s` at pl450 on cor04: ~1 h, ~2 h, ~4 h uptime, arm held fixed |
| 5 | **Interleaving** — the question the harness was built for. | Weakened further: the 8192 arm moved 0.8% of Gen5 x16 versus 2.5-3.0% in the baseline and faulted *more*. | `--coll-min/--coll-max` 128M vs 4G at fixed compute. 32x bytes moved. If time-to-fault is unchanged, PCIe traffic is exonerated outright. |
| 6 | Per-machine susceptibility. | **Confirmed, and per-slot.** cor04 GPU5 `a0:01.1` + GPU0 `00:01.1` (alternating, 4 faults); rgca18 GPU4 `2b:10.0` (2 faults); rgca17 zero. rgca17/18 share identical BDF maps, so these are different physical slots on identical boards. Term A. | read-only `Lane Error Status` and `LnkSta2` equalization on the three implicated ports vs healthy peers; then card/slot swap |
| 7 | Topology class (riser vs switchboard). | **Confirmed, distinct in both mechanism and blast radius.** COR04 faults are `SDES`/`ERR_FATAL` with no precursor and contain 1 GPU. RGCA's fault ran a 200 s correctable ramp first and, because all eight GPUs sit behind the single root port `1a:01.1`, containment took **all 8** down with `device recovery failed`. | never pool the classes; compare within class only |

**New this round, not previously on the ledger:**

| # | Hypothesis | Basis | Next test |
|---|---|---|---|
| 8 | **PHY/SerDes margin falls as cap clipping rises** (term C). | **Downgraded: clipping fraction fails as a general metric** — r(clip%, AER) = +0.20, r(clip%, fault) = +0.03 over 26 runs; rgca17 ran 100% clipped at 574 W clean, lgc2100 faulted at 0% clip. What survives is within-slot: cor04 GPU0's BadTLP scales inversely with cap (450 W: 32k/4k; 575 W: 1/922), and only under the 8192 workload. | If pursued, pursue it per-slot on cor04 GPU0 with cap as the only variable; do not use clipping fraction as a fleet metric. |
| 9 | **Large coherent current transitions trigger the fatal, on either edge.** | Both rgca18 faults sat on one: #4 with all eight GPUs clipping at 577–587 W (~4.6 kW coherent), #5 one second after ~2 kW was shed to idle. The workload steps all eight GPUs together by ~800 W inside a single 100 ms sample. **Weak: the other four fatals (all cor04) were mid-load, and correctable-error timing clusters at release in only 2 of 7 runs.** Supporting circumstantial evidence from outside this corpus: `dcgmi diagnostic` reproduces and drives all GPUs from one process; `gpu-burn` uses the same GEMM loop from independent per-GPU processes and has never reproduced. | N × 100 s runs vs 1 × 600 s at equal total load — if the release edge matters, fault probability scales with the number of teardowns. Also `--rank-stagger 5` to de-correlate the steps across ranks, never yet run. |
| 11 | **IOMMU/VFIO changes the PCIe error-handling path**, so a passthrough host survives what a baremetal host does not. | hivenet reported zero servers down over 28 days; the fault corpus is entirely `tenant: fal` baremetal. Both tenants share the `cor`, `rgca` and `roca` sites, so site and topology class are *not* the difference. `perfvm_host` membership sets `kernel_iommu: true` → GRUB IOMMU + VFIO, which routes error recovery through `vfio-pci` rather than the NVIDIA RM. We already know `_OSC` AER/DPC ownership varies across this fleet with identical board and BIOS. | Enable IOMMU on a test node via kernel cmdline — no code, no new instrumentation — and re-run a known-reproducing arm. **First, though, resolve the two cheaper questions below: the comparison may be an artifact.** |
| 12 | **Transient count, not exposure time, drives the fault** — the rate of coherent current *edges* rather than how long the load runs. | **REFUTED, and this is the investigation's first properly controlled single-factor test.** ~90x more coherent current edges at matched duty cycle and matched PCIe traffic changed time-to-fault by 11% (107 s -> 95 s) on cor04. A transient-count mechanism predicts TTF collapsing toward ~1 s. The arm also overshot power by 9% (541 vs 496 W), so it was *harsher* than the reference and still did not fault meaningfully faster. Together with hypothesis 1 (100x *less* dispatch -> worse), the transient-rate family is now empty in both directions. | Closed. n=1 per arm and exponential TTF gives +/-100% on a single event, so the 11% itself is noise — but the *absence* of a 90x effect is not a subtle inference. Reopen only with a mechanism that survives both refutations. |
| 13 | **The failing direction is uniformly GPU→upstream (GPU TX / board RX).** The marginal element is on the GPU side of every link: its transmitter, its PHY supply, or the TX pairs of its connector path. | Corpus-wide aggregation of all 27 runs' AER tables: upstream-facing receivers logged **697,865 RxErr / 93,547 BadTLP / 6,274 BadDLLP**; GPU (dev) rows logged **zero** of all three, while logging 959 Rollover — so their AER reporting works and the zero is real. The story is coherent: GPU transmits bad → receiver logs RxErr/BadTLP → GPU replays → GPU Rollover. Sole exception: 58 Rollover on `34:10.0` (downstream TX) in the one bad-training run, 0.008% of the corpus. Both fatal classes fit too: `SDES` = the root port stops hearing the GPU; `ERR_FATAL received from <GPU>` = the GPU reports its own escalation. Also explains the BMC symptom: host-side receivers seeing bad data is what IPMI misreads as PERR/SERR. | On a card/slot swap (h6's test), direction sharpens the interpretation: if errors follow the card, GPU TX PHY or its supply; if they stay with the slot, the slot's RX path or the connector TX pairs. Read-only now: `Lane Error Status` on the three implicated receiving ports localizes which lanes take the hits. |
| 14 | **Cumulative wear, not within-boot soak** — repeated faults/containments progressively degrade the marginal slot, and h4's "uptime" is proxying it. | The cap-575 TTF decline on cor04 (282 → 214 → 107 → 95 s) spans **four different boots**, so within-boot uptime, wall-clock campaign time, cumulative fault count (0, 1, 2, 4 prior), and ambient time-of-day all rise together across it — the series cannot distinguish them. Weak counter-evidence: GPU0's BadTLP at pl450 *fell* from 32,275 (early) to 3,954 (late). Ambient is entirely unmeasured. | Cheap and near-decisive: re-run `baseline-uncapped` exactly, at matched ~700 s uptime. Wear predicts TTF ≪ 282 s; soak predicts ≈ 282 s. Also start logging BMC inlet temperature per run — one `ipmitool sdr` read in the wrapper — to kill the ambient confound. |
| 15 | **The soak variable is idle link-state residency / training age, not time or temperature.** GPUs autonomously downtrain to gen1 while idle; hours of idle-at-gen1 (or a stale training) may be what erodes the link, and retraining may reset it. | The pl450-warm fault (TTF 224 s at 33,347 s uptime) followed **~9 h of idle** — the node was at ambient temperature at load start, so whatever "soak" is, it survives complete thermal relaxation and **cannot be junction/board temperature**. What does persist across 9 h idle: link-training age, idle gen1 residency, driver/GSP state age. | Before a warm run, force a retrain (`setpci CAP_EXP+10.w=0020:0020` on the implicated ports, confirm gen5 x16 via `linkcheck.sh`), then run the arm. TTF back at cold values → training age/residency is the mechanism and a periodic retrain is a cheap fleet mitigation. TTF still short → driver/system state age or wear (h14). |
| 10 | **Per-boot link training outcome sets the link's margin**, independently of workload. | rgca17 GPU6 trained to three different states across three boots (265 s gen1 dwell; clean x16; x8 twice) and its error counts followed the trained state, not the arm. | `linkcheck.sh` at every boot before any load; retrain with `setpci CAP_EXP+10.w=0020:0020` and record whether the state holds. Never compare error counts across boots without recording the trained state. |


Observed times-to-fault. Post-freeze, on `933b21f` and therefore usable: **107,
214, 282, 598 s** (cor04 x3, rgca18 x1). Pre-freeze and indicative only: 77, 117,
178, 200, 209, 1556 s.

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
as evidence.** `aer_uncorrectable.csv` came back empty and was first explained as
firmware clearing the registers — until an OS-first node produced the same empty
file. At 1 Hz the poll simply never catches the bit before DPC contains the link.
Absence of `SDES` there is not evidence; the kernel log's `ERR_FATAL received
from <device>` is.

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
