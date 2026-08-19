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

**Post-freeze measurement has begun.** Six arms x three nodes on `933b21f` are
recorded in [Post-freeze results](#post-freeze-results-933b21f). They refute
hypothesis 1 outright and establish that no single-factor model fits the data.
They also inherited a design flaw that bounds what they can prove: arm order and
node uptime are perfectly collinear across the whole set.

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

Six arms × three nodes, 600 s each, all on the frozen binary. **The goal here is
the platform's margin envelope, not a single guilty part.** Read the ledger with
that framing: an arm that eliminates a candidate has narrowed the envelope, and a
*clean* run that accumulated enormous error counts is a more informative result
than one that simply failed.

| arm | workload | ~W/GPU (cor04/17/18) | cor04 | rgca17 | rgca18 |
|---|---|---|---|---|---|
| `baseline-uncapped` | 2048 mixed | 480 / 402 / 400 | **FAULT 282 s** | clean, 8018 BadTLP | clean, 0 |
| `pl450` | 2048 mixed | 437 / 362 / 360 | clean, 0 | clean, 0 | clean, 0 |
| `pl525` | 2048 mixed | 468 / 390 / 388 | clean, 0 | clean, 8 BadTLP | clean, 0 |
| `pl575-lgc2100` | 2048 pinned | 398 / 326 / 321 | **FAULT 214 s** | clean, 21 RxErr | clean, 0 |
| `pl575-rgc-notensor-single-8192` | 8192 single | 494 / 525 / 522 | **FAULT 107 s** | clean, 25 BadTLP | **FAULT 598 s** |
| `pl450-rgc-notensor-single-8192` | 8192 single | 438 / 419 / 416 | clean, **32,275** BadTLP | clean, 0 | clean, **694,377** RxErr |

Power figures are means over a matched t=60–105 s window. `faulty=0 nan=0` in all
eighteen runs — data corruption has still never been observed.

The five faults, with their signatures:

| node | arm | t | port | GPU | signature | blast radius |
|---|---|---|---|---|---|---|
| cor04 | uncapped | 282 s | `a0:01.1` | GPU5 `a1:00.0` | `SDES`, no correctables | 1 GPU → x0 |
| cor04 | lgc2100 | 214 s | `00:01.1` | GPU0 `01:00.0` | `ERR_FATAL`, zero AER counters | 1 GPU → x0 |
| cor04 | pl575-8192 | 107 s | `a0:01.1` | GPU5 `a1:00.0` | `SDES`, 3 correctables *on GPU0's port* | 1 GPU → x0 |
| rgca18 | pl575-8192 | 598 s | `1a:01.1` (shared) | GPU4 `2d:00.0` | 1836 RxErr → Rollover → `ERR_FATAL` | **all 8 GPUs** → x0 |

### Three single-factor models are dead

**Hypothesis 1, kernel dispatch rate: refuted.** The 8192-dim single arm runs at
**54 GEMM/s** against 4,490–6,004 for the 2048 default — a ~100× reduction — and
it is the most destructive arm tested, producing cor04's fastest failure ever
(107 s) and rgca18's first failure ever. The prediction was that time-to-fault
falls as dim falls; it rose. PCIe payload moved the same way: 0.8% of Gen5 x16
against 2.5–3.0% in the baseline, i.e. *less* traffic with *more* faults, which
independently re-confirms that payload volume is not the driver.

**"A degraded link causes the fatal": refuted.** Same node, same port, same
workload, cap alone differing — rgca18 `2b:10.0`:

- 575 W → **1,836** RxErr → *fatal containment*
- 450 W → **694,377** RxErr (378× more), 46,587 BadTLP, 6,231 BadDLLP → *clean,
  no DPC, link held `gen5 x16` for the whole run*

**Correctable error volume does not predict fatal escalation, and must not be
used as a proxy for it.** The decoupling is also spatial: on cor04, in both the
pl575 and pl450 8192 arms, the correctable errors land on **GPU0's** port while
the fatal lands on **GPU5's** port. Different slots, same run, twice.

**"Power level alone": refuted as sufficient.** It is monotonic *within* the 8192
workload (575 W → 2 of 3 nodes fault, 450 W → 0 of 3) but not within the 2048
workload, where 480 W faulted, 468 W and 437 W were clean, and 398 W faulted.

### The margin model

What fits is a product of separate terms, not one cause:

**A — per-slot susceptibility (fixed).** Exactly one susceptible link per node,
node-specific: cor04 GPU5 `a0:01.1` and GPU0 `00:01.1`; rgca17 GPU6 `34:10.0`;
rgca18 GPU4 `2b:10.0`. rgca17 and rgca18 have *identical* BDF maps, so these are
genuinely different physical slots on identical boards. This term sets whether a
node can fault at all, and where.

**B — electrical stress (workload-set).** Sustained power at high clock. Governs
how fast a susceptible slot is driven to its limit.

**C — PHY/SerDes margin (inversely related to B).** Correctable errors rise
sharply as power and clock *fall* — 694k at 450 W versus 1.8k at 575 W. Candidate
mechanisms are lower core/PHY voltage and constant power-limit throttling churn,
both of which cost eye margin. This term explains why `lgc2100`, the lowest-power
arm at 398 W, faulted cor04 at all, and why the counters mislead: C generates the
countable errors, B causes the fatal.

Scoring four structural models against five discriminating observations with
equal priors — likelihoods are judgments, and the specific values are soft while
the ordering is not:

| observation | power only | dispatch | bad link → fatal | A × B × C |
|---|---|---|---|---|
| rgca18: 378× more errors when clean | 0.3 | 0.3 | **0.02** | 0.8 |
| cor04 2048 non-monotonic in power | **0.05** | 0.2 | 0.3 | 0.6 |
| 8192: 2/3 fault at 575 W, 0/3 at 450 W, 100× less dispatch | 0.9 | **0.02** | 0.4 | 0.8 |
| exactly one susceptible slot per node, stable across arms | 0.1 | 0.1 | 0.7 | 0.9 |
| rgca17 noisiest link, zero faults in six arms | 0.2 | 0.2 | **0.05** | 0.7 |
| **posterior** | 0.02% | 0.02% | 0.06% | **99.9%** |

Each single-factor model is killed by a different observation, and no reweighting
rescues one: reweighting to save power-only worsens the non-monotonicity, and
saving bad-link worsens both the rgca18 and rgca17 rows.

### The erosion clock, and where the threshold might be

cor04's three faults, ordered by node uptime at run start:

| uptime at start | time-to-fault |
|---|---|
| 697 s | 282.2 s |
| 2441 s | 214.5 s |
| 3067 s | **106.9 s** |

Monotonic. A least-squares line through these three points reaches zero at
**~5100 s (85 min) of uptime** (R² = 0.85), which would mean a susceptible node
becomes near-instantly faultable somewhere around an hour of soak. rgca18's only
fault came at the highest uptime tested (6971 s), and rgca17 held clean at the
same 6938 s — consistent with term A gating whether the erosion ever cashes out.

**This is the "defined interval where things come crashing down" worth chasing,
and the fit is not yet evidence for it.** Three points, and the confound below
means workload moved together with uptime.

### The confound that limits every row above

**Arm order and node uptime are perfectly collinear.** All five of the first arms
ran back-to-back on a single boot per node, so uptime rises monotonically with
arm index (cor04: 697 → 1003 → 1675 → 2441 → 3067 s). The sixth arm, `pl450`
8192, is the *only* low-uptime run in the set — it started at 216 s uptime after
the reboot that rgca18's `Xid 154` forced — and it is also the only 8192 arm that
came back clean. Its cleanliness is therefore equally attributable to the cap or
to the cold boot, and the same ambiguity contaminates the 694k-error storm, which
ran on a freshly retrained link.

This is protocol item 5 ("one variable per set... check for collinear or empty
cells") being violated in exactly the way the protocol warned about, and item 7
already noted that a power cycle resets hypothesis 4's variable.

### The crossed design that resolves it

Two of the four cells exist; the two in flight complete it. Predictions are
recorded here **before** the data lands, so the result is diagnostic rather than
narrated after the fact:

| | ~3.5 min uptime | ~2 h uptime |
|---|---|---|
| **pl450** 8192 | done — 3/3 clean, 694k errors on rgca18 | *in flight* |
| **pl575** 8192 | *in flight, after reboot* | done — cor04 107 s, rgca18 598 s |

- If **soak dominates**: pl575-cold comes back clean or much slower to fault, and
  pl450-warm faults. The power ordering in the table above then needs re-reading
  as a soak ordering.
- If **power dominates**: pl575-cold still faults on cor04, and pl450-warm stays
  clean. Term B is confirmed and hypothesis 4 drops.
- If **both**: pl575-cold faults but later than 107 s, and pl450-warm stays clean
  but with error counts above 694k.
- Term C predicts, independently of which of the above holds, that pl450-warm
  produces *more* correctable errors than pl575-warm did on the same port. A
  single-factor power model predicts fewer.

---

## Hypothesis ledger

Status is post-freeze evidence only. "Prior indication" columns from the
scratched rounds have been dropped where real data now exists.

| # | Hypothesis | Post-freeze status | Next discriminating test |
|---|---|---|---|
| 1 | **Kernel dispatch rate.** Small kernels impose high-frequency load modulation; VRM output impedance peaks near its loop bandwidth, and rail ripple costs PCIe eye margin. | **REFUTED.** 54 GEMM/s vs 4,490-6,004 — a ~100x reduction — produced the most destructive arm tested (cor04 107 s, rgca18's first-ever fault). Predicted direction was the opposite. | closed unless a mechanism is proposed that survives this result |
| 2 | Power level. | **Necessary but not sufficient.** Monotonic within the 8192 workload (575 W: 2/3 fault, 450 W: 0/3) but not within 2048 (480 F, 468 C, 437 C, 398 F). Retained as term B of the margin model. Currently confounded with uptime. | the crossed uptime x cap design in flight; then `-pl 400` (the `power.min_limit` floor) |
| 3 | Clock/voltage pinning (transient suppression). | **Actively harmful, 2/2 nodes.** `lgc2100` did worse than the free-clock arm at *higher* power on both cor04 (398 W faulted vs 437 W clean) and rgca17 (326 W, 21 errors vs 362 W, 0). Folded into term C. | `-lgc` **near boost** (~2550) so only variation is removed, not level. A low lock moves level, variation and dispatch rate at once. |
| 4 | Uptime / time since cold boot. | **Promoted to primary.** cor04's three faults fall monotonically with uptime at start: 697 s -> 282 s TTF, 2441 s -> 214 s, 3067 s -> 107 s. Linear fit reaches zero near 85 min uptime (R2 = 0.85, n=3, workload confounded). | the crossed design in flight; then repeats at ~1 h / ~4 h / ~9 h on one boot with the arm held fixed |
| 5 | **Interleaving** — the question the harness was built for. | Weakened further: the 8192 arm moved 0.8% of Gen5 x16 versus 2.5-3.0% in the baseline and faulted *more*. | `--coll-min/--coll-max` 128M vs 4G at fixed compute. 32x bytes moved. If time-to-fault is unchanged, PCIe traffic is exonerated outright. |
| 6 | Per-machine susceptibility. | **Confirmed, and localized to one slot per node.** cor04 GPU5 `a0:01.1` + GPU0 `00:01.1`; rgca17 GPU6 `34:10.0`; rgca18 GPU4 `2b:10.0`. rgca17/18 share identical BDF maps, so these are different physical slots on identical boards. Term A. | read-only `Lane Error Status` and `LnkSta2` equalization on the three implicated ports vs healthy peers; then card/slot swap using correctable accumulation as the readout |
| 7 | Topology class (riser vs switchboard). | **Confirmed, distinct in both mechanism and blast radius.** COR04 faults are `SDES`/`ERR_FATAL` with no precursor and contain 1 GPU. RGCA's fault ran a 200 s correctable ramp first and, because all eight GPUs sit behind the single root port `1a:01.1`, containment took **all 8** down with `device recovery failed`. | never pool the classes; compare within class only |

**New this round, not previously on the ledger:**

| # | Hypothesis | Basis | Next test |
|---|---|---|---|
| 8 | **PHY/SerDes margin falls as core voltage/clock falls** (term C), so lower power caps buy fatal-margin at the cost of raw link margin. | rgca18 `2b:10.0`: 694,377 RxErr at 450 W clean vs 1,836 at 575 W fatal. rgca17 quieted 8018 -> 25 when power rose. | `-pl 400` on rgca18 with the 8192 arm. Term C predicts *more* than 694k errors and still no fatal; a power-only model predicts fewer. |


Observed times-to-fault. Post-freeze, on `933b21f` and therefore usable: **107,
214, 282, 598 s** (cor04 x3, rgca18 x1). Pre-freeze and indicative only: 77, 117,
178, 200, 209, 1556 s.

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
11. **Report correctable counts and the fatal outcome separately.** They are
    decoupled (see hypothesis 8), so a summary that blends them into "errors
    present" destroys the discriminating information.

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

**State plainly which of your own claims a new result withdraws.** Several
conclusions here were superseded, and the value of the ledger depends on old
claims being retired explicitly rather than left standing alongside their
replacements.
