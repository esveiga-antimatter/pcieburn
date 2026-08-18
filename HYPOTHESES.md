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

Consequently **every hypothesis below is untested**, including ones earlier
rounds appeared to exclude. Prior indications are recorded only so the same
ground is not re-covered blindly, never as evidence.

### Open item

The manifest records the *source* commit but nothing verifies the **binary** was
built from it. `pcieburn` is untracked, so two nodes can sit on identical commits
and run different binaries with nothing in the artifacts revealing it. A
`sha256sum` of the binary in the manifest closes this. Until then, binary
identity must be checked by hand before any comparison set.

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

## Hypothesis ledger

| # | Hypothesis | Prior indication (not evidence) | Discriminating test |
|---|---|---|---|
| 1 | **Kernel dispatch rate.** Small kernels impose high-frequency load modulation; VRM output impedance peaks near its loop bandwidth, and rail ripple costs PCIe eye margin. | **never tested** | `--matrix-dim` 8192 → 4096 → 2048 → 1024 with `--precision single --no-tensor`. ~510× span in launches/s. Predicts time-to-fault falls monotonically as dim falls. |
| 2 | Power level. | mixed: faults at 501–542 W per GPU, but also a clean 300 s run at 535 W | `-pl` sweep at fixed duration and uptime, ≥3 trials per level |
| 3 | Clock/voltage pinning (transient suppression). | looked negative, but on scratched data | `-lgc` **near boost** (~2550) so power stays high and only variation is removed. A low lock (e.g. 2100) reduces level, variation *and* dispatch rate at once and cannot separate them. |
| 4 | Uptime / time since cold boot. | two faults within 4 min of each other in uptime, then contradicted by a clean 96 h run | repeats at ~1 h / ~4 h / ~9 h uptime on one boot session |
| 5 | **Interleaving** — the question the harness was built for. | looked irrelevant: diag reproduces with zero collectives | `--coll-min/--coll-max` 128M vs 4G at fixed compute. 32× bytes moved. If time-to-fault is unchanged, PCIe traffic is exonerated outright. |
| 6 | Per-machine susceptibility. | one node survived where two failed | ≥3 uncapped repeats per node to establish a base rate |
| 7 | Topology class (riser vs switchboard). | fault signatures differed (`RxErr` present on RGCA, absent on COR04) | never pool the two classes; compare within class only |

Observed times-to-fault, pre-freeze and therefore indicative only: 77, 117, 178,
200, 209, 1556 s.

---

## Experimental protocol

1. **Verify identical commit *and* binary hash on every node** before a
   comparison set. This is the failure that cost a full round.
   ```sh
   for d in runs/*/; do printf '%-46s %s\n' "$(basename $d)" \
     "$(sed -n '/--- git ---/{n;p;q}' $d/manifest.txt)"; done | sort -k2
   ```
   Anything that does not group by one hash is not comparable.
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

**State plainly which of your own claims a new result withdraws.** Several
conclusions here were superseded, and the value of the ledger depends on old
claims being retired explicitly rather than left standing alongside their
replacements.
