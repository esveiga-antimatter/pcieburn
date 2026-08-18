# pcieburn — interleaved compute + PCIe/NCCL stress test

Combined GEMM-burst and NCCL-collective load generator, built to close the gap
between DCGM's `diagnostic` sub-test (which hammers compute and moves no data)
and `nccl-tests` (which moves data and barely touches compute).

| File | Purpose |
|---|---|
| `pcieburn.cu` | The harness: supervisor + per-GPU rank processes, GEMM burst, collectives, compare kernels |
| `Makefile` | `make preflight` to check the resolved toolchain, then `make` |
| `run_pcieburn.sh` | Run wrapper: safety gate, provenance, telemetry, orphan-free cleanup |
| `linkcheck.sh` | **Zero-cost PCIe link health screen — no load, safe on any node, usable as a fleet sweep** |
| `plot_run.gp` | Interactive gnuplot explorer for a run directory (`PNG=1` writes image files) |
| `../dataviz/` | Separate repo: multi-run explorer — overlay up to 4 runs, hover for values, oscilloscope cursors, zoom to area |

If you are triaging a node rather than stress-testing one, start with
`linkcheck.sh`. It needs no CUDA, applies no load, and on this fleet it has
flagged the failing GPU before any test was run.

## Which design option this is, and why

**This is Option B** — a custom harness that interleaves compute and collectives
per rank. Option A (a `gpu-burn` process and an `all_reduce_perf` loop running
side by side) is *not* built, though the wrapper makes it easy to run one
alongside anything else if you want that comparison.

The reasoning:

Option A produces *simultaneous* load but not *interleaved* load. Two
independent processes give no guarantee that a GEMM burst and a collective ever
overlap in time, and — more importantly — no guarantee they ever overlap in the
specific ordering a transformer layer produces: compute, then a synchronization
point where every rank enters a collective together, then compute again. Since
the open question is precisely *whether the interleaving itself matters* (as
opposed to just aggregate load), Option A cannot answer it. It would tell us
"concurrent load didn't reproduce it either", which is a weaker version of what
plain `nccl-tests` has already told us.

The effort argument for A also mostly evaporated here: this repo lives on a box
with no GPU and no CUDA toolchain, so neither option can be iterated locally.
The cost difference between A and B is authoring time, not debug-cycle time, and
B is the one that produces a result worth having.

What B buys concretely:

- The GEMM burst and the collective are issued **on the same CUDA stream**, so
  the collective is genuinely ordered behind the compute — a real
  synchronization point, not coincidental overlap.
- Every rank reaches that point together, because a collective *is* a barrier.
- The interleave granularity is a knob (`--gemms-per-coll`), so the same binary
  spans "one DCGM-sized burst per collective" through "a handful of GEMMs per
  collective", which is what inference actually looks like.

## Fidelity to the pattern that has actually reproduced the fault

The compute loop is a deliberate reimplementation of DCGM's
`GpuBurnWorker::Compute()` (`nvvs/plugin_src/diagnostic/DiagnosticPlugin.cpp:1249`,
Apache-2.0), cross-checked against `gpu-burn`'s `compute()`
(`gpu_burn-drv.cpp:213`, BSD-2-Clause). Preserved exactly:

| Property | Value | Source |
|---|---|---|
| GEMM shape | `CUBLAS_OP_N`/`CUBLAS_OP_N`, `m=n=k=lda=ldb=ldc=matrix_dim` | DiagnosticPlugin.cpp:1249 |
| α, β | 1 and 0, in the native type (`__half` for Hgemm — *not* mixed precision) | DiagnosticPlugin.cpp:1258-1263 |
| Synchronization inside the burst | **none** — no sync, no events, no pacing | confirmed absent repo-wide |
| Stream | legacy NULL stream by default (DCGM never calls `cublasSetStream`) | DiagnosticPlugin.cpp:1353 |
| A/B | one pair per precision, reused by every GEMM; only C advances | DiagnosticPlugin.cpp:1249-1324 |
| A/B fill | `srand(10)`, values in `[0,10)`, FP32/FP16 downcast from the FP64 draw | DiagnosticPlugin.h:583-605 |
| C buffer count | derived from 90% of free VRAM (`USEMEM`) | DiagnosticPlugin.cpp:1046 |
| C aliasing | `baseIterations` FP64-sized buffers; FP32 aliases 2 per buffer, FP16 aliases 4 | DiagnosticPlugin.cpp:1082-1106 |
| Iterations per precision | `×4` half, `×2` single, `×1` double | DiagnosticPlugin.cpp:1390-1401 |
| `matrix_dim` default | 2048 | PluginStrings.h:322, DiagnosticPlugin.cpp:111 |
| Default precisions | `half,single` (DCGM drops double when the FP64:FP32 ratio is poor, which excludes every consumer part) | DiagnosticPlugin.cpp:172-199 |
| Tensor cores | `cublasSetMathMode(CUBLAS_TENSOR_OP_MATH)` at the halfway mark, not from the start | DiagnosticPlugin.cpp:1432-1439 |
| Result check | grid `(64,64)` block `(32,8)`, ε=1e-3 (FP32/FP16) / 1e-7 (FP64), warp-shuffle reduce then one atomic per warp | compare.cu, DiagnosticPlugin.cpp:1157-1188 |
| GFLOPS accounting | DCGM's `OPS_PER_2048_MUL` scaled by `(dim/2048)³` | DiagnosticPlugin.cpp:61-77 |

Two things were deliberately **not** copied:

- **`gpu-burn`'s kernel launch path.** It loads `compare.fatbin` at runtime and
  launches via the pre-CUDA-4.0 driver API (`cuParamSetv`, `cuLaunchGridAsync`,
  `cuFuncSetBlockShape`, behind `#define CUDA_ENABLE_DEPRECATED`). Those entry
  points are gone in CUDA 13. The compare kernels here are ordinary `__global__`
  functions in the same translation unit, so there is no sidecar file to ship
  and nothing to JIT.
- **`nccl-tests`' MPI dependency.** nccl-tests needs `MPI=1` for multi-process
  (`src/Makefile:8`). pcieburn forks its own ranks and brokers the
  `ncclUniqueId` over pipes, so there is no `mpirun` and no MPI install needed.

## Correctness details that are easy to get wrong

These are the non-obvious parts, recorded because they are load-bearing:

1. **The supervisor never initializes CUDA.** A CUDA context does not survive
   `fork()`, so the parent must fork before any context exists. Device count is
   probed in a separate short-lived child.
2. **Ranks must agree on the iteration count.** It is derived from free VRAM,
   which differs per GPU (a display attached to one card is enough to skew it).
   If ranks disagree they issue *different numbers of collectives* and the job
   deadlocks. `baseIterations` is reduced with `ncclAllReduce`/`ncclMin` before
   any C buffer is allocated.
3. **Shutdown is coordinated through a collective.** Each rank contributes a
   "want to stop" flag and takes the max, so every rank leaves the loop on the
   same iteration. Without this, SIGTERM landing at slightly different times
   would leave ranks issuing mismatched collectives — a guaranteed hang on the
   way out of an otherwise clean run.
4. **NCCL buffers and the communicator are created before the GEMM working set.**
   `gpu-burn` grabs 90% of VRAM in a single allocation, leaving nothing for
   NCCL. Building the comm first means `cudaMemGetInfo` already accounts for
   NCCL's internal channel buffers, with no guessing at the overhead.
5. **Nothing blocks in `cudaStreamSynchronize`.** Completion is detected by
   polling `cudaStreamQuery` plus `ncclCommGetAsyncError` with a timeout, then
   `ncclCommAbort` — the pattern from nccl-tests' `testStreamSynchronize`
   (`src/common.cu:482`). This is the whole reason a GPU falling off the bus
   produces a timestamped log line instead of a silently wedged process.
6. **One lost rank takes down the group.** A rank that stops participating hangs
   every other rank, so the supervisor kills the whole group on the first loss
   rather than waiting. gpu-burn's "any child may die, keep going" model is
   incompatible with collectives.
7. **Every host/device scalar transfer goes through pinned memory.**
   `cudaMemcpyAsync` to or from *pageable* host memory is synchronous with
   respect to the host, so a pageable readback blocks inside the copy until the
   whole enqueued stream drains. Control would never reach the poll loop,
   `--coll-timeout` would never arm, and a dead GPU would wedge the process in
   `cudaMemcpyAsync` — silently defeating point 5 while looking correct. This
   was a real bug, caught in review before the first fault run.
8. **Log lines are emitted with a single `write()`.** All ranks share one stderr,
   and `fprintf` on an unbuffered stream is not guaranteed to be one syscall.
   The first smoke run tore lines apart mid-field (`...nan=` on one line, `0` on
   the next). A torn line at the instant of a fault would corrupt the one
   artifact that timestamps it.
9. **The supervisor's deadlines start when all ranks are ready, not at fork.**
   Startup (NCCL init plus ~26 GiB of per-buffer allocations per rank) takes
   several seconds and is not bounded by the same timer as the load phase.
   Anchoring the watchdog and overrun backstop at fork time would let a slow
   startup produce a false `SUSPECTED HANG` on a perfectly healthy run — the one
   result that would poison this investigation. Startup gets its own separate
   300 s deadline.

## Build

Nothing is hardcoded from a local probe. Check what the build resolved first:

```sh
make preflight
```

It prints the CUDA path and version, the discovered `NCCL_HOME` and NCCL
version, the target arch, and the driver version. Then:

```sh
make                              # defaults: CUDA_HOME=/usr/local/cuda, sm_120
make NCCL_HOME=/opt/nccl          # explicit NCCL location
make CUDA_HOME=/usr/local/cuda-13.0
make SM=90                        # a different GPU generation
```

Target toolchain for this investigation: **driver 580.178.04, CUDA 13.0**, RTX
5090 → compute capability 12.0 → `sm_120`, with a `compute_120` PTX fallback.
CUDA 13 requires C++17, which the Makefile sets.

**On `CUBLAS_TENSOR_OP_MATH`:** that enumerator has been deprecated since CUDA
11, but it is still present and works in CUDA 13.3 — verified by an actual build.
If a future toolkit removes it, rebuild with
`make NVCCFLAGS_EXTRA=-DPCIEBURN_TENSOR_MATH=CUBLAS_TF32_TENSOR_OP_MATH` rather
than editing the source. Do **not** substitute `CUBLAS_DEFAULT_MATH`: that would
silently turn the halfway tensor-core step into a no-op, changing the load
profile mid-run while still appearing to work.

## Run

Always via the wrapper, which creates a self-contained run directory:

```sh
./run_pcieburn.sh --duration 180 --with-nvml --with-dmon --with-aer --with-psu
```

Use `--with-aer` — it is the only graded measurement in the set. Use
`--with-psu` too: on this platform it is the only *trustworthy* power channel.
Both are explained below.

The wrapper gates on an interactive safety confirmation (BIOS settings, test-node
status), snapshots provenance, optionally starts an NVML trace, runs the test,
then diffs `dmesg` and greps the delta for Xid/AER/DPC activity.

```
runs/20260811T2115Z-<tag>/
  manifest.txt          provenance, topology + NUMA, PCIe link capability, verdict
  pcieburn.log          timestamped console output
  events.csv            per-rank event log, for telemetry correlation
  nvml_trace.csv        per-GPU NVML trace incl. PCIe link gen/width (--with-nvml)
  pcie_dmon.txt         PCIe rx/tx throughput, independent of pcieburn's own
                        byte accounting (--with-dmon)
  pcie_link_states.txt  distinct PCIe link states observed, with sample counts
  pcie_link_baseline.csv  link gen/width per GPU before any load, via NVML
  pcie_link_rootports.txt   root-port link state before load (authoritative)
  pcie_link_rootports.csv   the same, machine-readable
  pcie_link_rootports_after.txt  re-read after the run; a link that was at
                        capability before and below it after is a strong signal,
                        and the wrapper flags the difference on the console
  aer_counters.csv      raw AER correctable counters, per GPU and root port
  aer_delta.txt         per-link error accumulation over the run (--with-aer)
  aer_uncorrectable.csv NONZERO uncorrectable/fatal AER rows only — names the
                        actual fatal error; empty on a healthy run
  aer_baseline.txt      complete starting state of all three AER registers
  psu_current.csv       per-PSU output current + derived watts (--with-psu)
  psu_summary.txt       min/max/mean current, peak swing, % of PSU rating
  dmesg_delta.txt       only this run's kernel messages
  faults.txt            Xid / AER / DPC / "fallen off the bus" lines, if any
```

### `aer_delta.txt` — the graded measurement

Every other artifact here is binary: the run either faulted or it didn't. That
makes experiments expensive, because each reproduction costs a node reboot, and
time-to-fault has enough variance that comparing two configurations needs several
trials each.

AER correctable counters break that. The kernel exposes cumulative per-device
counts in `/sys/bus/pci/devices/<bdf>/aer_dev_correctable`, and link errors
accumulate *before* anything fatal happens — in the first reproduction on this
node the `REPLAY_NUM Rollover` storm preceded the fatal event by ~1.9 s. So the
delta over a run ranks the eight links against each other whether or not
anything failed:

```
role     gpu bdf           RxErr  BadTLP  BadDLLP  Rollover  Timeout
dev      0   0000:01:00.0      0       2        0       138        0
rootport 0   0000:00:01.1      0       0        0       131        0
dev      5   0000:a1:00.0      0       0        0        67        0
dev      3   0000:61:00.0      0       0        0         0        0
```

Sorted by `Rollover`, which counts data-link-layer TLP retries — the signature of
marginal signal integrity rather than a logical or thermal problem. A clean run
where two links accumulate Rollovers and six do not is a *result*, and it costs
no reboot.

### `aer_uncorrectable.csv` — which fatal error, and the Surprise-Down question

The correctable counters say the link is struggling. The **uncorrectable** register
says what finally killed it, and that distinction discriminates between competing
mechanisms:

| Field | Meaning |
|---|---|
| `SDES` | **Surprise Down** — the link dropped with no warning, i.e. the device stopped responding |
| `DLP` | Data Link Protocol error |
| `RxOF` | Receiver overflow |
| `MalfTLP` | Malformed TLP |
| `CmpltTO` | Completion timeout |

A **power-brownout** story predicts `SDES`: the GPU momentarily vanishes and the
link surprise-drops. A **margin-erosion** story predicts no `SDES` — correctable
errors accumulate, replays exhaust, and the device signals `ERR_FATAL` over a
still-working link. Both reproductions on `cptcor04` logged
`DPC: containment event, ERR_FATAL received from 0000:01:00.0` with no Surprise
Down anywhere, which favours the latter — but until this file existed we could
not see which uncorrectable bit the device actually set.

**Why it must be sampled during the run:** by the time the kernel tries to read
the device's AER state, DPC has already contained the link, which is exactly why
the log says `can't recover (no error_detected callback)`. A post-mortem cannot
recover it.

Only nonzero rows are written, so the file is empty on a healthy run and any
content at all is significant. The field set is not hardcoded — it is read from
whatever the kernel exposes, since it varies by version. `aer_baseline.txt` holds
the complete starting state of all three registers for reference.

`pcie_link_states.txt` exists to catch downtraining — a link dropping below
Gen5 x16 is the step before falling off the bus entirely.

**Do not dismiss a low idle link speed as power saving on this platform.** An
earlier version of this README said exactly that, and it was wrong: all eight
root ports report `ASPM Disabled` with `Target Link Speed: 32GT/s`, so there is
no power-management mechanism that would legitimately drop a link to Gen1. A port
below its capability here is a hardware indicator. On `cptcor04` the GPU that
idles at 2.5GT/s while its seven identical peers hold 32GT/s is one of the two
that have historically fallen off the bus.

`linkcheck.sh` encodes that distinction and is the authoritative check — see
below.

Useful configurations:

```sh
# First pass: DCGM-faithful full bursts, allreduce between them.
./run_pcieburn.sh --duration 90 --with-nvml

# Inference-faithful interleave: a handful of GEMMs per collective.
./run_pcieburn.sh --duration 90 -- --gemms-per-coll 16

# Wide message sweep, since activation tensors are multi-megabyte.
./run_pcieburn.sh --duration 120 -- --coll-min 128M --coll-max 4G

# Long-duration / sustained-exposure mode.
./run_pcieburn.sh --duration 7200 --tag soak -- --gemms-per-coll 32

# Does the legacy-stream ordering matter? A/B it.
./run_pcieburn.sh --duration 90 -- --stream explicit

# Other collective shapes.
./run_pcieburn.sh --duration 90 -- --collective alltoall
./run_pcieburn.sh --duration 90 -- --collective sendrecv
```

Run `./pcieburn --help` for the full option list.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | all ranks completed cleanly |
| 1 | usage or setup error (no GPUs, bad arguments) |
| 2 | a rank reported compute faults or NaNs |
| 3 | a rank hung, was lost, or died — **the signature this test hunts** |

## Correlating against existing telemetry

`events.csv` timestamps are UTC ISO-8601 with milliseconds and a trailing `Z`
(`2026-08-11T21:15:03.412Z`) — byte-identical in format to
`telemetry/bmc_power_poll.py`, so the two join on `timestamp` with no
reformatting. Columns:

```
timestamp,tag,event,rank,outer_iter,gemms,colls,
coll_bytes_nominal,coll_bytes_algorithmic,coll_bytes_pcie_link,
faulty,nans,gflops,note
```

`event` is one of `start`, `ready`, `all_ready`, `progress`, `error`,
`watchdog`, `startup_timeout`, `killall`, `overrun`, `unkillable`, `rank_lost`,
`bad_message`, `waitpid_failed`, `done`, `peak`, `finish`. The interesting rows
for a reproduction are `watchdog` (a rank went silent — what a GPU falling off
the bus looks like from userspace) and `error` (a rank's own collective timeout
or NCCL async error fired first). Both carry the accumulated counters, so you can
read off how much traffic had moved at the moment it broke.

### Interpreting the three byte counters

They are cumulative per rank, not averages, and they differ by design:

| Column | What it is |
|---|---|
| `coll_bytes_nominal` | The collective buffer sizes, summed. What was *asked* for. |
| `coll_bytes_algorithmic` | Bytes actually leaving the rank in one direction. A ring allreduce moves `2(N-1)/N ×` the buffer (1.75× at 8 ranks); alltoall moves `(N-1)/N` since the self-share never leaves; sendrecv moves 1×. These are the same factors `nccl-tests` uses for `busBw`, so the figures are directly comparable to a `nccl-tests` run on the same fabric. |
| `coll_bytes_pcie_link` | Total bytes over that rank's own PCIe link, both directions. **This fleet has P2P disabled** (every pair `CNS`), so NCCL stages through host RAM and each remote byte crosses PCIe twice — up the sender's link, down the receiver's. Hence `2 × algorithmic`. |

At 8 ranks doing allreduce, the nominal figure understates real link-level work
by **3.5×**. The end-of-run summary prints all three per rank and in total, plus
the achieved egress rate as a percentage of Gen5 x16 (63 GB/s per direction,
a spec figure — not a measurement).

That percentage is the number to watch when choosing a configuration. In the
default `--gemms-per-coll 0` full-burst mode the GEMMs dominate the wall clock
and the bus sits idle between collectives, so link utilization is low. If the
fault turns out to be driven by link activity — or by transitions in link
activity — rather than by compute alone, then a smaller `--gemms-per-coll` and a
larger `--coll-max` are the configurations that actually exercise it.

pcieburn does not touch persistence mode, clocks, or any global GPU state, and
does not assume exclusive access to monitoring. Run it alongside the existing
instrumentation:

```sh
# terminal 1 — BMC PSU/voltage
python3 ../../telemetry/bmc_power_poll.py --host 10.11.3.184 --user admin \
    --password '...' --interval 1 --output power_trace.csv

# terminal 2 — the test (NVML trace included)
./run_pcieburn.sh --duration 300 --with-nvml --tag combined-01

# rasdaemon is already running persistently; read AER events after the fact
ras-mc-ctl --errors
```

## `linkcheck.sh` — zero-cost link health screen

Run this on any GPU node, including ones you have no intention of stress-testing.
It reads each GPU's upstream **root port** config space and compares the
negotiated link speed and width against that port's own capability. No load, no
GPU state touched, no risk of taking a node down.

```sh
sudo ./linkcheck.sh              # table
sudo ./linkcheck.sh --csv        # machine-readable
sudo ./linkcheck.sh --with-index # add nvidia-smi indices (uses NVML)
watch -n 10 'sudo ./linkcheck.sh'
```

Exit status is 0 when every link is at capability, 1 when any is not — so it
drops straight into a fleet sweep.

Two design points that matter:

- **It reads the root port, not the GPU.** Reading an *endpoint's* config space
  requires the link to be in L0, so querying the GPU (or NVML) can wake a link
  out of a low-power state and perturb the very thing being measured. The root
  port's `LnkSta` describes the same link and is local to the host bridge. GPU
  enumeration is pure sysfs for the same reason.
- **The verdict depends on ASPM**, because that is what makes a low speed
  interpretable:

| Verdict | Meaning |
|---|---|
| `DEGRADED` | below capability **while ASPM is disabled** — no power-management explanation exists, treat as a hardware indicator |
| `LOW` | below capability but ASPM is enabled, so it may be legitimate; re-check under load |
| `OK` | at full capability |

### Boards with a PCIe switch

Some machines in this fleet use a switchboard instead of risers, which inserts a
hop:

```
riser board:      root port ─────────────────────────────► GPU
switch board:     root port ─► switch upstream ─┬─► downstream ─► GPU
                                                └─► downstream ─► GPU  ...
```

Three consequences, all of which the tooling now accounts for:

- **`linkcheck.sh` and the AER poller walk the whole path**, not just the GPU's
  immediate parent. On a switch board that parent is the *downstream* port, so
  looking one hop up would leave the root-port↔switch link unmonitored — and
  that link carries every GPU behind the switch.
- **A containment event upstream takes down every GPU behind the switch.** The
  runbook records that "exactly one GPU fails while the other seven hold" — that
  invariant is a property of the riser topology and should *not* be expected
  here. Multiple GPUs dropping together on a switch board is ordinary DPC scope,
  not a new phenomenon.
- **P2P may be available** between GPUs under the same switch, so NCCL can route
  peer traffic directly instead of staging through host RAM. The wrapper detects
  this from `nvidia-smi topo -p2p r` and says so. Per-rank link byte totals are
  unaffected — a rank's own link still carries egress plus ingress either way —
  but host DRAM traffic and shared-upstream-link load differ, so communication
  behaviour is not directly comparable against a P2P-disabled node.

Each link is judged against **its own** capability, so a switch that is
legitimately Gen4, or a bifurcated x8 slot, reports `OK` rather than `DEGRADED`.

A degraded link can be restored without rebooting, by writing the Retrain Link
bit on the port above it:

```sh
sudo setpci -s 00:01.1 CAP_EXP+10.w=0020:0020
```

On `cptcor04` that brings the link straight back to 32GT/s and it becomes
register-identical to its healthy peers. **Whether it then holds that speed while
idle is the actual diagnostic** — a link that retrains fine but will not stay up
is the signature seen here.

## Power instrumentation on this platform — which sensors to trust

Established by measurement against NVML (~3.94 kW of GPU draw under load) on
`cptcor04`, a 4+1 × 1600 W CRPS chassis. **Every watt-denominated sensor is
range-limited and unusable above idle:**

| Sensor | Multiplier | 8-bit ceiling | Behaviour |
|---|---|---|---|
| `PWR_PSU*_PIN` | 1 W | 255 W | **wraps** — printed 3 W at load onset |
| `PWR_PSU*_POUT` | 2 W | 510 W | **saturates** — never exceeds ~440 W |
| `CurConsumedWatts` / DCMI | derived from `PIN` | 255 W | garbage above idle |
| `CUR_PSU*_IOUT` | 0.6 A | ~153 A | **works** — 13.2 A idle → 81.6 A load |

At idle the watt sensors are exactly right, which is what made this hard to
spot. Three independent checks confirm it: `IOUT × VOLT_12V` = `POUT` exactly
(13.20 A × 12.3 V = 162 W), `POUT/PIN` = 86% (a credible CRPS efficiency), and
`PIN1 + PIN2` = the DCMI system total exactly. Nothing is miscalibrated — the
watt sensors simply cannot count that high.

So use current, not watts. `--with-psu` records `CUR_PSU1/2_IOUT` and derives
watts from the measured 12 V rail. `telemetry/bmc_power_poll.py` gained the same
columns (`psu*_iout_a`, `psu*_pout_calc_w`, `est_system_w`), appended so older
traces stay column-compatible.

Two caveats worth carrying forward:

- **Only 2 of the 5 CRPS units are instrumented at all** — no interface exposes
  the others. `est_system_w` extrapolates from the instrumented pair assuming
  even sharing across `--active-supplies` (default 4). Measured peak of 81.6 A
  per supply implies 4-way sharing: 4 × ~1004 W ≈ 4.0 kW, against NVML's 4.2 kW
  total. Five-way sharing would predict 68 A, which is 20% off. It is an
  estimate, not a measurement.
- **Steady-state loading is comfortable:** ~63% of the 1600 W rating per supply,
  ~65% of the 6400 W redundant capacity. Capacity is not the problem.

## The power-transient hypothesis, and how to test it

The most promising open hypothesis, which fits the evidence better than "PCIe
traffic causes it":

`IOUT` swings 42 → 81.6 A per instrumented supply — a ~2 kW load step across
four active supplies. And **the collective is a barrier, so all 8 GPUs enter
their GEMM burst simultaneously**, making their power transients *correlated*
rather than averaging out. Correlated sub-millisecond excursions across eight
575 W-class cards is close to the worst case for PSU transient response, and rail
droop costs PCIe link margin — which is exactly what `REPLAY_NUM Rollover`
(data-link-layer retries) indicates.

It accounts for every existing observation: `diag` reproduces (8 GPUs, unpaced
GEMMs, maximal correlated transients); `pcie`/`memory_bandwidth`/`nccl-tests`
never do (low compute power, small transients); pcieburn reproduced at only
**2.4%** average bus utilization; and only ~2 of 8 GPUs ever fail, those being
the slots with the least margin.

Two experiments discriminate it:

```sh
# 1. Cap power, hold traffic constant. Needs no code change.
nvidia-smi -q -d POWER | grep -i 'power limit'    # record the default first
nvidia-smi -pl 400
./run_pcieburn.sh --duration 180 --with-nvml --with-aer --with-psu --tag pl400
nvidia-smi -pl <default>                           # restore

# 2. De-correlate the ranks so their peaks do not align.
./run_pcieburn.sh --duration 180 --with-aer --with-psu --tag stagger \
    -- --rank-stagger 5
```

If either stops faulting — or measurably lowers the `Rollover` rate — while
still moving the same collective bytes, power transients are implicated and PCIe
traffic is exonerated.

Note on #1: a lower cap lowers clocks, so GEMM bursts lengthen and collectives
occur less often per second. Drop `--gemms-per-coll` proportionally to hold the
collective rate constant.

Note on #2: `--rank-stagger` is **deliberately less faithful**. Real
tensor-parallel inference genuinely is synchronized, so a stagger is a diagnostic
manipulation, not a fix. If it helps, the remedy is electrical or platform-level,
not software.

## Comparing runs — the `dataviz` repo

`plot_run.gp` above shows one run at a time. To put **several** runs on one time
axis — with a value readout under the pointer, draggable oscilloscope cursors and
a delta between them, and zoom to an arbitrary region — use the separate
`dataviz` repo, normally checked out alongside this one:

```
../dataviz/runviz --preset "Fault forensics" runs/<a> runs/<b> --xmode fault
../dataviz/runviz --list                       # inventory runs/
../dataviz/runviz --png /tmp/run.png ...       # no display needed
```

It finds `runs/` on its own (or takes `--root` / `$PCIEBURN_RUNS`). Its
`--xmode fault` puts each run's `rank_lost` event at t=0, which is what makes two
reproductions' final seconds directly comparable.

It is a separate repo on purpose: this one is a CUDA harness that has to build and
run on a GPU node, that one is pure Python that runs on a laptop against
copied-off artefacts. They share no code and no build.

Worth knowing when reading its output: it **blanks** NVML `power`/`clock`/`temp`
from the moment a GPU's link width stops reading as a number, because those fields
keep returning their last value indefinitely after a GPU falls off the bus. It
takes that onset from the first non-numeric width rather than the first
`[GPU is lost]` — in the cptrgca17 reproduction the frozen values start 420 ms
earlier than that string appears. See `../dataviz/README.md` for the full list of
things it refuses to draw.

## Safety

This test exists to provoke a fault that has previously required a **hard power
cycle** — the graceful shutdown path has hung waiting on a broken GPU's modeset
handshake. Expect that a successful reproduction may crash a GPU or wedge the
node.

- Run only on a **designated test node**, never anything customer-facing.
- Confirm the runbook BIOS settings first, so a platform difference doesn't
  confound the result: `PCIE Link Speed Capability: GEN5`,
  `Multi Upstream Auto Speed Change: Enabled`.
- The wrapper prompts for both of these; `--yes` skips the prompt for automation.
- On a clean exit no orphaned processes or CUDA contexts are left: ranks tear
  down their comms, free their buffers, and exit; the supervisor reaps every
  child, escalating SIGTERM → SIGKILL. If a rank survives SIGKILL it is stuck in
  the driver, and the log says so explicitly — that is the case where a power
  cycle is required.

## Status

**Built and smoke-tested on `cptcor04`.** Toolchain: CUDA 13.3, NCCL 2.30.7,
driver 580.178.04, 8× RTX 5090 at `sm_120`. It compiles clean with no warnings.

First smoke run (`--duration 20 --tag smoke`), all 8 ranks:

| | |
|---|---|
| C buffers per rank | 820 (25.62 GiB), 2.78 GiB VRAM left free |
| Iterations | half 3280 / single 1640 per pass |
| Startup | ~7.6 s to all-ranks-ready |
| Total | 1,180,800 GEMMs, 480 collectives, 225 GiB moved |
| FP16 | ~245 TFLOP/s |
| FP32 | ~45.9 TFLOP/s, rising to ~63.7 TFLOP/s when tensor-op math engages at the halfway mark |
| Verdict | clean, exit 0, no faults, no NaNs, no orphaned processes |

The FP32 step from 45.9 to 63.7 TFLOP/s at the halfway point is the
`cublasSetMathMode` transition working as intended, and matches DCGM's behaviour
of pulling in tensor cores partway through rather than from the start.

**Not yet done:** no fault has been provoked. The smoke run only establishes that
the harness works and loads the hardware as intended. The real attempts —
duration sweeps, `--gemms-per-coll` granularity sweeps, `alltoall`/`sendrecv`,
and long-duration soak — are the actual experiment.

## Licensing

The compute loop and compare kernels are reimplementations derived from DCGM
(`nvvs/plugin_src/diagnostic/`, Apache-2.0) and gpu-burn (BSD-2-Clause). The
NCCL bootstrap pattern follows nccl-tests (BSD-3-Clause). All three are
permissively licensed and freely reusable; retain this note if the harness is
distributed.
