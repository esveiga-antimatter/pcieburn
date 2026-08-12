# pcieburn — interleaved compute + PCIe/NCCL stress test

Combined GEMM-burst and NCCL-collective load generator, built to close the gap
between DCGM's `diagnostic` sub-test (which hammers compute and moves no data)
and `nccl-tests` (which moves data and barely touches compute).

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
./run_pcieburn.sh --duration 180 --with-nvml --with-dmon --with-aer
```

Use `--with-aer`. It is the only graded measurement in the set — see below.

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
  pcie_link_baseline.csv  link gen/width per GPU before any load
  aer_counters.csv      raw AER correctable counters, per GPU and root port
  aer_delta.txt         per-link error accumulation over the run (--with-aer)
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

`pcie_link_states.txt` exists to catch downtraining — a link dropping below
Gen5 x16 under load is the step before falling off the bus entirely. Read it
with one caveat: consumer cards legitimately downtrain to Gen1 when idle, and
the trace spans the idle periods either side of the load phase, so a Gen1 sample
is only meaningful if its timestamp falls inside the load window.

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
