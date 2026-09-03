# pcieburn

A combined compute + PCIe/NCCL stress test for single-node multi-GPU systems.

## What it does

Neither a pure compute burner (`gpu-burn`, DCGM's `diagnostic`) nor a pure
communication benchmark (`nccl-tests`) loads a machine the way tensor-parallel
inference does: a dense GEMM burst, then a collective where every GPU
synchronizes, then compute again. `pcieburn` runs that interleaved shape.

It forks **one process per GPU** (a supervisor coordinates startup and teardown
but never touches CUDA itself). Each rank loops:

1. An unpaced `cublasXgemm` burst — no synchronization between calls, matching
   DCGM's `GpuBurnWorker::Compute()` and `gpu-burn`'s inner loop. Default
   2048-dim, cycling half/single precision, C matrices sized to fill ~90% of
   free VRAM. Tensor-op math is enabled at the halfway mark, as DCGM does.
2. A real NCCL collective (`allreduce` by default) issued **on the same CUDA
   stream**, so it is genuinely ordered behind the compute rather than merely
   running concurrently. Collective size sweeps 128M → 1G by default.
3. A result comparison (faulty/NaN counts) and a coordinated stop vote via
   `ncclAllReduce`, so every rank leaves the loop on the same iteration and
   teardown cannot deadlock.

Rendezvous is a fork + pipe exchange of the `ncclUniqueId` — no MPI required.
Ranks report progress to the supervisor over pipes; a per-rank stream timeout
and a supervisor watchdog turn a hung GPU into a fast, loud exit instead of an
indefinite stall.

The wrapper `run_pcieburn.sh` runs the binary inside a self-contained,
timestamped run directory with kernel-log, NVML, PCIe-AER and PSU telemetry
captured around it.

`linkcheck.sh` is a separate, zero-load screen that walks each GPU's full PCIe
path to the root complex and reports negotiated speed/width against each port's
own capability.

> **Safety:** this test is designed to provoke a PCIe link fault that has
> previously required a hard power cycle to recover from. Run it only on a
> designated test node.

## Requirements

| | |
|---|---|
| CUDA toolkit | `nvcc` + cuBLAS + CUDA runtime. C++17, so CUDA 11 or newer. |
| NCCL | `nccl.h` and `libnccl.so` (`libnccl-dev` on Debian/Ubuntu). |
| Driver | Any driver matching the toolkit. Developed against 580.178.04 / CUDA 13. |
| GPUs | 2 or more visible devices. |

The wrapper additionally uses, all optional and each self-disabling with a
warning if unavailable: `nvidia-smi` (NVML trace, `dmon` throughput), readable
`dmesg` (`sysctl kernel.dmesg_restrict=0`, or passwordless `sudo`), `sudo
ipmitool` (BMC PSU sensors, PMBus bridge), and `psu_pmbus_poll.py` alongside
the script.

## Build

```sh
make preflight     # print the resolved toolchain and exit — no compile
make               # build ./pcieburn
```

`preflight` reports the CUDA and NCCL it found, the target arch, and the driver
version. The build fails early with an actionable message if `nvcc` or `nccl.h`
is missing, rather than deep inside a compile.

Everything is overridable:

```sh
make CUDA_HOME=/usr/local/cuda-13.0
make NCCL_HOME=/opt/nccl NCCL_LIBDIR=/usr/lib/x86_64-linux-gnu
make SM=90                                   # default 120 (RTX 5090, sm_120)
make NVCCFLAGS_EXTRA=-DPCIEBURN_TENSOR_MATH=CUBLAS_DEFAULT_MATH
```

`make clean` removes the binary.

## Run

The wrapper is the normal entry point. It creates
`runs/<UTC-timestamp>[-tag]/`, starts every telemetry collector, runs the load,
holds the collectors through a settle window, then writes a manifest and
verdict.

```sh
./run_pcieburn.sh --duration 90 --tag baseline
./run_pcieburn.sh --duration 300 -- --gemms-per-coll 16
./run_pcieburn.sh --outdir /data/runs --tag alltoall -- --collective alltoall
```

Anything after `--` is passed to the binary verbatim. The wrapper prompts for
an interactive safety confirmation unless given `--yes`.

The binary also runs standalone:

```sh
./pcieburn --duration 60
./pcieburn --gpus 0,5 --duration 300 --precision half --event-log ev.csv
```

Set `PCIEBURN_BIN` to point the wrapper at a binary elsewhere.

### Wrapper options

| Option | Default | |
|---|---|---|
| `--outdir DIR` | `./runs` | parent directory for run dirs |
| `--tag NAME` | — | label used in the dir name and event log |
| `--duration SEC` | — | forwarded to `pcieburn`, also sizes the NVML trace |
| `--settle SEC` | `30` | keep collectors running this long after the load, before the post-run kernel snapshot |
| `--yes`, `-y` | off | skip the safety confirmation |
| `--nvml-interval MS` | `100` | NVML sample interval |
| `--aer-interval SEC` | `1` | AER counter poll interval |
| `--psu-interval SEC` | `0.25` | PSU poll interval, both channels |
| `--psu-pmbus-interval SEC` | `0.55` | PMBus channel only. Its round is 16 serial reads (4 supplies × VOUT/IOUT/POUT/PIN) and floors at ~0.48 s |
| `--psu-pmbus-cmds "LIST"` | `"0x8b 0x8c 0x96 0x97"` | PMBus registers to poll. A narrower round samples faster: `"0x8b 0x8c"` is rail voltage and current at 4 Hz, for chasing a 12 V transient rather than measuring an envelope |
| `--active-supplies N` | `4` | load-sharing PSU count, for the estimated-system-power column only |
| `--psu-rating W` | `1600` | per-supply rating, for the %-of-rating column |

Every collector is **on by default**. Turn one off with `--no-nvml`,
`--no-dmon`, `--no-aer`, `--no-psu` (both PSU channels), `--no-psu-bmc`,
`--no-psu-pmbus`, or `--no-telemetry` for load only. The manifest records which
collectors actually ran, so a missing CSV is never ambiguous. Note the
degraded-link verdict is computed from the NVML trace, so `--no-nvml` disables
it. The `--with-*` forms are accepted and ignored, for older command lines.

### `pcieburn` options

| Option | Default | |
|---|---|---|
| `--duration SEC` | `60` | run length |
| `--matrix-dim N` | `2048` | GEMM dimension (DCGM's default) |
| `--precision LIST` | `half,single` | comma list of `half,single,double` |
| `--gemms-per-coll N` | `0` | GEMMs between collectives; `0` = one full burst per collective. 8–64 is closer to a real transformer layer |
| `--collective NAME` | `allreduce` | `allreduce`, `alltoall`, `sendrecv` |
| `--coll-min SIZE` | `128M` | smallest collective (binary suffixes) |
| `--coll-max SIZE` | `1G` | largest; send and recv buffers are both this size, so 2× VRAM |
| `--coll-factor N` | `2` | sweep multiplier |
| `--mem-frac F` | `0.9` | fraction of remaining VRAM for C matrices |
| `--max-c-buffers N` | `0` | cap the C buffer count (`0` = unlimited) |
| `--gpus LIST` | all visible | comma list of device indices |
| `--stream MODE` | `legacy` | `legacy` (as DCGM/gpu-burn) or `explicit` |
| `--rank-stagger MS` | `0` | delay rank N's burst by N×MS each pass. Diagnostic only — real inference is synchronized, so nonzero is *less* faithful |
| `--always-tensor` | off | tensor-op math from the start, not at halfway |
| `--no-tensor` | off | never enable tensor-op math. With `--precision single --matrix-dim 8192` this builds a gpu-burn-equivalent arm |
| `--no-compare` | off | skip result verification (pure load) |
| `--coll-timeout SEC` | `120` | per-rank hang timeout (`0` = off) |
| `--watchdog SEC` | `60` | supervisor silence timeout (`0` = off) |
| `--report-interval SEC` | `1.0` | *console* cadence only; the event log always records every pass |
| `--event-log PATH` | — | append a CSV event log for telemetry correlation |
| `--tag NAME` | — | label recorded in the event log |

### Exit status

Both the binary and the wrapper use the same codes; the wrapper adds two.

| | |
|---|---|
| `0` | clean |
| `1` | setup/usage error |
| `2` | compute faults — nonzero faulty/NaN counts |
| `3` | rank lost or hung during load |
| `4` | *(wrapper)* load completed, then a fatal PCIe containment during the settle window |
| `5` | *(wrapper)* clean but a link ran degraded — not comparable against full-width runs |

## Run artifacts

Written to `runs/<timestamp>[-tag]/`:

| File | |
|---|---|
| `manifest.txt` | provenance, topology and link baseline, which collectors ran, verdict |
| `pcieburn.log` | timestamped console output and the final summary block |
| `events.csv` | per-rank event log: `start, ready, all_ready, progress, peak, rank_lost, finish, killall` — one `progress` row per pass, per rank |
| `nvml_trace.csv` | per-GPU power, clocks, temp, util, PCIe link gen/width |
| `pcie_dmon.txt` | `nvidia-smi dmon` PCIe rx/tx throughput, an independent cross-check of the tool's own byte accounting |
| `pcie_link_baseline.csv`, `pcie_link_states.txt`, `pcie_link_states_load.txt` | link gen/width before the run, and the distinct states observed with sample counts |
| `pcie_link_rootports*.txt` / `.csv` | root-port link state before and after |
| `aer_counters.csv`, `aer_delta.txt`, `aer_uncorrectable.csv`, `aer_baseline.txt` | per-device and per-root-port AER counters, and the run's delta |
| `psu_current.csv`, `psu_pmbus.csv`, `psu_summary.txt` | BMC PSU output current; PMBus per-supply 12 V rail, output current and input/output power for all four supplies, plus `vout_spread_v` (the informative aggregate for four supplies on one bus) and `psuN_vout_derived_v` (POUT/IOUT, which cross-checks the rail's LINEAR16 exponent); min/max/mean summary |
| `dmesg_before.txt`, `dmesg_after.txt`, `dmesg_delta.txt`, `faults.txt` | kernel log snapshots, their diff, and the PCIe/Xid/AER lines from it |

Timestamps are UTC with milliseconds (`YYYY-MM-DDTHH:MM:SS.mmmZ`) throughout,
matching the existing BMC/NVML pollers so traces join without reformatting.

## linkcheck.sh

```sh
sudo ./linkcheck.sh                  # table
sudo ./linkcheck.sh --csv            # machine-readable
sudo ./linkcheck.sh --with-index     # add nvidia-smi indices (uses NVML)
watch -n 10 'sudo ./linkcheck.sh'
```

Runs no load and touches no GPU state. Each link is read from its
downstream-facing port and compared against that port's own capability, so a
legitimately Gen4 switch or a bifurcated x8 slot reads `OK`. Verdicts:
`DEGRADED` (below capability with ASPM disabled — no power-management
explanation), `LOW` (below capability but ASPM enabled; re-check under load),
`OK`. Exits `0` all OK, `1` any DEGRADED or LOW, `2` could not run.

## Provenance

The GEMM loop and its parameters are adapted from DCGM's
`nvvs/plugin_src/diagnostic/DiagnosticPlugin.cpp` (Apache-2.0) and `gpu-burn`'s
`gpu_burn-drv.cpp` / `compare.cu` (BSD-2-Clause).
