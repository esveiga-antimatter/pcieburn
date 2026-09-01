#!/usr/bin/env bash
#
# run_pcieburn.sh — wrapper that puts a pcieburn run into a self-contained,
# correlatable run directory.
#
# Deliberately does NOT touch persistence mode, clocks, or any global GPU state,
# and does not assume exclusive access to monitoring: bmc_power_poll.py,
# rasdaemon, and any existing nvidia-smi trace can all run alongside this.
#
# Usage:
#   ./run_pcieburn.sh --duration 90
#   ./run_pcieburn.sh --duration 300 -- --gemms-per-coll 16
#   ./run_pcieburn.sh --outdir /data/runs --tag gen5-retest -- --collective alltoall
#
# Every telemetry collector is ON by default: an artifact bundle should be
# complete without the operator having to remember a flag for each channel.
# Turn one off with its --no-* flag (--no-dmon, --no-psu, ...) when a platform
# cannot support it. Each collector also self-disables with a warning when its
# data source is unavailable, and the manifest records which ones actually ran.
#
# Anything after `--` is passed straight through to the pcieburn binary.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${PCIEBURN_BIN:-$HERE/pcieburn}"

OUTDIR_BASE="$HERE/runs"
TAG=""
# Telemetry defaults to ON. Every one of these was previously opt-in, which meant
# the completeness of a run bundle depended on the operator remembering five
# flags — and a bundle missing a channel could not be distinguished from a run
# where that channel had nothing to report. Opt out per channel with --no-*.
WITH_NVML=1
WITH_DMON=1
WITH_AER=1
AER_INTERVAL=1
WITH_PSU=1
WITH_PSU_PMBUS=1
PSU_INTERVAL=0.25
ACTIVE_SUPPLIES=4
PSU_RATING_W=1600
NVML_INTERVAL_MS=100
SETTLE_SECONDS=30
DURATION=""
ASSUME_YES=0
RELAX_PCIE_ERRORS=0
declare -a PCIE_POLICY_UNDO=()
PASSTHRU=()

ts_iso() { date -u +"%Y-%m-%dT%H:%M:%S.%3NZ"; }
say()    { printf '[%s] run    %s\n' "$(ts_iso)" "$*"; }

usage() {
    cat <<'EOF'
run_pcieburn.sh [wrapper options] [-- pcieburn options]

Wrapper options:
  --outdir DIR        parent directory for run dirs (default ./runs)
  --tag NAME          label for this run, used in the dir name and event log
  --duration SEC      forwarded to pcieburn, and used to size the NVML trace
  --active-supplies N assumed number of load-sharing PSUs, used only for the
                      estimated system power column (default 4, from measured
                      4-way sharing on a 4+1 1600W CRPS chassis)
  --psu-rating W      per-supply rating, for the %-of-rating column (default
                      1600, matching this chassis's CRPS units)
  --settle SEC        keep every collector running and delay the post-run kernel
                      log snapshot by SEC seconds (default 30, 0 disables).
                      This exists because a containment event can land AFTER the
                      load phase ends: one real run reported "clean" with
                      exit_code 0 while taking a DPC/ERR_FATAL 8.8 s after the
                      load stopped and 5 s after the snapshot was taken. The
                      fault was invisible in that run's own artifacts and was
                      only recovered from the NEXT run's dmesg_before.
  --relax-pcie-errors before the CUDA section, rewrite PCIe error policy on
                      every port and endpoint in each GPU's path:
                        * DPC trigger OFF on downstream-facing ports, so an
                          ERR_FATAL is decoded by AER instead of being contained
                          before the causing bit can be read;
                        * Completion Timeout and Unexpected Completion demoted
                          from fatal to non-fatal in the severity register,
                          which is what the PCIe base spec specifies for both.
                      Originals are recorded in the manifest and restored on a
                      clean exit. OFF BY DEFAULT: it removes containment, so a
                      fault takes the host down harder than it otherwise would.
  --yes               skip the interactive safety confirmation
  -h, --help          this message

Telemetry: every collector below is ON by default, so a bundle is complete
without having to remember a flag for each channel. Each one also self-disables
with a warning if its data source is missing, and the manifest's "telemetry
collectors" section records which ones actually ran -- so an absent CSV is never
ambiguous between "not asked for" and "nothing to report".

  --no-nvml           skip the per-GPU nvidia-smi trace (power, clocks, temp,
                      util, and PCIe link gen/width). Note the degraded-link
                      verdict (exit 5) is computed from this trace, so a run
                      without it cannot be checked for silent downtraining.
  --nvml-interval MS  NVML sample interval (default 100)
  --no-dmon           skip the nvidia-smi dmon PCIe rx/tx throughput trace. It
                      is independent of pcieburn's own byte accounting, so it
                      cross-checks it. Reports '-' on some GeForce parts, in
                      which case the file is simply empty of numbers.
  --no-aer            skip the per-GPU and per-root-port AER error counters from
                      sysfs. Strongly discouraged: this is the only graded
                      measurement in the set. Link errors accumulate before
                      anything fatal happens, so it ranks configurations, and
                      the eight links against each other, without having to
                      reach a fault and reboot the node each time.
  --aer-interval SEC  AER poll interval (default 1)
  --no-psu            skip BOTH PSU channels, the BMC sensors and PMBus.
  --no-psu-bmc        skip the BMC current sensors, keep PMBus. Of the BMC's
                      channels only CUR_PSU*_IOUT has adequate range: PWR_*_PIN
                      wraps at 255 W, PWR_*_POUT saturates near 510 W, and DCMI
                      derives from PIN, so all three read garbage above idle.
  --no-psu-pmbus      skip PMBus, keep the BMC sensors. The two are
                      complementary: PMBus reads per-PSU input and output power
                      for all four supplies over the ASRock bridge via
                      psu_pmbus_poll.py (read-only commands, 4 Hz, confirmed on
                      hardware), while the BMC sees only two supplies and is the
                      only independent cross-check on those. Aggregate the PMBus
                      CSV with medians; the summary at the end does this for you.
  --no-telemetry      skip every collector above: load only, no measurement.
  --with-nvml, --with-dmon, --with-aer, --with-psu, --with-psu-pmbus
                      accepted and ignored -- these are on by default now. Kept
                      so existing command lines and runbook entries still work.

Everything after `--` goes to pcieburn verbatim. See ./pcieburn --help.

Exit status:
  0  clean
  2  COMPUTE FAULTS reported (nonzero faulty/nan counts)
  3  RANK LOST OR HUNG — a rank died or stopped responding during load
  4  FAULT AFTER LOAD PHASE — load completed, then a fatal PCIe containment
     appeared during the settle window. Same severity as 3; distinguished only
     because the load phase itself completed.
  5  CLEAN BUT LINK DEGRADED — nothing failed, but at least one GPU ran the
     whole run below its negotiated width, or dwelt far too long below Gen5.
     The run is not comparable against full-width runs; see the note it prints.
  1  setup/usage error

SAFETY: this test tries to reproduce a fault that has previously needed a hard
power cycle. Run it only on a designated test node.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --outdir)        OUTDIR_BASE="$2"; shift 2 ;;
        --tag)           TAG="$2"; shift 2 ;;
        --duration)      DURATION="$2"; shift 2 ;;
        # Accepted and ignored: these collectors are on by default now. Kept so
        # older command lines and runbook entries keep working rather than
        # aborting on an unknown option.
        --with-nvml|--with-dmon|--with-aer|--with-psu|--with-psu-pmbus)
                         shift ;;
        --no-nvml)       WITH_NVML=0; shift ;;
        --no-dmon)       WITH_DMON=0; shift ;;
        --no-aer)        WITH_AER=0; shift ;;
        --aer-interval)  AER_INTERVAL="$2"; shift 2 ;;
        --no-psu)        WITH_PSU=0; WITH_PSU_PMBUS=0; shift ;;
        --no-psu-bmc)    WITH_PSU=0; shift ;;
        --no-psu-pmbus)  WITH_PSU_PMBUS=0; shift ;;
        --no-telemetry)  WITH_NVML=0; WITH_DMON=0; WITH_AER=0
                         WITH_PSU=0; WITH_PSU_PMBUS=0; shift ;;
        --psu-interval)  PSU_INTERVAL="$2"; shift 2 ;;
        --active-supplies) ACTIVE_SUPPLIES="$2"; shift 2 ;;
        --psu-rating)    PSU_RATING_W="$2"; shift 2 ;;
        --nvml-interval) NVML_INTERVAL_MS="$2"; shift 2 ;;
        --settle)        SETTLE_SECONDS="$2"; shift 2 ;;
        --relax-pcie-errors) RELAX_PCIE_ERRORS=1; shift ;;
        --yes|-y)        ASSUME_YES=1; shift ;;
        -h|--help)       usage; exit 0 ;;
        --)              shift; PASSTHRU=("$@"); break ;;
        *)  echo "unknown wrapper option '$1' (use -- to pass options to pcieburn)" >&2
            exit 1 ;;
    esac
done

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: pcieburn binary not found at $BIN" >&2
    echo "       build it first:  make -C $HERE" >&2
    exit 1
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUNDIR="$OUTDIR_BASE/${STAMP}${TAG:+-$TAG}"
mkdir -p "$RUNDIR" || { echo "ERROR: cannot create $RUNDIR" >&2; exit 1; }

MANIFEST="$RUNDIR/manifest.txt"
CONSOLE="$RUNDIR/pcieburn.log"
EVENTS="$RUNDIR/events.csv"
NVML="$RUNDIR/nvml_trace.csv"
DMON="$RUNDIR/pcie_dmon.txt"
LINKSTATES="$RUNDIR/pcie_link_states.txt"
AER="$RUNDIR/aer_counters.csv"
AERDELTA="$RUNDIR/aer_delta.txt"
AER_UNCORR="$RUNDIR/aer_uncorrectable.csv"
AER_BASE="$RUNDIR/aer_baseline.txt"
PSUCSV="$RUNDIR/psu_current.csv"
PMBUSCSV="$RUNDIR/psu_pmbus.csv"
PSUSUM="$RUNDIR/psu_summary.txt"

# --- safety gate ----------------------------------------------------------
if [[ $ASSUME_YES -eq 0 ]]; then
    cat <<'EOF'

  ####################################################################
  #  pcieburn is designed to provoke a PCIe link fault that has       #
  #  previously required a HARD POWER CYCLE to recover from.          #
  #                                                                   #
  #  Confirm before continuing:                                       #
  #    - this is a designated TEST node, not customer-facing           #
  #    - BIOS: PCIE Link Speed Capability      = GEN5                 #
  #    - BIOS: Multi Upstream Auto Speed Change = Enabled              #
  ####################################################################

EOF
    read -r -p "Proceed? [y/N] " reply
    [[ "$reply" == "y" || "$reply" == "Y" ]] || { echo "aborted."; exit 1; }
fi

# --- uptime -----------------------------------------------------------------
# Captured explicitly rather than derived later. Reproduction has correlated with
# time since cold boot, and the only previous way to recover it was the first
# timestamp in dmesg_before.txt, which is unavailable when the kernel log is
# restricted.
UPTIME_S=$(awk '{printf "%d", $1}' /proc/uptime 2>/dev/null || echo 0)
UPTIME_H=$(printf '%dh %02dm %02ds' $((UPTIME_S/3600)) $(((UPTIME_S%3600)/60)) $((UPTIME_S%60)))
BOOT_UTC=$(date -u -d "@$(( $(date +%s) - UPTIME_S ))" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || echo unknown)
say "uptime at start: $UPTIME_H (booted $BOOT_UTC)"

# --- provenance -----------------------------------------------------------
{
    echo "pcieburn run manifest"
    echo "run_dir           : $RUNDIR"
    echo "tag               : ${TAG:-<none>}"
    echo "start_utc         : $(ts_iso)"
    echo "host              : $(hostname)"
    echo "kernel            : $(uname -r)"
    echo "binary            : $BIN"
    echo "pcieburn_args      : ${PASSTHRU[*]:-<defaults>}"
    [[ -n "$DURATION" ]] && echo "duration          : ${DURATION}s"
    # Uptime is recorded because it turned out to correlate with reproduction and
    # was previously only recoverable by digging the first timestamp out of
    # dmesg_before.txt — which fails entirely when the kernel log is restricted.
    # Bumped whenever this wrapper changes what it measures or how it decides a
    # verdict, so an artifact bundle is self-describing. v2 added the settle
    # window and made link degradation gate the verdict; a v1 bundle's "clean"
    # is a weaker claim than a v2 bundle's.
    # v3: the degraded-link verdict is computed from load-window samples only.
    # v2 counted the whole trace, so pre-load idle at gen1 (RGCA links idle at
    # gen1 and switch-node prep takes ~35 s) plus the settle window's post-load
    # idle exceeded the 20 s dwell limit on EVERY switch-node run — four uniform
    # all-8-GPU false DEGRADED verdicts before it was caught. Uniform dwell on
    # all GPUs is the signature of that artifact; real degradation is per-link.
    # v5: every telemetry collector is on by default, where each previously
    # needed its own --with-* flag. In a v4 bundle a missing nvml_trace.csv or
    # aer_delta.txt may just mean nobody asked for it; in a v5 bundle that
    # channel was either disabled with --no-* or self-disabled for want of a
    # data source, and the "telemetry collectors" section records which.
    echo "wrapper_version   : 5"
    echo "settle_seconds    : $SETTLE_SECONDS"
    echo "uptime_seconds    : $UPTIME_S"
    echo "uptime_human      : $UPTIME_H"
    echo "boot_utc          : $BOOT_UTC"
    echo
    echo "--- git ---"
    git -C "$HERE" rev-parse HEAD 2>/dev/null || echo "(not a git repo)"
    git -C "$HERE" status --porcelain 2>/dev/null | head -20
    echo
    echo "--- driver ---"
    head -1 /proc/driver/nvidia/version 2>/dev/null || echo "(no nvidia driver)"
    echo
    echo "--- nvidia-smi topo -p2p r ---"
    nvidia-smi topo -p2p r 2>&1 | head -30 || true
    echo
    echo "--- nvidia-smi -L ---"
    nvidia-smi -L 2>&1 || true
    echo
    echo "--- GPU power limits (an applied -pl persists for the whole boot"
    echo "    session, so a run that does not set it inherits the previous"
    echo "    value; recorded here so every run is self-auditing) ---"
    nvidia-smi --query-gpu=index,power.limit,power.default_limit,enforced.power.limit,power.min_limit,power.max_limit \
        --format=csv 2>&1 \
        || nvidia-smi -q -d POWER 2>&1 | grep -iE 'GPU 0000|Power Limit' \
        || true
    echo
    echo "--- GPU clocks (a locked clock via -lgc is a run variable, and the tag"
    echo "    is the only other place it is recorded; the 100ms NVML trace shows"
    echo "    empirically whether SM clock is pinned or still free-running) ---"
    nvidia-smi -q -d CLOCK 2>&1 \
        | grep -E '^GPU |Clocks|^ +(Graphics|SM|Memory|Video) ' | head -60 \
        || true
    echo
    echo "--- PCIe link capability vs current, via NVML (baseline before load) ---"
    nvidia-smi --query-gpu=index,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current \
        --format=csv 2>&1 || true
    echo
    echo "--- PCIe root-port link state (non-perturbing; see linkcheck.sh) ---"
    if [[ -x "$HERE/linkcheck.sh" ]]; then
        "$HERE/linkcheck.sh" 2>&1 || true
    else
        echo "(linkcheck.sh not found next to this script)"
    fi
    echo
    echo "--- NUMA affinity (cross-socket staging matters: P2P is disabled) ---"
    nvidia-smi topo -m 2>&1 | head -25 || true
} > "$MANIFEST"

# Baseline PCIe link check, before any load, from the root-port side. This is the
# authoritative view: reading the ENDPOINT's config space requires the link to be
# in L0, so an NVML or endpoint query can itself wake a link and perturb the
# measurement. The root port reports the same link and is local to the host bridge.
#
# On this platform ASPM is Disabled on every root port with Target Link Speed at
# maximum, so a link idling below its capability has NO power-management
# explanation — it is a hardware indicator, not normal power saving. linkcheck.sh
# encodes that distinction (DEGRADED vs LOW) and is the single source of truth.
BASELINE_LINK="$RUNDIR/pcie_link_baseline.csv"
ROOTPORT_LINK="$RUNDIR/pcie_link_rootports.txt"

# Keep the NVML view too, for comparison against the root-port view.
nvidia-smi --query-gpu=index,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current \
    --format=csv,noheader,nounits > "$BASELINE_LINK" 2>/dev/null || true

# Topology check. On a board with a PCIe switch, GPUs behind the same switch may
# be able to reach each other directly instead of staging through host RAM —
# which changes NCCL's transport, the traffic pattern, and how much of it crosses
# the shared upstream link. The runbook's premise that every byte is host-staged
# is specific to boards where P2P reports CNS on all pairs.
if nvidia-smi topo -p2p r 2>/dev/null | grep -qE '\bOK\b'; then
    say "NOTE: PCIe P2P is available between some GPU pairs on this host."
    say "    NCCL will route peer traffic directly rather than staging through"
    say "    host RAM, so the collective traffic pattern differs from a"
    say "    P2P-disabled node and the two are not directly comparable on the"
    say "    communication axis. Per-rank link byte totals are unaffected"
    say "    (a rank's own link still carries its egress plus its ingress),"
    say "    but host DRAM traffic and shared-upstream-link load will differ."
fi

if [[ -x "$HERE/linkcheck.sh" ]]; then
    "$HERE/linkcheck.sh" > "$ROOTPORT_LINK" 2>&1
    LINK_RC=$?
    "$HERE/linkcheck.sh" --csv > "$RUNDIR/pcie_link_rootports.csv" 2>/dev/null || true
    if [[ $LINK_RC -ne 0 ]]; then
        say "NOTE: PCIe link(s) below capability before any load:"
        while IFS= read -r l; do
            [[ "$l" =~ (DEGRADED|LOW)$ ]] && say "    $l"
        done < "$ROOTPORT_LINK"
        say "    DEGRADED means below capability while ASPM is DISABLED — there is"
        say "    no power-management explanation for it. On this platform that has"
        say "    marked the GPU that subsequently fell off the bus."
    fi
else
    say "WARNING: linkcheck.sh not found; skipping root-port link check"
fi

# Kernel log watermark, so the after-diff shows only this run's messages.
# dmesg is often restricted (kernel.dmesg_restrict=1), which silently produced
# an empty delta and no faults.txt on the first real run — the single most
# important artifact. Try unprivileged, then sudo -n, then journalctl.
capture_kernel_log() {
    local out="$1"
    if dmesg --ctime > "$out" 2>/dev/null && [[ -s "$out" ]]; then return 0; fi
    # Only dmesg needs privilege here; the redirect is deliberately performed by
    # the caller so "$out" stays owned by the invoking user and the run directory
    # remains readable without sudo. SC2024 flags the pattern generically.
    # shellcheck disable=SC2024
    if sudo -n dmesg --ctime > "$out" 2>/dev/null && [[ -s "$out" ]]; then return 0; fi
    if journalctl -k --no-pager -o short-precise > "$out" 2>/dev/null \
       && [[ -s "$out" ]]; then return 0; fi
    echo "(kernel log unreadable: need CAP_SYSLOG — try" \
         "'sudo sysctl kernel.dmesg_restrict=0' or run this script with sudo)" \
         > "$out"
    return 1
}

if ! capture_kernel_log "$RUNDIR/dmesg_before.txt"; then
    say "WARNING: cannot read the kernel log — Xid/AER capture will NOT work."
    say "         Fix with: sudo sysctl kernel.dmesg_restrict=0"
fi
DMESG_BEFORE_LINES=$(wc -l < "$RUNDIR/dmesg_before.txt")

# PCIe error ownership, recorded because it is now a variable across this fleet.
# Whether firmware or the OS owns AER decides whether uncorrectable-AER sampling
# can see anything at all: under firmware-first the platform reads and clears
# those registers, so aer_uncorrectable.csv comes out structurally empty rather
# than genuinely clean. It also governs whether DPC contains a fault or the link
# simply drops.
{
    echo
    echo "--- PCIe error ownership (_OSC negotiation) ---"
    grep -E '_OSC.*(does not support|OS now controls)' \
        "$RUNDIR/dmesg_before.txt" 2>/dev/null \
        | sed 's/^\[[^]]*\] //' | sort -u | head -6
    printf 'dpc_ports_enabled : %s\n' \
        "$(grep -c 'DPC: enabled' "$RUNDIR/dmesg_before.txt" 2>/dev/null || echo 0)"
    printf 'ghes_records_boot : %s\n' \
        "$(grep -c 'Hardware Error' "$RUNDIR/dmesg_before.txt" 2>/dev/null || echo 0)"
} >> "$MANIFEST"

# --- NVML trace (on by default) -------------------------------------------
# Every collector from here down is enabled unless explicitly turned off, so each
# has to fail soft: a missing data source disables that one collector, says so,
# and is recorded in the manifest -- it never aborts the run, because the load
# phase is the part that costs a reboot to repeat.
if ! command -v nvidia-smi >/dev/null 2>&1; then
    if [[ $WITH_NVML -eq 1 || $WITH_DMON -eq 1 ]]; then
        say "WARNING: nvidia-smi not on PATH; skipping the NVML and dmon traces"
    fi
    WITH_NVML=0
    WITH_DMON=0
fi

NVML_PID=""
if [[ $WITH_NVML -eq 1 ]]; then
    say "starting NVML trace at ${NVML_INTERVAL_MS}ms -> $NVML"
    # Field order is load-bearing: the post-run downtrain check below reads
    # gen from column 7 and width from column 8.
    nvidia-smi \
        --query-gpu=timestamp,index,power.draw,clocks.sm,temperature.gpu,utilization.gpu,pcie.link.gen.current,pcie.link.width.current \
        --format=csv,nounits \
        -lms "$NVML_INTERVAL_MS" > "$NVML" 2>"$RUNDIR/nvml_trace.err" &
    NVML_PID=$!
fi

# --- AER correctable error counters --------------------------------------
# The kernel exposes cumulative per-device AER counters in sysfs. Correctable
# errors (notably REPLAY_NUM Rollover, i.e. data-link-layer TLP retries)
# accumulate BEFORE anything fatal happens — in the first reproduction on this
# node the Rollover storm preceded the fatal event by ~1.9s. Sampling them turns
# a binary pass/fail test into a graded one: you can rank configurations, and
# rank individual GPU links against each other, without needing to reach a fault
# and reboot the node each time.
AER_FIELDS="RxErr BadTLP BadDLLP Rollover Timeout NonFatalErr CorrIntErr HeaderOF"

aer_targets() {
    # Emits "index bdf role" for each GPU and for EVERY port in its PCIe path.
    #
    # Walking the whole ancestry rather than just the immediate parent matters on
    # boards with a PCIe switch, where the topology is
    #     root port -> switch upstream -> switch downstream -> GPU
    # so the GPU's parent is the switch downstream port and the root-port-to-
    # switch link would otherwise go unmonitored. That upstream link carries
    # every GPU behind the switch, and a containment event there takes all of
    # them down at once. On riser boards the chain is just root port -> GPU and
    # this yields exactly the two targets it always did.
    nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader,nounits \
        2>/dev/null | while IFS=',' read -r idx bus; do
        idx="${idx// /}"
        # nvidia-smi prints an 8-digit domain (00000000:01:00.0); sysfs uses 4.
        bdf=$(printf '%s' "$bus" | tr -d ' ' | tr 'A-Z' 'a-z' | sed 's/^0000//')
        [[ -e "/sys/bus/pci/devices/$bdf" ]] || continue
        echo "$idx $bdf dev"

        local -a chain=()
        local n i role
        mapfile -t chain < <(readlink -f "/sys/bus/pci/devices/$bdf" 2>/dev/null \
            | tr '/' '\n' \
            | grep -E '^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$')
        n=${#chain[@]}
        for (( i = 0; i < n - 1; i++ )); do
            [[ -e "/sys/bus/pci/devices/${chain[$i]}" ]] || continue
            if (( i == 0 )); then role="rootport"; else role="switch"; fi
            echo "$idx ${chain[$i]} $role"
        done
    done
}

aer_snapshot() {
    # Correctable counters go to a wide CSV (fixed columns, differenced later).
    # Counters are cumulative since boot, so deltas are what matter.
    #
    # UNCORRECTABLE counters go to a separate long-format file, and only rows
    # with a nonzero count are emitted. Two reasons: the field set differs
    # between kernel versions so hardcoding columns would break, and these are
    # zero right up until the fatal event — so a sparse file makes the signal
    # unmissable instead of burying it in a wall of zeros.
    #
    # This is the register that names WHICH fatal error the device reported.
    # `SDES` is Surprise Down Error — its presence or absence discriminates a
    # device that momentarily vanished from one that degraded and then signalled
    # ERR_FATAL over a working link. The post-mortem cannot supply this: by the
    # time the kernel tries to read the device's AER state, DPC has already
    # contained the link ("can't recover (no error_detected callback)").
    local targets="$1" out="$2" ts base cls
    ts=$(ts_iso)
    while read -r idx bdf role; do
        [[ -n "${bdf:-}" ]] || continue
        base="/sys/bus/pci/devices/$bdf"

        if [[ -r "$base/aer_dev_correctable" ]]; then
            awk -v ts="$ts" -v g="$idx" -v b="$bdf" -v r="$role" \
                -v fields="$AER_FIELDS" '
                { v[$1] = $2 }
                END {
                    n = split(fields, F, " ")
                    printf "%s,%s,%s,%s", ts, g, b, r
                    for (i = 1; i <= n; i++)
                        printf ",%s", (F[i] in v) ? v[F[i]] : ""
                    printf "\n"
                }' "$base/aer_dev_correctable" >> "$out"
        fi

        for cls in fatal nonfatal; do
            [[ -r "$base/aer_dev_$cls" ]] || continue
            awk -v ts="$ts" -v g="$idx" -v b="$bdf" -v r="$role" -v c="$cls" \
                '$2 + 0 > 0 {
                     printf "%s,%s,%s,%s,%s,%s,%s\n", ts, g, b, r, c, $1, $2
                 }' "$base/aer_dev_$cls" >> "$AER_UNCORR"
        done
    done <<< "$targets"
}

AER_PID=""
AER_TARGETS=""
if [[ $WITH_AER -eq 1 ]]; then
    AER_TARGETS=$(aer_targets)
    if [[ -z "$AER_TARGETS" ]]; then
        say "WARNING: could not enumerate any AER-capable PCI devices;"
        say "         skipping AER capture (kernel built without CONFIG_PCIEAER?)"
        WITH_AER=0
    else
        {
            printf 'timestamp,gpu,bdf,role'
            for f in $AER_FIELDS; do printf ',%s' "$f"; done
            printf '\n'
        } > "$AER"
        echo "timestamp,gpu,bdf,role,class,field,count" > "$AER_UNCORR"

        # Full baseline dump of all three registers, zeros included, so we know
        # exactly which fields this kernel exposes and where they started.
        {
            echo "AER register baseline at $(ts_iso)"
            echo "(uncorrectable counters are logged sparsely during the run;"
            echo " this is the complete starting state for reference)"
            while read -r idx bdf role; do
                [[ -n "${bdf:-}" ]] || continue
                echo
                echo "=== gpu$idx $bdf ($role) ==="
                for cls in correctable nonfatal fatal; do
                    f="/sys/bus/pci/devices/$bdf/aer_dev_$cls"
                    if [[ -r "$f" ]]; then
                        echo "--- $cls ---"
                        cat "$f"
                    else
                        echo "--- $cls: not readable ---"
                    fi
                done
            done <<< "$AER_TARGETS"
        } > "$AER_BASE" 2>&1

        say "polling AER counters every ${AER_INTERVAL}s -> $AER"
        say "  uncorrectable (incl. SDES/Surprise-Down) -> $AER_UNCORR"
        (
            while :; do
                aer_snapshot "$AER_TARGETS" "$AER"
                sleep "$AER_INTERVAL"
            done
        ) &
        AER_PID=$!
    fi
fi

# --- PSU output current ---------------------------------------------------
# `ipmitool sensor reading <name>...` is far cheaper than `sdr elist full`
# (one IPMI transaction per named sensor, no full SDR walk), which is what makes
# a few Hz achievable over the local KCS interface.
PSU_PID=""
if [[ $WITH_PSU -eq 1 ]]; then
    if ! sudo -n ipmitool sensor reading CUR_PSU1_IOUT >/dev/null 2>&1; then
        say "WARNING: cannot read CUR_PSU1_IOUT via ipmitool (needs sudo and"
        say "         the ipmi_devintf/ipmi_si modules); skipping PSU capture"
        WITH_PSU=0
    else
        # VOLT_12V may not exist under that name; fall back to a nominal 12.0.
        PSU_HAS_V12=0
        if sudo -n ipmitool sensor reading VOLT_12V >/dev/null 2>&1; then
            PSU_HAS_V12=1
        fi
        echo "timestamp,psu1_iout_a,psu2_iout_a,volt_12v,psu1_w,psu2_w,pair_w,est_system_w" \
            > "$PSUCSV"
        say "polling PSU output current every ${PSU_INTERVAL}s -> $PSUCSV"
        (
            names="CUR_PSU1_IOUT CUR_PSU2_IOUT"
            [[ $PSU_HAS_V12 -eq 1 ]] && names="$names VOLT_12V"
            while :; do
                ts=$(ts_iso)
                # shellcheck disable=SC2086
                sudo -n ipmitool sensor reading $names 2>/dev/null | awk -F'|' \
                    -v ts="$ts" -v n="$ACTIVE_SUPPLIES" '
                    {
                        gsub(/^[ \t]+|[ \t]+$/, "", $1)
                        gsub(/^[ \t]+|[ \t]+$/, "", $2)
                        v[$1] = $2 + 0
                    }
                    END {
                        i1 = v["CUR_PSU1_IOUT"]; i2 = v["CUR_PSU2_IOUT"]
                        vv = ("VOLT_12V" in v && v["VOLT_12V"] > 6) ? v["VOLT_12V"] : 12.0
                        w1 = i1 * vv; w2 = i2 * vv
                        # est_system_w assumes the instrumented pair is
                        # representative and load is shared evenly across n
                        # supplies. It is an estimate, not a measurement.
                        est = (i1 + i2) > 0 ? ((w1 + w2) / 2.0) * n : 0
                        printf "%s,%.2f,%.2f,%.2f,%.1f,%.1f,%.1f,%.1f\n",
                            ts, i1, i2, vv, w1, w2, w1 + w2, est
                    }' >> "$PSUCSV"
                sleep "$PSU_INTERVAL"
            done
        ) &
        PSU_PID=$!
    fi
fi

# --- PMBus: all four supplies, input and output power -------------------------
# The BMC exposes sensors for only two of the four supplies, so the BMC channel
# alone measures half the chassis and extrapolates the rest. The ASRock PMBus
# bridge reaches all four directly. See psu_pmbus_poll.py for the register map, the
# LINEAR11 decode, and the read-only command allowlist.
#
# Aggregate the resulting CSV with MEDIANS, not means: the four reads in a round
# are ~30 ms apart and the file sums them, so one stale-low reading drags the sum
# down while lifting it needs all four high. On a measured steady-load run, 44%
# of summed samples fell below the concurrent GPU-only draw and the mean sat
# 478 W below the median. Idle is unaffected.
PMBUS_PID=""
if [[ $WITH_PSU_PMBUS -eq 1 ]]; then
    if [[ ! -x "$HERE/psu_pmbus_poll.py" ]]; then
        say "WARNING: $HERE/psu_pmbus_poll.py not found or not executable;"
        say "         skipping PMBus capture"
        WITH_PSU_PMBUS=0
    elif ! sudo -n ipmitool raw 0x3a 0x52 0x0c 0xb0 0x02 0x97 >/dev/null 2>&1; then
        say "WARNING: PMBus bridge read failed (needs sudo and the OEM command"
        say "         0x3a 0x52 on this platform); skipping PMBus capture."
        say "         Diagnose with: sudo $HERE/psu_pmbus_poll.py --scan"
        WITH_PSU_PMBUS=0
    else
        say "polling all 4 PSUs over PMBus every ${PSU_INTERVAL}s -> $PMBUSCSV"
        "$HERE/psu_pmbus_poll.py" --out "$PMBUSCSV" \
            --interval "$PSU_INTERVAL" >/dev/null 2>"$RUNDIR/psu_pmbus.err" &
        PMBUS_PID=$!
    fi
fi

DMON_PID=""
if [[ $WITH_DMON -eq 1 ]]; then
    say "starting PCIe throughput trace (dmon) -> $DMON"
    # -s t is rx/txpci in MB/s; -o DT prefixes date and time so this joins
    # against the other traces. Counters come from NVML with a ~20ms window, so
    # treat them as a coarse cross-check on pcieburn's own accounting, not as
    # ground truth.
    nvidia-smi dmon -s t -d 1 -o DT > "$DMON" 2>"$RUNDIR/pcie_dmon.err" &
    DMON_PID=$!
fi

# Which collectors are actually running, written after each has had its chance to
# self-disable. This is the record that keeps an absent CSV unambiguous: with
# telemetry on by default, "no aer_delta.txt" means the counters were unreachable
# or explicitly waived, not that nobody asked -- a materially different claim
# about the run when the bundle is read back weeks later.
on_off() { [[ "$1" -eq 1 ]] && echo "on${2:+  $2}" || echo "off"; }
{
    echo
    echo "--- telemetry collectors (on by default; --no-* to waive) ---"
    printf 'nvml              : %s\n' "$(on_off "$WITH_NVML" "${NVML_INTERVAL_MS}ms")"
    printf 'dmon              : %s\n' "$(on_off "$WITH_DMON" "1s")"
    printf 'aer               : %s\n' "$(on_off "$WITH_AER" "${AER_INTERVAL}s")"
    printf 'psu_bmc           : %s\n' "$(on_off "$WITH_PSU" "${PSU_INTERVAL}s")"
    printf 'psu_pmbus         : %s\n' "$(on_off "$WITH_PSU_PMBUS" "${PSU_INTERVAL}s")"
} >> "$MANIFEST"

# --- cleanup: never leave orphans behind ---------------------------------
BURN_PID=""
TAIL_PID=""
: "${AER_PID:=}"
: "${DMON_PID:=}"
: "${NVML_PID:=}"
: "${PSU_PID:=}"
: "${PMBUS_PID:=}"
cleanup() {
    local rc=$?
    trap - EXIT INT TERM

    if [[ -n "$TAIL_PID" ]] && kill -0 "$TAIL_PID" 2>/dev/null; then
        kill -TERM "$TAIL_PID" 2>/dev/null
    fi

    if [[ -n "$BURN_PID" ]] && kill -0 "$BURN_PID" 2>/dev/null; then
        say "stopping pcieburn (pid $BURN_PID)"
        # pcieburn's supervisor forwards teardown to its ranks; give it a
        # chance to stop them cleanly before escalating.
        kill -TERM "$BURN_PID" 2>/dev/null
        for _ in $(seq 1 100); do
            kill -0 "$BURN_PID" 2>/dev/null || break
            sleep 0.1
        done
        if kill -0 "$BURN_PID" 2>/dev/null; then
            say "pcieburn did not exit; sending SIGKILL"
            kill -KILL "$BURN_PID" 2>/dev/null
        fi
    fi

    if [[ -n "$NVML_PID" ]] && kill -0 "$NVML_PID" 2>/dev/null; then
        say "stopping NVML trace"
        kill -TERM "$NVML_PID" 2>/dev/null
        wait "$NVML_PID" 2>/dev/null
    fi

    if [[ -n "$DMON_PID" ]] && kill -0 "$DMON_PID" 2>/dev/null; then
        say "stopping PCIe throughput trace"
        kill -TERM "$DMON_PID" 2>/dev/null
        wait "$DMON_PID" 2>/dev/null
    fi

    if [[ -n "$AER_PID" ]] && kill -0 "$AER_PID" 2>/dev/null; then
        say "stopping AER counter poll"
        kill -TERM "$AER_PID" 2>/dev/null
        wait "$AER_PID" 2>/dev/null
    fi

    if [[ -n "${PMBUS_PID:-}" ]] && kill -0 "$PMBUS_PID" 2>/dev/null; then
        say "stopping PMBus poll"
        kill -TERM "$PMBUS_PID" 2>/dev/null
        wait "$PMBUS_PID" 2>/dev/null
        PMBUS_PID=""
    fi
    if [[ -n "$PSU_PID" ]] && kill -0 "$PSU_PID" 2>/dev/null; then
        say "stopping PSU current poll"
        kill -TERM "$PSU_PID" 2>/dev/null
        wait "$PSU_PID" 2>/dev/null
    fi
    exit $rc
}
trap cleanup EXIT INT TERM

# --- the run --------------------------------------------------------------
ARGS=(--event-log "$EVENTS")
[[ -n "$TAG" ]]      && ARGS+=(--tag "$TAG")
[[ -n "$DURATION" ]] && ARGS+=(--duration "$DURATION")
# Expand safely under `set -u`: an empty PASSTHRU must add no argument at all,
# not an empty string (pcieburn would reject '' as an unknown option).
ARGS+=(${PASSTHRU[@]+"${PASSTHRU[@]}"})


# --- PCIe error policy ------------------------------------------------------
# Applied to every port and endpoint in each GPU's PCIe path, immediately before
# the CUDA section, and only under --relax-pcie-errors.
#
# Why DPC off: DPC contains the link within microseconds of an ERR_FATAL, which
# is why every fault in this investigation logs "ERR_FATAL received from <dev>"
# and never the uncorrectable bit that caused it. That bit lives in the
# ENDPOINT's own AER capability, and the endpoint is unreachable once contained
# — aer_uncorrectable.csv has been header-only in every bundle for exactly this
# reason. Disarming DPC lets the kernel's AER handler read and decode it first.
# It only helps where the device signalled while the link was still up; a
# Surprise Down has already taken the link away and stays undecodable.
#
# Why the severity edit: the PCIe base spec leaves Completion Timeout (bit 14)
# and Unexpected Completion (bit 16) NON-fatal. A platform that marks them fatal
# turns a recoverable transaction error into a link-killing event. Only those
# two bits are touched, read-modify-write, so every other platform-specific
# severity choice survives untouched.
#
# Capability offsets are resolved symbolically (ECAP_DPC 0x001d, ECAP_AER
# 0x0001) because they differ per device — cor04's root ports put DPC at 0x380,
# another board will not.
#
# Register map used here:
#   ECAP_DPC + 0x06 (16-bit)  DPC Control        bits 1:0 = Trigger Enable
#   ECAP_AER + 0x0c (32-bit)  Uncorrectable Error Severity   set bit = fatal
PCIE_SEV_DEMOTE=0x14000        # bit 14 CmpltTO, bit 16 UnxCmplt

# Read one register; empty return means absent, unreadable, or all-ones.
# setpci needs root even to read extended config space (offset >= 0x100).
pcie_reg_read() {
    local v
    v=$(sudo -n setpci -s "$1" "$2" 2>/dev/null) || return 1
    [[ -n "$v" && "$v" != "ffff" && "$v" != "ffffffff" ]] || return 1
    printf '%s' "$v"
}

pcie_reg_write_verify() {      # bdf regspec newvalue -> 0 ok, 1 failed
    sudo -n setpci -s "$1" "$2=$3" 2>/dev/null || return 1
    local back
    back=$(pcie_reg_read "$1" "$2") || return 1
    [[ "$back" == "$3" ]]
}

apply_pcie_error_policy() {
    local idx bdf role cur new dpc_note sev_note
    local changed=0 readable=0

    {
        echo
        echo "--- PCIe error policy (--relax-pcie-errors) ---"
        echo "    DPC trigger disarmed on downstream ports; CmpltTO + UnxCmplt"
        echo "    demoted to non-fatal. Values shown as before->after."
        printf '%-14s %-9s %-22s %s\n' bdf role dpc_ctl uncorr_severity
    } >> "$MANIFEST"

    while read -r idx bdf role; do
        [[ -n "$bdf" ]] || continue
        dpc_note="-"; sev_note="-"

        # 1. DPC trigger off — downstream-facing ports only; endpoints have no DPC
        if [[ "$role" != "dev" ]]; then
            if cur=$(pcie_reg_read "$bdf" ECAP_DPC+0x06.W); then
                readable=$((readable + 1))
                new=$(printf '%04x' $(( 0x$cur & ~0x3 )))
                if [[ "$new" == "$cur" ]]; then
                    dpc_note="$cur (already off)"
                elif pcie_reg_write_verify "$bdf" ECAP_DPC+0x06.W "$new"; then
                    dpc_note="$cur->$new"
                    PCIE_POLICY_UNDO+=("$bdf ECAP_DPC+0x06.W $cur")
                    changed=$((changed + 1))
                else
                    dpc_note="$cur->FAILED"
                    say "WARNING: could not disarm DPC on $bdf"
                fi
            else
                dpc_note="no-dpc-cap"
            fi
        fi

        # 2. severity: demote CmpltTO + UnxCmplt, both ends of every link
        if cur=$(pcie_reg_read "$bdf" ECAP_AER+0x0c.L); then
            readable=$((readable + 1))
            new=$(printf '%08x' $(( 0x$cur & ~PCIE_SEV_DEMOTE )))
            if [[ "$new" == "$cur" ]]; then
                sev_note="$cur (already spec-compliant)"
            elif pcie_reg_write_verify "$bdf" ECAP_AER+0x0c.L "$new"; then
                sev_note="$cur->$new"
                PCIE_POLICY_UNDO+=("$bdf ECAP_AER+0x0c.L $cur")
                changed=$((changed + 1))
            else
                sev_note="$cur->FAILED"
                say "WARNING: could not set severity on $bdf"
            fi
        else
            sev_note="no-aer-cap"
        fi

        printf '%-14s %-9s %-22s %s\n' "$bdf" "$role" "$dpc_note" "$sev_note" \
            >> "$MANIFEST"
    done < <(aer_targets | sort -u -k2,2)

    # Fail closed: readable registers but nothing written means setpci cannot
    # write — almost always missing passwordless sudo. Running anyway would
    # produce a bundle that looks like a policy run and is not one.
    if (( readable > 0 && changed == 0 )); then
        echo "ERROR: --relax-pcie-errors changed nothing despite readable registers." >&2
        echo "       Check that 'sudo -n setpci' works without a password prompt." >&2
        exit 1
    fi
    if (( readable == 0 )); then
        echo "ERROR: --relax-pcie-errors could not read any AER/DPC register." >&2
        echo "       setpci needs root for extended config space (offset >= 0x100)." >&2
        exit 1
    fi
    say "PCIe error policy: $changed register(s) changed, originals in manifest"
}

restore_pcie_error_policy() {
    (( ${#PCIE_POLICY_UNDO[@]} )) || return 0
    local e bdf reg val n=0
    for e in "${PCIE_POLICY_UNDO[@]}"; do
        read -r bdf reg val <<< "$e"
        sudo -n setpci -s "$bdf" "$reg=$val" 2>/dev/null && n=$((n + 1))
    done
    say "restored $n/${#PCIE_POLICY_UNDO[@]} PCIe error-policy register(s)"
}

if [[ $RELAX_PCIE_ERRORS -eq 1 ]]; then
    apply_pcie_error_policy
fi

say "run dir: $RUNDIR"
say "launching: $BIN ${ARGS[*]}"
say "START $(ts_iso)"

# Redirect straight to the log and tail it for the console, rather than piping
# through tee: in a pipeline $! is tee's pid, and the cleanup trap must be able
# to signal the supervisor itself. pcieburn line-buffers its own output, so a
# hang still leaves a log whose last line carries a usable timestamp.
LOAD_T0=$(date +%s)
"$BIN" "${ARGS[@]}" > "$CONSOLE" 2>&1 &
BURN_PID=$!
tail -f --pid="$BURN_PID" -n +1 "$CONSOLE" 2>/dev/null &
TAIL_PID=$!

wait "$BURN_PID"
RC=$?
LOAD_T1=$(date +%s)
BURN_PID=""
wait "$TAIL_PID" 2>/dev/null || true
TAIL_PID=""

say "STOP $(ts_iso) exit=$RC"

# --- settle window -------------------------------------------------------
# The load ending is not the end of the experiment. Shedding ~2 kW of coherent
# GPU load in under a second is itself a large electrical transient, and a
# containment event has been observed firing one second after idle was reached —
# 8.8 s after the load phase ended. Every collector is still running here
# (they are only torn down by the EXIT trap), so the fault lands in
# nvml_trace.csv, aer_counters.csv and psu_current.csv as well as the kernel log.
if [[ "$SETTLE_SECONDS" -gt 0 ]]; then
    say "settling ${SETTLE_SECONDS}s before the post-run snapshot (collectors still running)"
    sleep "$SETTLE_SECONDS"
    say "SETTLED $(ts_iso)"
fi

# Restore only on a clean exit. After a fault the port may be contained or the
# endpoint gone, and writing config space to a device in that state is not worth
# the risk — the settings do not survive a reboot anyway.
if [[ $RELAX_PCIE_ERRORS -eq 1 && $RC -eq 0 ]]; then
    restore_pcie_error_policy
fi

# --- post-run capture ----------------------------------------------------
capture_kernel_log "$RUNDIR/dmesg_after.txt" || true
if [[ -s "$RUNDIR/dmesg_after.txt" ]]; then
    tail -n "+$((DMESG_BEFORE_LINES + 1))" "$RUNDIR/dmesg_after.txt" \
        > "$RUNDIR/dmesg_delta.txt" 2>/dev/null || true
    grep -iE 'xid|aer|dpc|pcieport|nvidia-?smi|gpu has fallen' \
        "$RUNDIR/dmesg_delta.txt" > "$RUNDIR/faults.txt" 2>/dev/null || true
fi

# Did a FATAL land in this run's window? Correctable AER is expected noise and is
# deliberately excluded — only containment and endpoint loss count. This is
# matched against the delta rather than the ranks' own exit status precisely
# because the ranks can all exit 0 and the link still be gone a few seconds
# later.
POST_FATAL=0
POST_FATAL_WHAT=""
if [[ -s "$RUNDIR/dmesg_delta.txt" ]]; then
    POST_FATAL_WHAT=$(grep -oE 'DPC: containment event|ERR_FATAL received|severity=Uncorrected \(Fatal\)|severity=Uncorrectable \(Fatal\)|GPU has fallen off the bus|\[ *5\] SDES' \
        "$RUNDIR/dmesg_delta.txt" 2>/dev/null | sort -u \
        | awk '{ printf "%s%s", sep, $0; sep = "; " } END { print "" }')
    [[ -n "$POST_FATAL_WHAT" ]] && POST_FATAL=1
fi

# PCIe link state summary. A link that downtrains below Gen5 x16 under load is a
# strong clue in its own right — it is the step before falling off the bus
# entirely — and it is nearly free to extract from the trace already captured.
#
# Note: consumer cards legitimately downtrain to Gen1 when idle to save power,
# and this trace spans the idle periods before and after the load phase. So the
# table below is informational; what matters is whether a low state appears
# during the load window, which you check against the timestamps in the trace.
LINK_NOTE=0
AER_NONZERO=0
AER_UNCORR_HIT=0
if [[ -s "$NVML" ]]; then
    awk -F',' '
        NR > 1 && NF >= 8 {
            gpu = $2 + 0; gen = $7 + 0; width = $8 + 0
            if (gen == 0 && width == 0) next
            key = sprintf("gpu%d\tgen%d\tx%d", gpu, gen, width)
            if (!(key in seen)) { seen[key] = 1; order[++n] = key }
            count[key]++
        }
        END {
            for (i = 1; i <= n; i++)
                printf "%s\tsamples=%d\n", order[i], count[order[i]]
        }' "$NVML" 2>/dev/null | sort > "$LINKSTATES" || true

    if [[ -s "$LINKSTATES" ]] && \
       awk -F'\t' '$2 != "gen5" || $3 != "x16" { f = 1 } END { exit !f }' \
           "$LINKSTATES" 2>/dev/null; then
        LINK_NOTE=1
    fi

    # Classify what kind of non-Gen5-x16 it is, because the three kinds mean
    # completely different things and lumping them into one note let a genuinely
    # degraded run be reported as clean twice:
    #
    #   x0            DPC has contained the link. Already fatal; not "degraded".
    #   width < 16    NEVER legitimate. ASPM negotiates speed, not width, and
    #                 every GPU slot here is x16. A marginal port came up x8 and
    #                 ran two entire 600 s runs that way, reporting clean with
    #                 zero AER errors, while being the most degraded link in the
    #                 fleet. Half width is a de-rated electrical configuration:
    #                 the run is not comparable against full-width runs.
    #   gen < 5       A brief ramp is normal — every healthy GPU here spends
    #                 ~5.5 s at gen1 while training. A long dwell is not: one
    #                 port sat at gen1 for 265 s, 48x its peers, in the run where
    #                 it logged 8018 BadTLP. Threshold is expressed in seconds and
    #                 converted using this run's own NVML interval.
    GEN_DWELL_MAX_S=20
    # Aggregate ONLY samples inside the load window [LOAD_T0+5s, LOAD_T1] so
    # idle-at-gen1 before launch and during the settle window cannot count
    # toward dwell. Requires gawk mktime; if unavailable, falls back to the
    # full-trace table (over-sensitive, never under-sensitive).
    LINKSTATES_LOAD="$RUNDIR/pcie_link_states_load.txt"
    if command -v gawk >/dev/null 2>&1 && [[ -n "${LOAD_T0:-}" && -n "${LOAD_T1:-}" ]]; then
        gawk -F',' -v t0="$((LOAD_T0 + 5))" -v t1="$LOAD_T1" '
            NR > 1 && NF >= 8 {
                split($1, dt, " "); gsub(/[\/:]/, " ", dt[1]); gsub(/:/, " ", dt[2])
                split(dt[2], tm, " ")
                ep = mktime(dt[1] " " tm[1] " " tm[2] " " int(tm[3]))
                if (ep < t0 || ep > t1) next
                gpu = $2 + 0; gen = $7 + 0; width = $8 + 0
                if (gen == 0 && width == 0) next
                key = sprintf("gpu%d\tgen%d\tx%d", gpu, gen, width)
                if (!(key in seen)) { seen[key] = 1; order[++n] = key }
                count[key]++
            }
            END {
                for (i = 1; i <= n; i++)
                    printf "%s\tsamples=%d\n", order[i], count[order[i]]
            }' "$NVML" 2>/dev/null | sort > "$LINKSTATES_LOAD" || true
    fi
    [[ -s "$LINKSTATES_LOAD" ]] || cp "$LINKSTATES" "$LINKSTATES_LOAD" 2>/dev/null || true
    DEGRADED_LINK=0
    DEGRADED_WHAT=""
    DEGRADED_WHAT=$(awk -F'\t' -v ivl="$NVML_INTERVAL_MS" -v maxs="$GEN_DWELL_MAX_S" '
        {
            split($3, w, "x"); width = w[2] + 0
            split($2, g, "gen"); gen = g[2] + 0
            split($4, c, "="); n = c[2] + 0
            if (width == 0) next                        # containment, reported elsewhere
            if (width < 16) { narrow[$1] = $2 " " $3; nsamp[$1] += n; next }
            if (gen < 5)      slow[$1] += n
        }
        END {
            lim = maxs * 1000 / (ivl > 0 ? ivl : 100)
            for (gpu in narrow)
                printf "%s ran %s for %d samples (never legitimate at x16 slots); ",
                       gpu, narrow[gpu], nsamp[gpu]
            for (gpu in slow)
                if (slow[gpu] > lim)
                    printf "%s dwelt below gen5 for %d samples (%.0fs, limit %ds); ",
                           gpu, slow[gpu], slow[gpu] * ivl / 1000, maxs
        }' "$LINKSTATES_LOAD" 2>/dev/null)
    [[ -n "$DEGRADED_WHAT" ]] && DEGRADED_LINK=1
fi
: "${DEGRADED_LINK:=0}"
: "${DEGRADED_WHAT:=}"
: "${POST_FATAL:=0}"

# AER delta table — per-link accumulation of correctable errors over this run.
# This is the discriminating measurement: it ranks the eight links against each
# other whether or not anything actually failed.
if [[ $WITH_AER -eq 1 && -n "$AER_TARGETS" ]]; then
    aer_snapshot "$AER_TARGETS" "$AER" || true   # final sample closes the window
    if [[ -s "$AER" ]]; then
        # All eight correctable fields, not just the first five. Printing only
        # RxErr..Timeout hid a real signal once: on a switched node every trunk
        # port recorded NonFatalErr=1 while showing zero in the printed columns,
        # which read as "the upstream fabric is clean" when it was not.
        awk -F',' 'NR > 1 {
            k = $4 "|" $2 "|" $3
            if (!(k in seen)) {
                seen[k] = 1; order[++n] = k
                for (i = 5; i <= 12; i++) f[k, i] = $i + 0
            }
            for (i = 5; i <= 12; i++) l[k, i] = $i + 0
        } END {
            for (j = 1; j <= n; j++) {
                k = order[j]; split(k, p, "|")
                printf "%-8s %-3s %-14s", p[1], p[2], p[3]
                for (i = 5; i <= 12; i++) printf " %8d", l[k, i] - f[k, i]
                printf "\n"
            }
        }' "$AER" 2>/dev/null | sort -k7 -nr > "$RUNDIR/.aer_body" || true

        {
            printf '%-8s %-3s %-14s %8s %8s %8s %8s %8s %8s %8s %8s\n' \
                role gpu bdf RxErr BadTLP BadDLLP Rollover Timeout \
                NonFatalErr CorrIntErr HeaderOF
            cat "$RUNDIR/.aer_body"
        } > "$AERDELTA"
        rm -f "$RUNDIR/.aer_body"

        if awk 'NR > 1 { for (i = 4; i <= 11; i++) if ($i + 0 > 0) { f = 1 } }
                END { exit !f }' "$AERDELTA" 2>/dev/null; then
            AER_NONZERO=1
        fi
    fi

    # Any uncorrectable row at all is significant — these are zero on a healthy
    # link, and the field name identifies the actual fatal error.
    if [[ -s "$AER_UNCORR" ]] \
       && awk -F',' 'NR > 1 { f = 1 } END { exit !f }' "$AER_UNCORR" 2>/dev/null; then
        AER_UNCORR_HIT=1
    fi
fi

# PSU current summary. The headline figures are peak per-supply load against the
# CRPS rating, and the size of the current SWING — the latter is the transient
# that the correlated-power hypothesis is about, and it is invisible in any of
# the watt-denominated sensors on this platform.
if [[ $WITH_PSU -eq 1 && -s "$PSUCSV" ]]; then
    awk -F',' -v n="$ACTIVE_SUPPLIES" -v rating="$PSU_RATING_W" '
        NR > 1 && $2 + 0 > 0 {
            for (k = 2; k <= 3; k++) {
                c = $k + 0
                if (c > mx[k]) mx[k] = c
                if (mn[k] == "" || c < mn[k]) mn[k] = c
                s[k] += c
            }
            cnt++
            if ($7 + 0 > pairmx) pairmx = $7 + 0
            if ($8 + 0 > estmx)  estmx  = $8 + 0
        }
        END {
            if (!cnt) { print "no usable samples"; exit }
            amps = rating / 12.0
            printf "samples                  %d\n", cnt
            printf "PSU1 IOUT                min %6.2f A   max %6.2f A   mean %6.2f A\n",
                mn[2], mx[2], s[2] / cnt
            printf "PSU2 IOUT                min %6.2f A   max %6.2f A   mean %6.2f A\n",
                mn[3], mx[3], s[3] / cnt
            printf "peak swing PSU1          %6.2f A  (~%.0f W at 12V)\n",
                mx[2] - mn[2], (mx[2] - mn[2]) * 12.0
            printf "peak swing PSU2          %6.2f A  (~%.0f W at 12V)\n",
                mx[3] - mn[3], (mx[3] - mn[3]) * 12.0
            printf "peak instrumented pair   %.0f W\n", pairmx
            printf "peak est. system (x%d)    %.0f W   [estimate, assumes even sharing]\n",
                n, estmx
            printf "peak per-supply load     %.0f%% of %.0fW rating (%.0f A max)\n",
                100.0 * mx[2] / amps, rating, amps
            printf "\nNote: sampled at the poll interval, so these are envelope\n"
            printf "values. Sub-millisecond excursions are not visible here.\n"
        }' "$PSUCSV" > "$PSUSUM" 2>/dev/null || true
fi

# Re-read the root-port link state after the run. A link that was at capability
# before and is below it after is a much stronger signal than either reading alone.
if [[ -x "$HERE/linkcheck.sh" ]]; then
    "$HERE/linkcheck.sh" > "$RUNDIR/pcie_link_rootports_after.txt" 2>&1 || true
    if [[ -s "$ROOTPORT_LINK" && -s "$RUNDIR/pcie_link_rootports_after.txt" ]] \
       && ! diff -q "$ROOTPORT_LINK" "$RUNDIR/pcie_link_rootports_after.txt" >/dev/null 2>&1; then
        say "  *** root-port link state CHANGED across this run ***"
        diff "$ROOTPORT_LINK" "$RUNDIR/pcie_link_rootports_after.txt" 2>/dev/null \
            | while IFS= read -r l; do say "      $l"; done
    fi
fi

{
    echo
    echo "end_utc           : $(ts_iso)"
    echo "uptime_at_end_s   : $(awk '{printf "%d", $1}' /proc/uptime 2>/dev/null || echo 0)"
    echo "rank_exit_code    : $RC"
    # The ranks' exit status is not the whole verdict. A run whose ranks all
    # exited 0 can still have lost a link seconds later, and a run that completed
    # normally can have done so on a de-rated link. Both were observed, and both
    # were originally recorded as "clean".
    case "$RC" in
        0)  if [[ $POST_FATAL -eq 1 ]]; then
                VERDICT_RC=4
                echo "verdict           : FAULT AFTER LOAD PHASE (post-window containment)"
                echo "post_window_fatal : $POST_FATAL_WHAT"
            elif [[ $DEGRADED_LINK -eq 1 ]]; then
                VERDICT_RC=5
                echo "verdict           : CLEAN BUT LINK DEGRADED (not comparable)"
                echo "degraded_link     : $DEGRADED_WHAT"
            else
                VERDICT_RC=0
                echo "verdict           : clean"
            fi ;;
        2)  VERDICT_RC=2; echo "verdict           : COMPUTE FAULTS reported" ;;
        3)  VERDICT_RC=3; echo "verdict           : RANK LOST OR HUNG (possible reproduction)" ;;
        *)  VERDICT_RC=$RC; echo "verdict           : setup/usage error" ;;
    esac
    # Kept for continuity with v1 bundles, where exit_code and the verdict were
    # the same thing.
    echo "exit_code         : $VERDICT_RC"
    if [[ $DEGRADED_LINK -eq 1 && $VERDICT_RC -ne 5 ]]; then
        echo "degraded_link     : $DEGRADED_WHAT"
    fi
} >> "$MANIFEST"

say "artifacts in $RUNDIR:"
say "  manifest.txt   run provenance, BIOS/topology snapshot, verdict"
say "  pcieburn.log   timestamped console output"
say "  events.csv     per-rank event log for telemetry correlation"
[[ $WITH_NVML -eq 1 ]] && say "  nvml_trace.csv per-GPU NVML trace incl. PCIe link gen/width"
if [[ $WITH_DMON -eq 1 ]]; then
    # dmon reports '-' instead of a rate on parts that do not expose the PCIe
    # throughput counters. That is not a failure, but the file is then empty of
    # numbers, and saying so beats listing it as if it held data.
    # Columns are date, time, gpu, rxpci, txpci (from -s t -o DT above), so the
    # two throughput fields are $4 and $5. Testing the whole line instead would
    # always pass: the date and GPU index are digits even on an all-'-' row.
    if awk '$0 ~ /^#/ { next }
            $4 ~ /^[0-9]+$/ || $5 ~ /^[0-9]+$/ { n++ }
            END { exit !(n > 0) }' "$DMON" 2>/dev/null; then
        say "  pcie_dmon.txt  PCIe rx/tx throughput (independent cross-check)"
    else
        say "  pcie_dmon.txt  (no throughput reported — these counters are not"
        say "                 exposed on this part; not a failure)"
    fi
fi
if [[ -s "$PSUSUM" ]]; then
    say "  psu_current.csv / psu_summary.txt  per-PSU output current:"
    while IFS= read -r l; do [[ -n "$l" ]] && say "      $l"; done < "$PSUSUM"
fi
if [[ $AER_UNCORR_HIT -eq 1 ]]; then
    say "  *** UNCORRECTABLE AER recorded — this names the fatal error ***"
    awk -F',' 'NR > 1 {
        k = $2 "|" $3 "|" $4 "|" $5 "|" $6
        if (!(k in seen)) { seen[k] = 1; order[++n] = k }
        last[k] = $7
    } END {
        for (i = 1; i <= n; i++) {
            split(order[i], p, "|")
            printf "gpu%s %s (%s)  %s.%s = %s\n", p[1], p[2], p[3], p[4], p[5], last[order[i]]
        }
    }' "$AER_UNCORR" 2>/dev/null | while IFS= read -r l; do say "      $l"; done
    if grep -q ',SDES,' "$AER_UNCORR" 2>/dev/null; then
        say "      SDES = Surprise Down Error: the link dropped unexpectedly,"
        say "      i.e. the device stopped responding rather than degrading and"
        say "      then signalling ERR_FATAL over a working link."
    else
        say "      No SDES (Surprise Down) — consistent with a link that degraded"
        say "      and reported a fatal error, not one that vanished."
    fi
fi
if [[ -s "$AERDELTA" ]]; then
    say "  aer_counters.csv / aer_delta.txt  per-link AER accumulation:"
    while IFS= read -r l; do say "      $l"; done < "$AERDELTA"
    if [[ $AER_NONZERO -eq 0 ]]; then
        say "    (all zero — no correctable link errors accrued this run)"
    fi
fi
if [[ -s "$LINKSTATES" ]]; then
    say "  pcie_link_states.txt  distinct PCIe link states observed"
    if [[ $LINK_NOTE -eq 1 ]]; then
        say "    note: at least one sample below Gen5 x16. ASPM is disabled on"
        say "    this platform's root ports, so this is NOT normal power saving —"
        say "    check the timestamps to see whether it falls inside the load window:"
        while IFS= read -r l; do say "      $l"; done < "$LINKSTATES"
    fi
    if [[ $DEGRADED_LINK -eq 1 ]]; then
        say "    *** LINK DEGRADED: $DEGRADED_WHAT"
        say "    This run's electrical configuration differs from a full-width,"
        say "    fully-trained one. Do NOT compare its AER counts against other"
        say "    runs: a link running at half width can log zero errors while"
        say "    being the most degraded one present. Retrain and re-check before"
        say "    using this node for anything comparative:"
        say "      sudo setpci -s <port> CAP_EXP+10.w=0020:0020   # Retrain Link"
        say "      sudo ./linkcheck.sh"
    fi
fi
if [[ $POST_FATAL -eq 1 && $RC -eq 0 ]]; then
    say "  *** FATAL AFTER LOAD PHASE — the ranks all exited 0 and then the link"
    say "      was lost during the ${SETTLE_SECONDS}s settle window:"
    say "      $POST_FATAL_WHAT"
    say "      Without --settle this run would have been recorded as clean."
fi
if [[ -s "$PMBUSCSV" ]]; then
    say "  psu_pmbus.csv  all 4 PSUs, input and output power:"
    awk -F',' '
        NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
        $col["read_ok"] != 1 { bad++; next }
        { n++
          sp[n] = $col["system_pout_w"] + 0
          si[n] = $col["system_pin_w"]  + 0
          for (p = 1; p <= 4; p++) po[p][n] = $col["psu" p "_pout_w"] + 0
          if (n == 1 || sp[n] < lo) lo = sp[n]
          if (n == 1 || sp[n] > hi) hi = sp[n] }
        END {
            if (n == 0) { printf "      no successful reads (see psu_pmbus.err)\n"; exit }
            # Split idle from load at the midpoint of the observed range and take a
            # median WITHIN each regime; a whole-file median would depend on how
            # much idle padding the capture happens to contain. A flat trace has
            # no split to make.
            flat = (hi <= 0 || (hi - lo) / hi < 0.20)
            thr = lo + 0.5 * (hi - lo)
            for (i = 1; i <= n; i++) {
                if (flat || sp[i] >= thr) {
                    L[++nl] = sp[i]; LI[nl] = si[i]
                    for (p = 1; p <= 4; p++) LP[p][nl] = po[p][i]
                } else I[++ni] = sp[i]
            }
            if (!flat && ni > 0) {
                asort(I)
                printf "      idle   median out %7.1f W  (n=%d)\n", I[int(ni/2)+1], ni
            }
            if (nl > 0) {
                asort(L); asort(LI)
                for (p = 1; p <= 4; p++) {
                    asort(LP[p]); s = s sprintf(" %.0f", LP[p][int(nl/2)+1])
                }
                printf "      %s median out %7.1f W  in %7.1f W  (n=%d)\n",
                       (flat ? "steady" : "loaded"), L[int(nl/2)+1], LI[int(nl/2)+1], nl
                printf "      %s per-PSU out:%s W\n", (flat ? "steady" : "loaded"), s
            }
            printf "      %d reads", n
            if (bad > 0) printf ", %d FAILED", bad
            printf ". Aggregate with medians: instantaneous 4-PSU sums\n"
            printf "      are unreliable under load (one stale read drags a sum down).\n"
        }' "$PMBUSCSV" 2>/dev/null | while IFS= read -r l; do say "$l"; done
fi
if [[ -s "$RUNDIR/faults.txt" ]]; then
    say "  faults.txt     *** kernel reported PCIe/Xid activity — read this ***"
fi

exit "${VERDICT_RC:-$RC}"
