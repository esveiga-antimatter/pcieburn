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
#   ./run_pcieburn.sh --with-nvml --duration 300 -- --gemms-per-coll 16
#   ./run_pcieburn.sh --outdir /data/runs --tag gen5-retest -- --collective alltoall
#
# Anything after `--` is passed straight through to the pcieburn binary.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${PCIEBURN_BIN:-$HERE/pcieburn}"

OUTDIR_BASE="$HERE/runs"
TAG=""
WITH_NVML=0
WITH_DMON=0
WITH_AER=0
AER_INTERVAL=1
NVML_INTERVAL_MS=100
DURATION=""
ASSUME_YES=0
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
  --with-nvml         also capture a per-GPU nvidia-smi trace for this run
                      (power, clocks, temp, util, and PCIe link gen/width)
  --with-dmon         also capture nvidia-smi dmon PCIe rx/tx throughput.
                      Independent of pcieburn's own byte accounting, so it
                      cross-checks it. May report '-' on GeForce parts.
  --with-aer          poll per-GPU and per-root-port AER correctable error
                      counters from sysfs. This is the graded measurement: link
                      errors accumulate before anything fatal happens, so you
                      can rank configurations without having to reach a fault
                      and reboot. Strongly recommended.
  --aer-interval SEC  AER poll interval (default 1)
  --nvml-interval MS  NVML sample interval (default 100)
  --yes               skip the interactive safety confirmation
  -h, --help          this message

Everything after `--` goes to pcieburn verbatim. See ./pcieburn --help.

SAFETY: this test tries to reproduce a fault that has previously needed a hard
power cycle. Run it only on a designated test node.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --outdir)        OUTDIR_BASE="$2"; shift 2 ;;
        --tag)           TAG="$2"; shift 2 ;;
        --duration)      DURATION="$2"; shift 2 ;;
        --with-nvml)     WITH_NVML=1; shift ;;
        --with-dmon)     WITH_DMON=1; shift ;;
        --with-aer)      WITH_AER=1; shift ;;
        --aer-interval)  AER_INTERVAL="$2"; shift 2 ;;
        --nvml-interval) NVML_INTERVAL_MS="$2"; shift 2 ;;
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
    echo "--- PCIe link capability vs current (baseline before load) ---"
    nvidia-smi --query-gpu=index,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current \
        --format=csv 2>&1 || true
    echo
    echo "--- NUMA affinity (cross-socket staging matters: P2P is disabled) ---"
    nvidia-smi topo -m 2>&1 | head -25 || true
} > "$MANIFEST"

# Baseline PCIe link check, before any load. Worth surfacing rather than burying
# in the manifest: in the first reproduction on this node, the only GPU idling at
# a lower link generation than its peers was the one that fell off the bus.
BASELINE_LINK="$RUNDIR/pcie_link_baseline.csv"
if nvidia-smi --query-gpu=index,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current \
        --format=csv,noheader,nounits > "$BASELINE_LINK" 2>/dev/null; then
    ODD_LINKS=$(awk -F', *' 'NF>=5 && ($3+0 < $2+0 || $5+0 < $4+0) {
        printf "GPU%s at gen%s x%s (max gen%s x%s)\n", $1, $3, $5, $2, $4 }' \
        "$BASELINE_LINK" 2>/dev/null)
    if [[ -n "$ODD_LINKS" ]]; then
        say "NOTE: PCIe link(s) below maximum at idle:"
        while IFS= read -r l; do say "    $l"; done <<< "$ODD_LINKS"
        say "    Idle downtraining is normal power saving, but a card that"
        say "    differs from its peers is a signal worth recording."
    fi
fi

# Kernel log watermark, so the after-diff shows only this run's messages.
# dmesg is often restricted (kernel.dmesg_restrict=1), which silently produced
# an empty delta and no faults.txt on the first real run — the single most
# important artifact. Try unprivileged, then sudo -n, then journalctl.
capture_kernel_log() {
    local out="$1"
    if dmesg --ctime > "$out" 2>/dev/null && [[ -s "$out" ]]; then return 0; fi
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

# --- optional NVML trace --------------------------------------------------
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
    # Emits "index bdf role" for each GPU and for its upstream root port.
    nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader,nounits \
        2>/dev/null | while IFS=',' read -r idx bus; do
        idx="${idx// /}"
        # nvidia-smi prints an 8-digit domain (00000000:01:00.0); sysfs uses 4.
        bdf=$(printf '%s' "$bus" | tr -d ' ' | tr 'A-Z' 'a-z' | sed 's/^0000//')
        [[ -e "/sys/bus/pci/devices/$bdf" ]] || continue
        echo "$idx $bdf dev"
        local rp
        rp=$(basename "$(dirname "$(readlink -f \
            "/sys/bus/pci/devices/$bdf" 2>/dev/null)")" 2>/dev/null) || rp=""
        if [[ -n "$rp" && -e "/sys/bus/pci/devices/$rp" ]]; then
            echo "$idx $rp rootport"
        fi
    done
}

aer_snapshot() {
    # One CSV row per target. Counters are cumulative since boot, so deltas are
    # what matter; absolute values are logged and differenced afterwards.
    local targets="$1" out="$2" ts
    ts=$(ts_iso)
    while read -r idx bdf role; do
        [[ -n "${bdf:-}" ]] || continue
        local f="/sys/bus/pci/devices/$bdf/aer_dev_correctable"
        [[ -r "$f" ]] || continue
        awk -v ts="$ts" -v g="$idx" -v b="$bdf" -v r="$role" \
            -v fields="$AER_FIELDS" '
            { v[$1] = $2 }
            END {
                n = split(fields, F, " ")
                printf "%s,%s,%s,%s", ts, g, b, r
                for (i = 1; i <= n; i++)
                    printf ",%s", (F[i] in v) ? v[F[i]] : ""
                printf "\n"
            }' "$f" >> "$out"
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
        say "polling AER counters every ${AER_INTERVAL}s -> $AER"
        (
            while :; do
                aer_snapshot "$AER_TARGETS" "$AER"
                sleep "$AER_INTERVAL"
            done
        ) &
        AER_PID=$!
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

# --- cleanup: never leave orphans behind ---------------------------------
BURN_PID=""
TAIL_PID=""
: "${AER_PID:=}"
: "${DMON_PID:=}"
: "${NVML_PID:=}"
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

say "run dir: $RUNDIR"
say "launching: $BIN ${ARGS[*]}"
say "START $(ts_iso)"

# Redirect straight to the log and tail it for the console, rather than piping
# through tee: in a pipeline $! is tee's pid, and the cleanup trap must be able
# to signal the supervisor itself. pcieburn line-buffers its own output, so a
# hang still leaves a log whose last line carries a usable timestamp.
"$BIN" "${ARGS[@]}" > "$CONSOLE" 2>&1 &
BURN_PID=$!
tail -f --pid="$BURN_PID" -n +1 "$CONSOLE" 2>/dev/null &
TAIL_PID=$!

wait "$BURN_PID"
RC=$?
BURN_PID=""
wait "$TAIL_PID" 2>/dev/null || true
TAIL_PID=""

say "STOP $(ts_iso) exit=$RC"

# --- post-run capture ----------------------------------------------------
capture_kernel_log "$RUNDIR/dmesg_after.txt" || true
if [[ -s "$RUNDIR/dmesg_after.txt" ]]; then
    tail -n "+$((DMESG_BEFORE_LINES + 1))" "$RUNDIR/dmesg_after.txt" \
        > "$RUNDIR/dmesg_delta.txt" 2>/dev/null || true
    grep -iE 'xid|aer|dpc|pcieport|nvidia-?smi|gpu has fallen' \
        "$RUNDIR/dmesg_delta.txt" > "$RUNDIR/faults.txt" 2>/dev/null || true
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
fi

# AER delta table — per-link accumulation of correctable errors over this run.
# This is the discriminating measurement: it ranks the eight links against each
# other whether or not anything actually failed.
if [[ $WITH_AER -eq 1 && -n "$AER_TARGETS" ]]; then
    aer_snapshot "$AER_TARGETS" "$AER" || true   # final sample closes the window
    if [[ -s "$AER" ]]; then
        awk -F',' 'NR > 1 {
            k = $4 "|" $2 "|" $3
            if (!(k in seen)) {
                seen[k] = 1; order[++n] = k
                f5[k]=$5+0; f6[k]=$6+0; f7[k]=$7+0; f8[k]=$8+0; f9[k]=$9+0
            }
            l5[k]=$5+0; l6[k]=$6+0; l7[k]=$7+0; l8[k]=$8+0; l9[k]=$9+0
        } END {
            for (i = 1; i <= n; i++) {
                k = order[i]; split(k, p, "|")
                printf "%-8s %-3s %-12s %6d %7d %8d %9d %8d\n",
                    p[1], p[2], p[3],
                    l5[k]-f5[k], l6[k]-f6[k], l7[k]-f7[k],
                    l8[k]-f8[k], l9[k]-f9[k]
            }
        }' "$AER" 2>/dev/null | sort -k7 -nr > "$RUNDIR/.aer_body" || true

        {
            printf '%-8s %-3s %-12s %6s %7s %8s %9s %8s\n' \
                role gpu bdf RxErr BadTLP BadDLLP Rollover Timeout
            cat "$RUNDIR/.aer_body"
        } > "$AERDELTA"
        rm -f "$RUNDIR/.aer_body"

        if awk 'NR > 1 { for (i = 4; i <= 8; i++) if ($i + 0 > 0) { f = 1 } }
                END { exit !f }' "$AERDELTA" 2>/dev/null; then
            AER_NONZERO=1
        fi
    fi
fi

{
    echo
    echo "end_utc           : $(ts_iso)"
    echo "exit_code         : $RC"
    case "$RC" in
        0) echo "verdict           : clean" ;;
        2) echo "verdict           : COMPUTE FAULTS reported" ;;
        3) echo "verdict           : RANK LOST OR HUNG (possible reproduction)" ;;
        *) echo "verdict           : setup/usage error" ;;
    esac
} >> "$MANIFEST"

say "artifacts in $RUNDIR:"
say "  manifest.txt   run provenance, BIOS/topology snapshot, verdict"
say "  pcieburn.log   timestamped console output"
say "  events.csv     per-rank event log for telemetry correlation"
[[ $WITH_NVML -eq 1 ]] && say "  nvml_trace.csv per-GPU NVML trace incl. PCIe link gen/width"
[[ $WITH_DMON -eq 1 ]] && say "  pcie_dmon.txt  PCIe rx/tx throughput (independent cross-check)"
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
        say "    note: at least one sample below Gen5 x16 — expected while idle,"
        say "    significant if it falls inside the load window:"
        while IFS= read -r l; do say "      $l"; done < "$LINKSTATES"
    fi
fi
if [[ -s "$RUNDIR/faults.txt" ]]; then
    say "  faults.txt     *** kernel reported PCIe/Xid activity — read this ***"
fi

exit "$RC"
