#!/usr/bin/env bash
# mgx_preflight.sh — read-only platform inventory for a pcieburn test node.
#
# Written for the intelmgx-CG480 / RTX PRO 6000 box (driver 595, kernel 6.8),
# but deliberately host-agnostic: run it on cptcor04 or an RGCA node FIRST, so
# you can see what each probe looks like where ground truth already exists.
# That comparison is the whole point for the AER section — on a firmware-first
# platform the sysfs counters never increment, and you cannot tell "zero errors"
# from "blind instrument" without having seen the instrument work somewhere.
#
# Everything here is a read. No power limits are set, no links are retrained,
# no load is started, nothing is written outside the output directory.
#
#   ./mgx_preflight.sh                 # writes ./preflight-<host>-<stamp>/
#   ./mgx_preflight.sh --outdir DIR
#
# Some probes need root (dmesg when kernel.dmesg_restrict=1, lspci -vvv for
# capability bodies, ipmitool). Each one says so rather than silently emitting
# an empty section — an empty section that looks like a clean result is the
# specific failure mode this whole investigation keeps re-learning.

set -uo pipefail

OUTBASE="./preflight"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --outdir) OUTBASE="$2"; shift 2 ;;
        -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option '$1'" >&2; exit 1 ;;
    esac
done

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUTBASE}-$(hostname)-${STAMP}"
mkdir -p "$OUT" || { echo "cannot create $OUT" >&2; exit 1; }

say()  { printf '%s\n' "$*"; }
hdr()  { printf '\n===== %s =====\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# Run a command into a file, and record explicitly whether it worked.
# A missing tool, a permission failure and an empty result are three different
# things and must not collapse into one blank file.
cap() {
    local label="$1" file="$2"; shift 2
    if ! have "$1"; then
        printf 'UNAVAILABLE: %s not installed\n' "$1" > "$OUT/$file"
        say "  $label: SKIPPED — $1 not installed"; return
    fi
    if "$@" > "$OUT/$file" 2>"$OUT/$file.err"; then
        if [[ -s "$OUT/$file" ]]; then
            say "  $label: ok ($(wc -l < "$OUT/$file") lines) -> $file"
        else
            say "  $label: RAN BUT EMPTY -> $file  (treat as no-data, not as zero)"
        fi
    else
        say "  $label: FAILED (see $file.err) — likely needs root"
    fi
    [[ -s "$OUT/$file.err" ]] || rm -f "$OUT/$file.err"
}

exec > >(tee "$OUT/preflight.log") 2>&1

say "pcieburn platform preflight"
say "host    : $(hostname)"
say "utc     : $STAMP"
say "outdir  : $OUT"

# ---------------------------------------------------------------------------
hdr "1. identity and the covariates the manifest does not record"
# Protocol items 16: kernel and toolkit are run variables. cor04's kernel
# changed 7.0.0-28 -> -29 mid-campaign and separates its long- and short-TTF
# faults perfectly; nothing recorded it, so it was invisible for a whole
# campaign. Capture both, every time, on every node.
{
    echo "hostname       : $(hostname)"
    echo "uname -r       : $(uname -r)"
    echo "uname -v       : $(uname -v)"
    echo "uptime_seconds : $(cut -d. -f1 /proc/uptime)"
    echo "boot_utc       : $(date -u -d "@$(( $(date +%s) - $(cut -d. -f1 /proc/uptime) ))" +%Y-%m-%dT%H:%M:%SZ)"
    if have nvcc; then echo "nvcc           : $(nvcc --version | tail -1)"
    else                echo "nvcc           : NOT FOUND (set CUDA_HOME)"; fi
    if have nvidia-smi; then
        echo "driver         : $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
        echo "gpu_count      : $(nvidia-smi -L | wc -l)"
        echo "gpu_name       : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    else
        echo "driver         : nvidia-smi NOT FOUND"
    fi
    if [[ -x ./pcieburn ]]; then
        echo "binary sha256  : $(sha256sum ./pcieburn | cut -c1-16)"
    else
        echo "binary sha256  : ./pcieburn not built here"
    fi
    echo "git HEAD       : $(git rev-parse --short=7 HEAD 2>/dev/null || echo 'not a git repo')"
    echo "pciutils       : $(lspci --version 2>/dev/null | head -1 || echo 'lspci NOT FOUND')"
} > "$OUT/identity.txt"
cat "$OUT/identity.txt"

# ---------------------------------------------------------------------------
hdr "2. driver query fields this plan depends on (enumerate, never assume)"
# The power x cooling arm needs clocks_event_reasons_counters.* and
# temperature.gpu.tlimit. They exist on 595 and not on 580. Prove it here
# rather than discovering it after a six-cell matrix has been run.
if have nvidia-smi; then
    nvidia-smi --help-query-gpu > "$OUT/nvidia_smi_query_fields.txt" 2>&1
    for f in temperature.gpu.tlimit clocks_event_reasons_counters.sw_power_cap \
             clocks_event_reasons_counters.hw_power_brake \
             clocks_event_reasons_counters.sw_thermal_slowdown \
             clocks_event_reasons_counters.hw_thermal_slowdown ; do
        if grep -q -- "$f" "$OUT/nvidia_smi_query_fields.txt"; then
            v=$(nvidia-smi --query-gpu="$f" --format=csv,noheader 2>&1 | head -1)
            say "  PRESENT  $f  -> $v"
        else
            say "  ABSENT   $f  <-- the power x cooling arm cannot use this"
        fi
    done
    cap "power limits"  gpu_power.txt  nvidia-smi -q -d POWER
    cap "clocks"        gpu_clocks.txt nvidia-smi -q -d CLOCK
    cap "topo -m"       topo_m.txt     nvidia-smi topo -m
    cap "topo -p2p r"   topo_p2p.txt   nvidia-smi topo -p2p r
    say ""
    say "  Min/max enforceable power limit per GPU (clamp the ladder to this;"
    say "  300 W may be below the floor on this SKU):"
    nvidia-smi --query-gpu=index,power.min_limit,power.max_limit,power.limit,power.default_limit \
        --format=csv 2>&1 | sed 's/^/    /'
else
    say "  nvidia-smi not present — sections 2 and 8 are unavailable on this host"
fi

# ---------------------------------------------------------------------------
hdr "2b. ECC — an observable the fleet structurally does not have"
# The fleet's RTX 5090s are consumer parts with no ECC, so memory errors there
# are undetectable except through pcieburn's own GEMM comparison (faulty/nan),
# which has been zero in 38 of 38 runs. This SKU has ECC, which means:
#   - it can report memory faults the fleet would never surface, and
#   - ECC mode is itself a run variable. It costs memory bandwidth, which moves
#     duty cycle, which moves power. Toggling it makes a NEW arm, not the same
#     arm at a different setting. Record the mode with every run.
# An uncorrectable error here is a memory-subsystem fault, NOT a link fault.
# Do not fold it into the PCIe corpus.
if have nvidia-smi; then
    cap "ECC state"      ecc.txt          nvidia-smi -q -d ECC
    cap "row remapper"   row_remapper.txt nvidia-smi -q -d ROW_REMAPPER
    say ""
    say "  ECC mode (current/pending) per GPU:"
    grep -E 'Ecc Mode|Current|Pending' "$OUT/ecc.txt" 2>/dev/null | head -12 | sed 's/^/    /' \
        || say "    not reported"
    say ""
    say "  ECC-related query fields this driver exposes:"
    nvidia-smi --help-query-gpu 2>/dev/null | grep -iE 'ecc|remap|retired' \
        | sed 's/^/    /' | head -20 || say "    none"
    say ""
    say "  Counters (enumerate the field names above before trusting these):"
    for f in ecc.errors.uncorrected.volatile.total \
             ecc.errors.uncorrected.aggregate.total \
             ecc.errors.corrected.volatile.total ; do
        if nvidia-smi --help-query-gpu 2>/dev/null | grep -q -- "$f"; then
            nvidia-smi --query-gpu=index,"$f" --format=csv 2>&1 | sed 's/^/    /'
        else
            say "    ABSENT: $f"
        fi
    done
    say ""
    say "  Any nonzero UNCORRECTABLE count, or any pending row remap, means this"
    say "  box is not a clean comparison platform until it is resolved. Localise"
    say "  it with 'dcgmi diag -r 2' before running any pcieburn arm."
else
    say "  nvidia-smi not present — ECC section unavailable"
fi

# ---------------------------------------------------------------------------
hdr "3. PCIe error-handling ownership — THE GATE for this platform"
# _OSC ownership is a BIOS setting: it differed between COR04 and RGCA with
# identical board and identical BIOS version. If it is settable on the MGX box,
# flipping to OS-first restores aer_delta.txt, makes DPC kernel-managed, and
# makes every number here comparable to the fleet. It also turns the
# firmware-first <-> OS-first pair into a one-variable test of h11's surviving
# limb, on identical hardware.
DMESG_OK=0
if dmesg --ctime > "$OUT/dmesg.txt" 2>/dev/null && [[ -s "$OUT/dmesg.txt" ]]; then
    DMESG_OK=1
elif sudo -n dmesg --ctime > "$OUT/dmesg.txt" 2>/dev/null && [[ -s "$OUT/dmesg.txt" ]]; then
    DMESG_OK=1
fi
if [[ $DMESG_OK -eq 1 ]]; then
    {
        echo "--- _OSC negotiation ---"
        grep -E '_OSC.*(does not support|OS now controls|control from|granted)' "$OUT/dmesg.txt" | sort -u
        echo
        echo "--- counts ---"
        printf 'AER lines            : %s\n' "$(grep -c 'AER' "$OUT/dmesg.txt")"
        printf 'DPC: enabled ports   : %s\n' "$(grep -c 'DPC: enabled' "$OUT/dmesg.txt")"
        printf 'Hardware Error (GHES): %s\n' "$(grep -c 'Hardware Error' "$OUT/dmesg.txt")"
        printf 'EDR lines            : %s\n' "$(grep -ci 'edr' "$OUT/dmesg.txt")"
        printf 'APEI/GHES init       : %s\n' "$(grep -cE 'APEI|GHES' "$OUT/dmesg.txt")"
        echo
        echo "--- kernel command line ---"
        grep -m1 'Kernel command line' "$OUT/dmesg.txt"
        echo
        echo "--- IOMMU state (h11: already enabled on every fleet node) ---"
        grep -E 'iommu: Default domain type|AMD-Vi|DMAR: IOMMU|Intel-IOMMU' "$OUT/dmesg.txt" | head -8
    } > "$OUT/error_ownership.txt"
    cat "$OUT/error_ownership.txt"
    if grep -q 'does not support.*AER' "$OUT/error_ownership.txt"; then
        say ""
        say "  >>> FIRMWARE-FIRST confirmed. Consequences, both silent:"
        say "      - run_pcieburn.sh --with-aer reads sysfs counters that will"
        say "        NEVER increment here. aer_delta.txt will be structurally"
        say "        zero and will look exactly like a clean result."
        say "      - containment recovery goes through EDR, not kernel DPC."
        say "      Check the BIOS for an AER/WHEA ownership option before running"
        say "      any arm. See section 5 for the config-space fallback."
    fi
else
    say "  dmesg unreadable. Fix with: sudo sysctl kernel.dmesg_restrict=0"
    say "  This section is REQUIRED — do not proceed without it."
fi

# ---------------------------------------------------------------------------
hdr "4. is EDR compiled in? (firmware-first recovery depends on it)"
KCFG="/boot/config-$(uname -r)"
if [[ -r "$KCFG" ]]; then
    grep -E 'CONFIG_PCIE_(EDR|DPC|AER)|CONFIG_ACPI_APEI(_GHES)?=' "$KCFG" \
        | tee "$OUT/kernel_config.txt" | sed 's/^/  /'
    grep -q '^CONFIG_PCIE_EDR=y' "$KCFG" \
        || say "  >>> CONFIG_PCIE_EDR is NOT y. On a firmware-first platform a"
    grep -q '^CONFIG_PCIE_EDR=y' "$KCFG" \
        || say "      contained link is then not OS-recoverable at all, which"
    grep -q '^CONFIG_PCIE_EDR=y' "$KCFG" \
        || say "      changes what a 'fault' means on this box."
elif [[ -r /proc/config.gz ]]; then
    zgrep -E 'CONFIG_PCIE_(EDR|DPC|AER)' /proc/config.gz | tee "$OUT/kernel_config.txt" | sed 's/^/  /'
else
    say "  no readable kernel config ($KCFG absent, /proc/config.gz absent)"
fi

# ---------------------------------------------------------------------------
hdr "5. per-link AER status/mask and Lane Error Status, from config space"
# This is the readout that replaces the blind sysfs counters. It reads the
# registers by NAME out of lspci's own decode, so no capability offsets are
# hardcoded — reciting offsets from memory is how this investigation got
# 'power.enforced_limit' wrong once already.
#
# What to look for:
#   CEMsk  bit names with '-' = UNMASKED = that error type is reported.
#          The fleet's three victim GPUs read RxErr- BadTLP- ... Timeout+,
#          i.e. only Replay Timer Timeout masked.
#   CESta  is what has actually latched. It latches regardless of the mask,
#          which is why it is stronger evidence than any counter.
#   LaneErrStat  localises which lanes take the hits. Never yet collected on
#          any platform in this investigation.
if have lspci; then
    if ! sudo -n lspci -vvv >/dev/null 2>&1 && [[ $EUID -ne 0 ]]; then
        say "  WARNING: lspci -vvv without root omits capability bodies."
        say "           Re-run this script with sudo for a usable section 5."
    fi
    LSPCI="lspci"; [[ $EUID -ne 0 ]] && sudo -n true 2>/dev/null && LSPCI="sudo -n lspci"
    $LSPCI -vvv > "$OUT/lspci_vvv.txt" 2>"$OUT/lspci_vvv.err" || true
    $LSPCI -tv  > "$OUT/lspci_tree.txt" 2>/dev/null || true

    # Every NVIDIA endpoint, and every bridge above it.
    mapfile -t GPUS < <(lspci -D -n 2>/dev/null | awk '$2 ~ /^0300|^0302/ && /10de:/ {print $1}')
    if [[ ${#GPUS[@]} -eq 0 ]]; then
        mapfile -t GPUS < <(lspci -D 2>/dev/null | grep -iE 'NVIDIA' | awk '{print $1}')
    fi
    say "  NVIDIA functions found: ${#GPUS[@]}"

    {
        printf '%-16s %-10s %s\n' "bdf" "kind" "decoded AER / lane registers"
        for bdf in "${GPUS[@]}"; do
            # walk up: the endpoint plus each parent bridge, from sysfs
            chain=("$bdf")
            p="/sys/bus/pci/devices/$bdf"
            while [[ -L "$p" ]]; do
                parent=$(basename "$(dirname "$(readlink -f "$p")")")
                [[ "$parent" =~ ^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$ ]] || break
                chain+=("$parent"); p="/sys/bus/pci/devices/$parent"
            done
            for d in "${chain[@]}"; do
                kind="bridge"; [[ "$d" == "$bdf" ]] && kind="endpoint"
                block=$(awk -v dev="$d" '
                    $0 ~ "^"dev {inblk=1; next}
                    inblk && /^[0-9a-f]{4}:/ {inblk=0}
                    inblk {print}' "$OUT/lspci_vvv.txt")
                ce=$(printf '%s\n' "$block" | grep -m1 'CESta:'      | sed 's/^[[:space:]]*//')
                cm=$(printf '%s\n' "$block" | grep -m1 'CEMsk:'      | sed 's/^[[:space:]]*//')
                ue=$(printf '%s\n' "$block" | grep -m1 'UESta:'      | sed 's/^[[:space:]]*//')
                le=$(printf '%s\n' "$block" | grep -m1 'LaneErrStat' | sed 's/^[[:space:]]*//')
                ls=$(printf '%s\n' "$block" | grep -m1 'LnkSta:'     | sed 's/^[[:space:]]*//')
                printf '%-16s %-10s %s\n' "$d" "$kind" "${ls:-<no LnkSta decoded>}"
                printf '%-16s %-10s   %s\n' "" "" "${ce:-<no CESta — pciutils too old, or needs root>}"
                printf '%-16s %-10s   %s\n' "" "" "${cm:-<no CEMsk>}"
                printf '%-16s %-10s   %s\n' "" "" "${ue:-<no UESta>}"
                printf '%-16s %-10s   %s\n' "" "" "${le:-<no LaneErrStat — check pciutils version>}"
            done
            echo
        done
    } > "$OUT/aer_and_lane_registers.txt"
    say "  wrote $OUT/aer_and_lane_registers.txt"
    say ""
    say "  CEMsk summary across all NVIDIA endpoints and their bridges:"
    grep -h 'CEMsk:' "$OUT/aer_and_lane_registers.txt" | sort | uniq -c | sed 's/^/    /'
    say ""
    say "  Any lane already flagged (should be all-zero on a clean box):"
    grep -h 'LaneErrStat' "$OUT/aer_and_lane_registers.txt" | grep -v 'LaneErrStat: 0' \
        | sed 's/^/    /' || say "    none"

    # 32 GT/s equalization presets — the per-boot record h10 and mechanism 1 need
    if grep -q 'Physical Layer 32' "$OUT/lspci_vvv.txt" 2>/dev/null; then
        grep -A12 'Physical Layer 32' "$OUT/lspci_vvv.txt" > "$OUT/phy32_caps.txt"
        say "  32.0 GT/s Physical Layer capability decoded -> phy32_caps.txt"
    else
        say "  32.0 GT/s Physical Layer capability NOT decoded by this pciutils."
        say "  Find its offset in the extended capability list and read it with"
        say "  setpci ECAP_ID+off — do NOT hardcode an offset from memory."
    fi
else
    say "  lspci not installed — section 5 unavailable"
fi

# ---------------------------------------------------------------------------
hdr "6. sysfs AER counters — the wrapper's data source, and whether it works"
found=0; nonzero=0
{
    for d in /sys/bus/pci/devices/*/; do
        [[ -r "$d/aer_dev_correctable" ]] || continue
        found=$((found+1))
        s=$(awk '{s+=$2} END{print s+0}' "$d/aer_dev_correctable")
        [[ "$s" -gt 0 ]] && nonzero=$((nonzero+1))
        printf '%s correctable_sum=%s\n' "$(basename "$d")" "$s"
    done
} > "$OUT/sysfs_aer.txt"
found=$(wc -l < "$OUT/sysfs_aer.txt")
nonzero=$(awk -F= '$2+0>0' "$OUT/sysfs_aer.txt" | wc -l)
say "  devices exposing aer_dev_correctable : $found"
say "  of those, currently nonzero          : $nonzero"
if [[ $found -eq 0 ]]; then
    say "  >>> No sysfs AER counters at all. --with-aer will produce nothing."
elif [[ $nonzero -eq 0 ]] && grep -q 'does not support.*AER' "$OUT/error_ownership.txt" 2>/dev/null; then
    say "  >>> Counters exist but the kernel does not own AER, so they will stay"
    say "      at zero no matter what the link does. This is the blind-instrument"
    say "      case: a zero here is NOT evidence. Use section 5 instead, and"
    say "      validate that reader on a fleet node where these counters DO move."
fi

# ---------------------------------------------------------------------------
hdr "7. BMC sensors — real names on THIS unit, not the fleet's"
# The wrapper hardcodes CUR_PSU1_IOUT / CUR_PSU2_IOUT / VOLT_12V. On the MGX
# box those do not exist, so --with-psu self-disables and the one platform with
# unsaturated PSU telemetry silently produces no PSU data.
if have ipmitool; then
    cap "sdr list"   ipmi_sdr.txt   sudo -n ipmitool sdr list
    cap "sel list"   ipmi_sel.txt   sudo -n ipmitool sel list
    if [[ -s "$OUT/ipmi_sdr.txt" ]]; then
        say ""
        say "  power / current / ambient sensors on this unit:"
        grep -iE 'psu|pwr|power|iout|volt|12v|amb|inlet|temp' "$OUT/ipmi_sdr.txt" \
            | sed 's/^/    /' | head -40
        say ""
        for n in CUR_PSU1_IOUT CUR_PSU2_IOUT VOLT_12V; do
            if grep -q "^$n" "$OUT/ipmi_sdr.txt"; then
                say "    $n: present (wrapper --with-psu will work as-is)"
            else
                say "    $n: ABSENT  <-- wrapper --with-psu will self-disable here"
            fi
        done
        say ""
        say "  Ambient sensor for the h14 proxy validation:"
        grep -iE 'amb|inlet' "$OUT/ipmi_sdr.txt" | sed 's/^/    /' || say "    none found"
    fi
else
    say "  ipmitool not installed — section 7 unavailable"
fi

# ---------------------------------------------------------------------------
hdr "8. trained link state, per link, before any load"
# Protocol item 13. Record this at EVERY boot. rgca17's gpu6 came up at gen1
# for 265 s on one boot, clean x16 on another, and x8 twice on a third, and its
# error counts followed the trained state rather than the workload.
if [[ -x ./linkcheck.sh ]]; then
    if sudo -n true 2>/dev/null; then
        sudo -n ./linkcheck.sh --csv > "$OUT/linkcheck.csv" 2>"$OUT/linkcheck.err" \
            && say "  wrote $OUT/linkcheck.csv" \
            || say "  linkcheck.sh failed — see linkcheck.err"
        sudo -n ./linkcheck.sh > "$OUT/linkcheck.txt" 2>/dev/null || true
        [[ -s "$OUT/linkcheck.txt" ]] && grep -E 'DEGRADED|LOW|SWITCH|Topology' "$OUT/linkcheck.txt" | sed 's/^/    /'
    else
        say "  linkcheck.sh needs root. Run: sudo ./linkcheck.sh --csv"
    fi
else
    say "  ./linkcheck.sh not found in \$PWD"
fi

# ---------------------------------------------------------------------------
hdr "done"
say "artifacts in: $OUT"
say ""
say "Next, in order:"
say "  1. Read section 3. If firmware-first, check the BIOS for an AER/WHEA"
say "     ownership option BEFORE running any load arm. That toggle is worth"
say "     more than any single arm — it restores the graded measurement and"
say "     makes this box comparable to the fleet."
say "  2. Run this same script on cptcor04 and diff sections 5 and 6. That is"
say "     what tells you whether the config-space reader is trustworthy here."
say "  3. Then MGX_RUNPLAN.md."
