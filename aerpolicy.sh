#!/usr/bin/env bash
#
# aerpolicy.sh — dump the AER *policy* registers for every link in each GPU's
# PCIe path, and flag where this platform deviates from the PCIe spec defaults.
#
# Belongs next to linkcheck.sh in tools/pcieburn/.
#
# linkcheck.sh answers "is the link healthy". aer_uncorrectable.csv answers
# "which error bit fired". This answers the question in between, which neither
# covers: *given* an error bit, does this machine treat it as ERR_FATAL?
#
# That is not a property of the error. It is a property of two registers the
# BIOS writes at POST:
#
#   Uncorrectable Error Severity (AER+0x0C)  1 = report as ERR_FATAL
#   Correctable Error Mask       (AER+0x14)  1 = never reported at all
#
# On the TURIN2D24G the BIOS initialises these from AMD PBS setup options
# (AMD_RAS15 root port / AMD_RAS18 device for severity, AMD_RAS13/16 for the
# correctable mask), and the shipped values are NOT the spec defaults:
#
#   severity  0x07EF6030 vs spec 0x00062030  -- 9 extra bits escalated to FATAL
#   corr mask 0x00001000 vs spec 0x00002000  -- Replay Timer Timeout suppressed
#
# The consequences are why this script exists:
#
#   * Completion Timeout is FATAL here and non-fatal per spec. A GPU that merely
#     stops answering -- hang, thermal, power droop -- produces CmpltTO, which on
#     this board becomes ERR_FATAL, which arms DPC, which drops the link. The
#     error need not be a signal-integrity event at all.
#   * Replay Timer Timeout is masked, so it generates no AER message, so the
#     kernel never counts it. A zero in the "Timeout" column of aer_delta.txt is
#     therefore NOT evidence that replay timers are not expiring -- the raw
#     status bit below is the only place that shows up.
#
# Both registers are static and readable at any time, so unlike the endpoint's
# UESta this needs no during-run sampling and survives a containment event.
#
# Usage:
#   sudo ./aerpolicy.sh            # table
#   sudo ./aerpolicy.sh --verbose  # add lspci's symbolic decode per port
#
# Exit status: 0 policy matches spec defaults everywhere, 1 deviations found,
# 2 could not run.

set -uo pipefail

VERBOSE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--verbose) VERBOSE=1; shift ;;
        -h|--help)    sed -n '2,44p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option '$1' (try --help)" >&2; exit 2 ;;
    esac
done

command -v lspci  >/dev/null 2>&1 || { echo "ERROR: lspci not found (pciutils)"  >&2; exit 2; }
command -v setpci >/dev/null 2>&1 || { echo "ERROR: setpci not found (pciutils)" >&2; exit 2; }

if [[ $EUID -ne 0 ]] && ! sudo -n true 2>/dev/null; then
    echo "ERROR: needs root (or passwordless sudo) to read extended config space" >&2
    exit 2
fi
PFX=""; [[ $EUID -ne 0 ]] && PFX="sudo"

# PCIe Base Spec reset values.
SPEC_UESVRT=$((0x00062030))   # DLP, SDES, FCP, RxOF, MalfTLP
SPEC_CEMSK=$((0x00002000))    # Advisory Non-Fatal only

declare -A UE=(
  [4]=DLP [5]=SDES [12]=PoisonedTLP [13]=FCP [14]=CmpltTO [15]=CmpltAbrt
  [16]=UnxCmplt [17]=RxOF [18]=MalfTLP [19]=ECRC [20]=UnsupReq [21]=ACSViol
  [22]=UncorrIntErr [23]=BlockedTLP [24]=AtomicOpBlocked [25]=TLPPrefixBlocked
  [26]=PoisonedTLPEgress )
declare -A CE=(
  [0]=RxErr [6]=BadTLP [7]=BadDLLP [8]=Rollover [12]=ReplayTimerTO
  [13]=AdvNonFatal [14]=CorrIntErr [15]=HeaderLogOverflow )

# decode <value> <name-of-bit-table>
decode() {
    local v=$1 out="" b names
    local -n tbl=$2
    for b in $(printf '%s\n' "${!tbl[@]}" | sort -n); do
        (( (v >> b) & 1 )) && out+="${tbl[$b]} "
    done
    printf '%s' "${out:-none}"
}

# rd <bdf> <aer-offset> -> decimal value, or nothing if no AER capability
rd() {
    local v
    v=$($PFX setpci -s "$1" "ECAP_AER+$2.l" 2>/dev/null) || return 1
    [[ -n "$v" ]] || return 1
    printf '%s' $((16#$v))
}

pci_chain() {
    readlink -f "/sys/bus/pci/devices/$1" 2>/dev/null | tr '/' '\n' \
        | grep -E '^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$'
}

declare -a EPS=()
for d in /sys/bus/pci/devices/*; do
    [[ -r "$d/vendor" && -r "$d/class" ]] || continue
    [[ "$(cat "$d/vendor")" == "0x10de" ]] || continue
    case "$(cat "$d/class")" in 0x0300*|0x0302*) EPS+=("$(basename "$d")") ;; esac
done
[[ ${#EPS[@]} -eq 0 ]] && { echo "ERROR: no NVIDIA display/3D devices in sysfs" >&2; exit 2; }

RC=0
declare -A SEEN=()

for ep in $(printf '%s\n' "${EPS[@]}" | sort); do
    mapfile -t CHAIN < <(pci_chain "$ep")
    (( ${#CHAIN[@]} < 2 )) && { echo "WARNING: no PCIe path for $ep" >&2; continue; }

    echo "=== GPU $ep ==============================================="
    for bdf in "${CHAIN[@]}"; do
        if [[ -n "${SEEN[$bdf]:-}" ]]; then echo "  $bdf  (shown above)"; continue; fi
        SEEN[$bdf]=1

        ptype=$($PFX lspci -vvv -s "$bdf" 2>/dev/null \
                | sed -n 's/.*Express (v[0-9]) \([^,(]*\).*/\1/p' | head -1 | sed 's/ *$//')
        echo "  $bdf  ${ptype:-Endpoint}"

        svrt=$(rd "$bdf" 0c) || { echo "      no AER capability"; continue; }
        uemsk=$(rd "$bdf" 08); uesta=$(rd "$bdf" 04)
        cemsk=$(rd "$bdf" 14); cesta=$(rd "$bdf" 10)

        printf '      UESvrt 0x%08X  FATAL: %s\n' "$svrt" "$(decode "$svrt" UE)"
        if (( svrt != SPEC_UESVRT )); then
            printf '        deviates from spec 0x%08X -- extra FATAL: %s\n' \
                "$SPEC_UESVRT" "$(decode $(( svrt & ~SPEC_UESVRT )) UE)"
            RC=1
        fi
        printf '      UEMsk  0x%08X  suppressed: %s\n' "$uemsk" "$(decode "$uemsk" UE)"
        printf '      UESta  0x%08X  latched: %s\n'    "$uesta" "$(decode "$uesta" UE)"
        (( uesta )) && RC=1

        printf '      CEMsk  0x%08X  suppressed: %s\n' "$cemsk" "$(decode "$cemsk" CE)"
        if (( cemsk != SPEC_CEMSK )); then
            printf '        deviates from spec 0x%08X -- also suppressed: %s\n' \
                "$SPEC_CEMSK" "$(decode $(( cemsk & ~SPEC_CEMSK )) CE)"
            RC=1
        fi
        # Raw status latches even for masked errors -- the only place a masked
        # Replay Timer Timeout is visible.
        printf '      CESta  0x%08X  latched: %s\n' "$cesta" "$(decode "$cesta" CE)"

        # DPC has its own capability; lspci decodes it, setpci has no alias.
        $PFX lspci -vvv -s "$bdf" 2>/dev/null \
            | grep -E '^[[:space:]]*(DPCCap|DPCCtl|DPCSta|Source):' | sed 's/^[[:space:]]*/      /'

        if (( VERBOSE )); then
            $PFX lspci -vvv -s "$bdf" 2>/dev/null \
                | grep -E '^[[:space:]]*(UESta|UEMsk|UESvrt|CESta|CEMsk|AERCap):' \
                | sed 's/^[[:space:]]*/      | /'
        fi
    done
    echo
done

cat <<'NOTE'
Reading this:
  UESvrt bit set  -> that error is signalled as ERR_FATAL, which is what arms DPC
                     ("DPCCtl: Trigger:1" means armed on ERR_FATAL only).
  CEMsk bit set   -> that correctable error sends no AER message, so the kernel's
                     /sys/.../aer_dev_correctable counter stays 0 for it. CESta
                     still latches it. Compare the two before concluding absence.
NOTE
exit $RC
