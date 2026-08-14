#!/usr/bin/env bash
#
# linkcheck.sh — zero-cost PCIe link health screen for GPU nodes.
#
# Walks the full PCIe path from each GPU up to the root complex and reports the
# negotiated speed/width of EVERY link in that path against each port's own
# capability. Runs no load, touches no GPU state, and cannot take a node down.
#
# Why the full path and not just the GPU's parent: on machines with a PCIe
# switchboard the topology is
#     root port -> switch upstream port -> switch downstream port -> GPU
# so the GPU's immediate parent is the switch downstream port, and the
# root-port-to-switch link is invisible if you only look one hop up. That
# upstream link carries every GPU behind the switch and is the most
# consequential single point of failure on such a board. On riser machines the
# path is just root port -> GPU and this reduces to one link per GPU.
#
# Why the downstream side of each link: reading an ENDPOINT's config space
# requires the link to be in L0, so polling the GPU can itself wake a link out
# of a low-power state and perturb the measurement. Every link is therefore read
# from its downstream-facing port (Root Port or Downstream Port), whose Link
# Status describes the same link. GPU enumeration is pure sysfs for the same
# reason (no NVML, no driver calls).
#
# Interpreting the verdict, which depends on ASPM:
#
#   DEGRADED  current speed/width below capability WHILE ASPM IS DISABLED.
#             There is no power-management explanation for this — the port is
#             configured to hold its maximum and is not doing so.
#   LOW       below capability but ASPM is enabled, so a low idle speed may be
#             legitimate link power management. Re-check under load.
#   OK        at full capability.
#
# This distinction is load-bearing. On cptcor04 all eight root ports report
# "ASPM Disabled" with Target Link Speed 32GT/s, yet one GPU idles at 2.5GT/s
# while seven hold 32GT/s. That is a defect indicator, not power saving — and
# the GPU showing it is one of the two that have historically fallen off the bus.
#
# Each link is compared against ITS OWN capability, so a switch that is
# legitimately Gen4, or a bifurcated x8 slot, reads OK rather than DEGRADED.
#
# Usage:
#   sudo ./linkcheck.sh                 # table
#   sudo ./linkcheck.sh --csv           # machine-readable
#   sudo ./linkcheck.sh --with-index    # add nvidia-smi indices (uses NVML,
#                                       # which may wake the GPUs — off by default)
#   watch -n 10 'sudo ./linkcheck.sh'   # watch links over time
#
# Exit status: 0 all OK, 1 at least one DEGRADED or LOW, 2 could not run.

set -uo pipefail

CSV=0
WITH_INDEX=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv)        CSV=1; shift ;;
        --with-index) WITH_INDEX=1; shift ;;
        -h|--help)    sed -n '2,50p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option '$1' (try --help)" >&2; exit 2 ;;
    esac
done

command -v lspci >/dev/null 2>&1 || {
    echo "ERROR: lspci not found (install pciutils)" >&2; exit 2; }

# lspci needs privilege to read the extended config space these fields live in.
if [[ $EUID -ne 0 ]] && ! sudo -n true 2>/dev/null; then
    echo "ERROR: needs root (or passwordless sudo) to read PCIe capabilities" >&2
    exit 2
fi
LSPCI="lspci"
[[ $EUID -ne 0 ]] && LSPCI="sudo lspci"

# --- enumerate GPUs from sysfs only -----------------------------------------
# vendor 0x10de = NVIDIA; class 0x0300xx = VGA controller, 0x0302xx = 3D
# controller. Reading these sysfs attributes returns values the kernel cached at
# enumeration, so it generates no config traffic.
declare -a EPS=()
for d in /sys/bus/pci/devices/*; do
    [[ -r "$d/vendor" && -r "$d/class" ]] || continue
    [[ "$(cat "$d/vendor")" == "0x10de" ]] || continue
    case "$(cat "$d/class")" in
        0x0300*|0x0302*) EPS+=("$(basename "$d")") ;;
    esac
done

if [[ ${#EPS[@]} -eq 0 ]]; then
    echo "ERROR: no NVIDIA display/3D devices found in sysfs" >&2
    exit 2
fi

# --- optional index map (this DOES use NVML) --------------------------------
declare -A IDX=()
if [[ $WITH_INDEX -eq 1 ]] && command -v nvidia-smi >/dev/null 2>&1; then
    while IFS=',' read -r i bus; do
        i="${i// /}"
        bdf=$(printf '%s' "$bus" | tr -d ' ' | tr 'A-Z' 'a-z' | sed 's/^0000//')
        [[ -n "$bdf" ]] && IDX["$bdf"]="$i"
    done < <(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader 2>/dev/null)
fi

# Full PCIe ancestry for a device, root-most first, from the resolved sysfs path.
# On a riser board this yields "<root port> <gpu>"; behind a switch it yields
# "<root port> <switch upstream> <switch downstream> <gpu>".
pci_chain() {
    readlink -f "/sys/bus/pci/devices/$1" 2>/dev/null | tr '/' '\n' \
        | grep -E '^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$'
}

# --- report -----------------------------------------------------------------
if [[ $CSV -eq 1 ]]; then
    echo "gpu,endpoint,hop,port,port_type,cap_speed,cap_width,cur_speed,cur_width,aspm,target_speed,verdict"
else
    printf '%-4s %-13s %-3s %-13s %-11s %-9s %-9s %-5s %-9s %-9s %s\n' \
        GPU ENDPOINT HOP PORT TYPE CAP CURRENT WIDTH ASPM TARGET VERDICT
fi

RC=0
SWITCHED=0

for ep in $(printf '%s\n' "${EPS[@]}" | sort); do
    mapfile -t CHAIN < <(pci_chain "$ep")
    if [[ ${#CHAIN[@]} -lt 2 ]]; then
        echo "WARNING: cannot resolve PCIe path for $ep; skipping" >&2
        continue
    fi
    [[ ${#CHAIN[@]} -gt 2 ]] && SWITCHED=1

    hop=0
    # Every element except the endpoint itself is a bridge in the path. Only the
    # downstream-facing ones (Root Port / Downstream Port) describe a distinct
    # link; a switch Upstream Port reports the same link as the root port above
    # it, so including it would double-count.
    for (( i = 0; i < ${#CHAIN[@]} - 1; i++ )); do
        port="${CHAIN[$i]}"

        read -r ptype cap_sp cap_w cur_sp cur_w aspm tgt < <(
            $LSPCI -vvv -s "$port" 2>/dev/null | awk '
                /Express \(v[0-9]\)/ {
                    if (!pt && match($0, /Express \(v[0-9]\) [^,(]*/)) {
                        pt = substr($0, RSTART, RLENGTH)
                        sub(/Express \(v[0-9]\) /, "", pt)
                        gsub(/ +$/, "", pt); gsub(/ /, "_", pt)
                    } }
                /LnkCap:/  { for (i=1;i<=NF;i++) {
                                 if ($i=="Speed") { cs=$(i+1); sub(/,$/,"",cs) }
                                 if ($i=="Width") { cw=$(i+1); sub(/,$/,"",cw) } } }
                /LnkSta:/  { for (i=1;i<=NF;i++) {
                                 if ($i=="Speed") { ss=$(i+1); sub(/,$/,"",ss) }
                                 if ($i=="Width") { sw=$(i+1); sub(/,$/,"",sw) } } }
                /LnkCtl:/  { if (match($0, /ASPM [^;]*/)) {
                                 a=substr($0, RSTART+5, RLENGTH-5); gsub(/ /,"_",a) } }
                /LnkCtl2:/ { if (match($0, /Target Link Speed: [^,]*/)) {
                                 t=substr($0, RSTART+19, RLENGTH-19); gsub(/ /,"",t) } }
                END { printf "%s %s %s %s %s %s %s\n",
                        (pt?pt:"?"), (cs?cs:"?"), (cw?cw:"?"), (ss?ss:"?"),
                        (sw?sw:"?"), (a?a:"?"), (t?t:"?") }')

        case "$ptype" in
            Root_Port|Downstream_Port) ;;
            *) continue ;;   # Upstream Port duplicates the link above it
        esac
        hop=$((hop + 1))

        capn=${cap_sp%GT/s}; curn=${cur_sp%GT/s}
        capw=${cap_w#x};     curw=${cur_w#x}

        verdict=OK
        below=0
        # Speeds are non-integer ("2.5GT/s"), so compare in awk. Guard on "?" so
        # a failed parse is never reported as degraded.
        if [[ "$cur_sp" != "?" && "$cap_sp" != "?" ]] \
           && awk -v a="$curn" -v b="$capn" 'BEGIN{exit !(a+0 < b+0)}'; then
            below=1
        fi
        if [[ "$curw" =~ ^[0-9]+$ && "$capw" =~ ^[0-9]+$ ]] && (( curw < capw )); then
            below=1
        fi
        if [[ $below -eq 1 ]]; then
            if [[ "$aspm" == "Disabled" ]]; then verdict=DEGRADED; else verdict=LOW; fi
            RC=1
        fi

        g="${IDX[$ep]:--}"
        if [[ $CSV -eq 1 ]]; then
            echo "$g,$ep,$hop,$port,$ptype,$cap_sp,$cap_w,$cur_sp,$cur_w,$aspm,$tgt,$verdict"
        else
            printf '%-4s %-13s %-3s %-13s %-11s %-9s %-9s %-5s %-9s %-9s %s\n' \
                "$g" "$ep" "$hop" "$port" "$ptype" "$cap_sp" "$cur_sp" \
                "$cur_w" "$aspm" "$tgt" "$verdict"
        fi
    done
done

if [[ $CSV -eq 0 ]]; then
    echo
    if [[ $SWITCHED -eq 1 ]]; then
        echo "Topology: PCIe SWITCH detected (more than one link per GPU path)."
        echo "  hop 1 is the shared upstream link — a fault there contains EVERY"
        echo "  GPU behind the switch, unlike a riser board where each GPU has"
        echo "  its own independent root port."
        echo
    fi
    if [[ $RC -eq 0 ]]; then
        echo "All links at full capability."
    else
        echo "DEGRADED = below capability with ASPM disabled: no power-management"
        echo "           explanation, treat as a hardware indicator."
        echo "LOW      = below capability but ASPM enabled: may be legitimate."
        echo
        echo "A link can be brought back up without a reboot:"
        echo "  sudo setpci -s <port> CAP_EXP+10.w=0020:0020   # Retrain Link"
        echo "Whether it then HOLDS the speed while idle is the diagnostic."
    fi
fi
exit $RC
