# MGX / RTX PRO 6000 run plan — exact commands

Companion to the pre-registration in `HYPOTHESES.md`
(*Third platform: intelmgx-CG480*). Every flag below was read out of
`run_pcieburn.sh`, `pcieburn.cu` and `linkcheck.sh` rather than recalled —
verify against `./run_pcieburn.sh --help` and `./pcieburn --help` on the box if
anything drifts.

Two standing constraints on this platform: **driver 595, kernel 6.8 generic**,
neither ours to change.

> **Open issue, 2026-08-21.** The first run on this box reported an
> **uncorrectable ECC error at iteration 0**. That is a memory-subsystem fault,
> not a link fault, and it must not be folded into the PCIe corpus — but it does
> mean the box is not yet a clean comparison platform. **Work through
> "ECC triage" below before running any arm in this plan.**

---

## 0. Before anything: build, then preflight

```sh
cd ~/pcieburn
make                                  # SM ?= 120 is already correct for GB202
./mgx_preflight.sh                    # read-only; re-run with sudo for sections 5/7/8
sudo ./mgx_preflight.sh
```

Then **run the same script on `cptcor04`** and diff sections 5 and 6. That
comparison is the only thing that tells you whether the config-space register
reader is trustworthy here — on a firmware-first platform the sysfs AER counters
never increment, and you cannot distinguish "no errors" from "blind instrument"
without having watched the instrument work somewhere it does.

**Stop and read section 3 of the preflight output before running any load arm.**
If it reports firmware-first, check the BIOS for an AER/WHEA ownership option.
That toggle is worth more than any single arm below.

---

## 0b. ECC triage — blocking, do this before any arm

An uncorrectable ECC error is a **different subsystem** from the PCIe link fault
this investigation is about. It gets triaged on its own terms and kept out of the
fault corpus. Two reasons it still blocks everything: a card with a memory defect
is not the clean same-die/different-board comparison the whole platform plan
rests on, and if it recurs mid-arm it will terminate runs for a reason unrelated
to what the arm is testing.

### The one question that sets the severity

Did `pcieburn`'s own GEMM comparison see corrupted results, or did ECC contain it?

```sh
D=$(ls -1dt ~/pcieburn/runs/*/ | head -1); echo "$D"
# events.csv fields: 11 = faulty, 12 = nans
awk -F, 'NR==1 || $11+0>0 || $12+0>0' "$D/events.csv" | head
grep -iE 'faulty|nan' "$D/pcieburn.log" | tail -20
```

- **`faulty` / `nans` still zero** → ECC did its job: it detected and reported
  rather than letting bad data through. Bad news about the card, no change to the
  investigation's standing claim.
- **`faulty` or `nans` nonzero** → this is the **first data corruption observed
  anywhere in this investigation**, across 38 fleet runs plus this one. Flag it
  distinctly and loudly; it is a materially more serious class of finding than
  anything in the corpus, and it does not belong in the same bucket as the link
  fault.

### What the kernel and driver saw

```sh
sudo dmesg --ctime | grep -E 'Xid|NVRM|ECC' | tail -40
nvidia-smi -q -d ECC          | sed -n '1,80p'
nvidia-smi -q -d ROW_REMAPPER
nvidia-smi --help-query-gpu | grep -iE 'ecc|remap|retired'   # enumerate first
```

Xid codes to look up rather than assume — check each against NVIDIA's Xid table,
because the meaning differs by architecture: **48** (double-bit ECC), **63 / 64**
(row-remap recording, and recording *failure*), **92** (high single-bit rate),
**94 / 95** (contained / uncontained ECC error). The contained-versus-uncontained
distinction is the one that matters most: *contained* means the blast radius was
one process, *uncontained* means the GPU's state is untrustworthy until reset.

Then check whether a remap is pending. A pending row remap needs a GPU reset to
apply, and until it applies the same address will keep failing:

```sh
nvidia-smi -q -d ROW_REMAPPER | grep -iE 'pending|failure|remapped'
sudo nvidia-smi -r -i <N>          # per-GPU reset; needs no compute clients
```

### Localise it before deciding

```sh
sudo dcgmi diag -r 2 2>&1 | tee dcgmi_r2_ecc.log     # includes the memory tests
sudo dcgmi diag -r 3 -p "diagnostic.is_allowed=true" 2>&1 | tee dcgmi_r3_ecc.log
```

### Why iteration 0 is informative rather than alarming on its own

`pcieburn` defaults to `--mem-frac 0.9`, so it allocates 90% of remaining VRAM
for C matrices and **iteration 0 is the first time nearly the whole framebuffer
gets touched**. On a 96 GB card that is a large first-touch sweep. An
uncorrectable error there reads much more like a pre-existing bad cell that only
a full-VRAM pass reaches than like load-induced degradation — the load had not
had time to do anything yet. That also means the harness is incidentally a
first-touch VRAM exerciser, which is worth knowing but is not a substitute for a
real pattern-based memory test.

### Decision

| finding | action |
|---|---|
| reproduces on one GPU after reset and remap | that card is an RMA. Do not use this box for board comparison until it is replaced — a defective part invalidates the one property that made it valuable. |
| clears after reset + row remap, does not return over ~3 × 3600 s idle-plus-load | usable, but record the remap in every subsequent run's sidecar; a remapped row is a permanent change to the part under test. |
| reproduces on multiple GPUs | suspect the platform or the driver-595/kernel-6.8 pairing, not the cards. Check whether ECC mode is even fully supported in that combination before blaming hardware. |
| `faulty`/`nans` nonzero | stop, preserve the bundle, and treat it as its own finding. Do not continue the arm plan on this box. |

### A workaround that is not free

Excluding the affected GPU with `--gpus 0,1,2,3,4,5,6` keeps the box usable, but
it is **a new arm, not the same arm**: seven ranks change the collective shape,
the coherent power step, and the per-switch load distribution. Record it as its
own arm label and never compare it against an 8-GPU run.

Likewise, disabling ECC to work around this changes memory bandwidth, which
moves duty cycle, which moves power. Same rule: new arm, not the same arm at a
different setting. Record `Ecc Mode: Current` in the sidecar either way.

### One angle worth watching, not asserting

Card-local power delivery is the one plausible common cause for memory errors and
PHY errors on the same board — it is mechanism 2 in the ledger's ranking
(GPU-local PHY supply disturbance from the card's own load transients). If ECC
error rate on this box turns out to scale with the power cap, that is worth
following. At iteration 0 on a first run, though, a defective memory device is
far more likely, and a single event cannot distinguish them. Do not build on it.

---

## 1. Tag and sidecar

The manifest records the driver but not the kernel or the toolkit, and cor04's
kernel changed mid-campaign without anything noticing. Until the wrapper records
them (protocol item 16), carry them in the tag and drop a sidecar.

```sh
# paste once per shell session
export PB=~/pcieburn
cd "$PB"

mktag() {                       # mktag <arm-label> <cap-watts>
    printf '%s-%s-pl%s-up%s-g%s-b%s-k%s\n' \
        "$(hostname)" "$1" "$2" \
        "$(cut -d. -f1 /proc/uptime)" \
        "$(git rev-parse --short=7 HEAD)" \
        "$(sha256sum "$PB/pcieburn" | cut -c1-8)" \
        "$(uname -r | tr -d '.-' | tail -c 8)"
}

sidecar() {                     # sidecar — call immediately after a run
    local d; d=$(ls -1dt "$PB"/runs/*/ | head -1)
    {
        echo "uname_r        : $(uname -r)"
        echo "nvcc           : $(nvcc --version | tail -1)"
        echo "p2p_disable    : ${NCCL_P2P_DISABLE:-unset}"
        echo "fan_profile    : ${FAN_PROFILE:-unset}"
        echo "osc_ownership  : ${OSC_MODE:-unset}"
        echo "sensors:"; sudo -n ipmitool sdr list 2>/dev/null \
            | grep -iE 'psu|iout|power_out|amb|inlet|fan' | sed 's/^/  /'
    } > "$d/platform.txt"
    echo "wrote $d/platform.txt"
}
```

Set `FAN_PROFILE` and `OSC_MODE` by hand each time you change either, so the
sidecar records the condition rather than your memory of it.

---

## 2. The two calibration arms — establish achieved stress first

Matched *flags* are not matched stress: byte-identical flags already gave a 4.0×
collective-throughput difference between cor04 and RGCA. More SMs, a 600 W
default and P2P-on will push these somewhere else again, so these two runs exist
to measure where, not to hunt a fault.

`NCCL_P2P_DISABLE=1` is the **baseline**, not one arm of an A/B — host-staged
collectives are what put traffic on the PCIe link at all, and with P2P on the
byte accounting is invalid (`PCIE_HOST_STAGING_MULT` assumes staging).

```sh
# clamp to whatever the preflight reported as power.max_limit
sudo nvidia-smi -pl 600
nvidia-smi --query-gpu=index,power.limit --format=csv        # confirm it took

# A1 — 2048 mixed, uncapped-equivalent. pcieburn defaults; nothing after `--`.
NCCL_P2P_DISABLE=1 sudo -E ./run_pcieburn.sh \
    --tag "$(mktag 2048mixed 600)" \
    --duration 600 --settle 30 \
    --with-nvml --nvml-interval 100 \
    --with-dmon --with-aer --aer-interval 1 \
    --with-psu --psu-interval 0.25 \
    --yes
sidecar

# A2 — 8192 single, the gpu-burn-shaped arm
NCCL_P2P_DISABLE=1 sudo -E ./run_pcieburn.sh \
    --tag "$(mktag 8192single 600)" \
    --duration 600 --settle 30 \
    --with-nvml --nvml-interval 100 \
    --with-dmon --with-aer --aer-interval 1 \
    --with-psu --psu-interval 0.25 \
    --yes \
    -- --no-tensor --precision single --matrix-dim 8192
sidecar
```

Two expected non-results, both silent, both already documented:

- `--with-psu` will **self-disable** — the wrapper probes `CUR_PSU1_IOUT`, which
  this platform does not have. Leave the flag on so the warning lands in the log;
  read the real sensor names out of the sidecar until the override exists.
- `--with-aer` will produce a **structurally zero** `aer_delta.txt` under
  firmware-first. Leave it on for the baseline file, but read the direction
  result out of the config-space registers, not out of that file.

Then read back what you actually applied, and compare against the fleet on
achieved numbers rather than on flags:

```sh
python3 corr_matrix.py 2>&1 | head -50      # COVERAGE line first, always
```

*Prediction: no fault, no SEL entry, no CPER.* The informative readout is the
correctable **direction split**, not the verdict:

| outcome | what it retires |
|---|---|
| GPU→upstream correctables appear, upstream→GPU stays zero | asymmetry is GB202/PHY-generic — retires card-local PHY supply (mech 2) and connector TX-pin degradation (mech 3) as *necessary*, promotes die-side EQ drift (mech 1) |
| zero in both directions | asymmetry stays a 5090-board / fleet-connector property — supports mech 2 and 3, points at the VRM and the CEM connector |
| errors in both directions | new phenomenon; different receiver silicon, so treat as platform-specific and do not pool |

---

## 3. The `_OSC` A/B — run this first if the BIOS exposes the setting

One arm, two BIOS settings, identical hardware and load. This is the direct
probe of what survives of h11: the IOMMU limb is refuted (all three fleet nodes
already boot `amd_iommu=on iommu=pt` and fault anyway), so what is left is the
error-handling path — and this isolates it without needing a passthrough guest.

```sh
# --- pass 1: as found (firmware-first) ---
export OSC_MODE=firmware-first
NCCL_P2P_DISABLE=1 sudo -E ./run_pcieburn.sh \
    --tag "$(mktag 2048mixed-oscFW 600)" \
    --duration 600 --settle 30 \
    --with-nvml --with-dmon --with-aer --with-psu --yes
sidecar

# --- reboot into BIOS, flip AER/DPC ownership to OS-native, boot, then: ---
sudo ./mgx_preflight.sh        # section 3 must now show "OS now controls"
export OSC_MODE=os-first
NCCL_P2P_DISABLE=1 sudo -E ./run_pcieburn.sh \
    --tag "$(mktag 2048mixed-oscOS 600)" \
    --duration 600 --settle 30 \
    --with-nvml --with-dmon --with-aer --with-psu --yes
sidecar
```

*Prediction: OS-first escalates to a contained GPU where firmware-first absorbs
or retries.* If OS-first also restores non-zero `aer_delta.txt`, every
subsequent arm on this box becomes directly comparable to the fleet — run
everything below in OS-first mode from then on.

---

## 4. Power × cooling — the decoupling the fleet cannot do

The power ladder on its own would just re-create the fleet's power/ΔT
collinearity on new hardware. A passively cooled server card with BMC fan
control is the only instrument in this investigation that can move ΔT
independently of power, which is exactly what h16 needs.

**Read the power floor first** — 300 W may be below `power.min_limit` on this
SKU, and the preflight prints it:

```sh
nvidia-smi --query-gpu=index,power.min_limit,power.max_limit --format=csv
```

### Fan control — do not guess

Get the fan-control command from the vendor's BMC documentation, or use the
Redfish `Thermal`/`ThermalSubsystem` schema, or the BMC web UI. **Do not invent
`ipmitool raw` bytes**: on a 600 W passively cooled card a wrong value is a
thermal event, not a failed command. Verify before every run that fans actually
moved and that idle temperatures are where you expect:

```sh
sudo ipmitool sdr list | grep -i fan
nvidia-smi --query-gpu=index,temperature.gpu,temperature.gpu.tlimit --format=csv
```

Abort the reduced-fan cells if `temperature.gpu.tlimit` margin goes below the
vendor's throttle threshold at idle.

### The matrix

Six cells, arm A2 throughout, 600 s each:

```sh
for W in 600 450 "$FLOOR"; do            # FLOOR from power.min_limit
  for FAN in max reduced; do
    # set fans to $FAN via the vendor's documented path, verify RPM, then:
    export FAN_PROFILE="$FAN"
    sudo nvidia-smi -pl "$W"
    NCCL_P2P_DISABLE=1 sudo -E ./run_pcieburn.sh \
        --tag "$(mktag "8192single-fan${FAN}" "$W")" \
        --duration 600 --settle 30 \
        --with-nvml --nvml-interval 100 \
        --with-dmon --with-aer --with-psu --yes \
        -- --no-tensor --precision single --matrix-dim 8192
    sidecar
  done
done
```

Run the six cells in **randomised blocks, not three-of-A then three-of-B**
(protocol item 14) — replicate index carries campaign time.

Readout per cell, none of it a verdict:

```sh
nvidia-smi --query-gpu=index,\
clocks_event_reasons_counters.sw_power_cap,\
clocks_event_reasons_counters.hw_power_brake,\
clocks_event_reasons_counters.sw_thermal_slowdown,\
clocks_event_reasons_counters.hw_thermal_slowdown \
  --format=csv
```

Cumulative seconds under each throttle cause is the rigorous replacement for the
clip% proxy, which failed at r = −0.04 against fault over 37 fleet runs. Sample
it immediately before and after each run and difference it.

*Prediction: correctable and lane-error rate track power and not ΔT if mechanism
2 dominates; the reverse if mechanism 4 does.*

---

## 5. Boot-training loop — cheapest arm, sharpest prediction

No load at all. Eight reboots, capturing the trained state and the per-lane
equalization presets each time.

```sh
for i in $(seq 1 8); do
    sudo ./mgx_preflight.sh --outdir "./boot-$i"
    sudo reboot                    # wait for the box, then continue the loop
done
```

*Prediction (h10): zero training variance on this platform — the variance is a
fleet BIOS/riser property, not GB202/driver.* rgca17's gpu6 came up three
different ways across three boots (265 s at gen1; clean x16; x8 twice) and its
error counts followed the trained state rather than the workload, so a flat
result here localises that to the fleet.

Compare across boots:

```sh
for d in boot-*/; do echo "== $d"; cat "$d"*/linkcheck.csv 2>/dev/null | tail -n +2; done
```

---

## 6. Cheap nulls — run when the box is otherwise idle

```sh
# P2P enabled (the platform default). Byte accounting invalid here — relative only.
sudo nvidia-smi -pl 600
sudo -E ./run_pcieburn.sh \
    --tag "$(mktag 2048mixed-p2pON 600)" \
    --duration 600 --settle 30 --with-nvml --with-dmon --with-aer --with-psu --yes
sidecar

# One PIX pair only: collective traffic never leaves a single switch.
NCCL_P2P_DISABLE=1 sudo -E ./run_pcieburn.sh \
    --tag "$(mktag 8192single-pix01 600)" \
    --duration 600 --settle 30 --with-nvml --with-dmon --with-aer --with-psu --yes \
    -- --no-tensor --precision single --matrix-dim 8192 --gpus 0,1
sidecar

# The original reproducer, as a cross-check.
sudo dcgmi diag -r 3 -p "diagnostic.is_allowed=true" 2>&1 | tee dcgmi_r3.log
```

*Prediction: no behavioural difference in fault terms.* Downgraded from the
earlier pre-registration — the corpus re-read strengthened the traffic
exoneration considerably: cor04 ran 59% of Gen5 x16 for 606 s twice with zero
AER and no fault, and r(egress, fault) = −0.18 across 37 runs.

---

## 7. Free calibration, worth taking while the box is idle

The retroactive ambient figures that eliminated ambient as the mediator of the
cor04 staircase came from pre-load idle GPU temperature, with no real ambient
sensor to check them against. This box has one.

```sh
# log both for ~20 min at idle, then correlate
( while :; do
    printf '%s,%s,%s\n' "$(date -u +%FT%TZ)" \
      "$(sudo -n ipmitool sdr list 2>/dev/null | grep -iE 'amb|inlet' | head -1 | awk -F'|' '{print $2}' | tr -d ' ')" \
      "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | sort -n | head -1)"
    sleep 5
  done ) | tee ambient_vs_proxy.csv
```

If the idle-GPU proxy tracks the real sensor here, the fleet-wide ambient
elimination gains independent support across all 38 runs. One sensor read.

---

## 8. Not in this plan, and why

**The cross-board slot swap** — a PRO 6000 into cor04's GPU5 slot (`a0:01.1`,
three fatals, zero correctables ever) — is the sharpest version of h6's swap
because it changes the board *class*, not the serial number. It is deliberately
excluded: a 600 W passively cooled card into a chassis built for axial-fan
cards, borrowed hardware into a node that has hard-faulted seven times, and
mechanical fit in a `TURIN2D24G-2L+` chassis that nobody has checked. **Do the
intra-fleet 5090↔5090 swap first** — same logic, no exposure, and it answers
"slot or card" before anyone decides whether the cross-board version is worth
the risk.

**The cor04 kernel question** — kernel 6.8 here is a third point on that axis
with the die, board, host, topology, driver and error path all moving with it.
Still needs the cor04 `7.0.0-28` downgrade run, which remains the cheapest
decisive run in the whole investigation:

```sh
# on cptcor04, not here
sudo grub-reboot "…7.0.0-28-generic…" && sudo reboot
# then, at ~700 s uptime:
sudo nvidia-smi -pl 575
sudo -E ./run_pcieburn.sh --tag "$(mktag 2048mixed-k28 575)" \
    --duration 600 --settle 30 --with-nvml --with-dmon --with-aer --with-psu --yes
```

*TTF back near 200–280 s implicates the kernel and dissolves the wear staircase;
still near 60 s removes the kernel and leaves wear versus campaign time.*

---

## Safety

`pcieburn` is trying to reproduce a fault that has previously required a hard
power cycle, and on a switch-node topology containment takes every GPU behind
the switch. Only run it on a designated test node. The `--yes` flags above skip
the wrapper's interactive confirmation — remove them for the first run on a new
box so you read the warning once.
