# MGX / RTX PRO 6000 run plan — exact commands

Companion to the pre-registration in `HYPOTHESES.md`
(*Third platform: intelmgx-CG480*). Every flag below was read out of
`run_pcieburn.sh`, `pcieburn.cu` and `linkcheck.sh` rather than recalled —
verify against `./run_pcieburn.sh --help` and `./pcieburn --help` on the box if
anything drifts.

Two standing constraints on this platform: **driver 595, kernel 6.8 generic**,
neither ours to change.

> **2026-08-22:** GPU `0000:99:00.0` is out of service pending RMA. Run the
> six-GPU pair-preserving configuration in section 0b, never a seven-GPU one.

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

## 0b. Config change: six GPUs, pair-preserving

GPU `0000:99:00.0` is out of service pending RMA.

**Exclude the whole PIX pair, not just the bad GPU.** The topology class is four
switches with two GPUs each, so dropping only `99:00.0` leaves its partner alone
behind that switch and changes that switch's uplink loading. Six GPUs in three
symmetric pairs is a self-consistent arm; seven is a different topology.

**Verify the pair map on this unit before trusting it** — the 0/1, 2/3, 4/5, 6/7
pairing on record was taken on the previous, compromised box:

```sh
nvidia-smi topo -m
nvidia-smi --query-gpu=index,pci.bus_id,uuid --format=csv
```

**Exclude by UUID, not by index.** `pcieburn` calls `cudaSetDevice()` straight on
the `--gpus` index, never logs a bus ID, and CUDA enumerates `FASTEST_FIRST` by
default rather than in PCI order, so an index is not a reliable handle on a
specific card. Put this in the shell that launches every arm:

```sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# the three healthy pairs: 1A/1B, 3F/40, BB/BC.
# excluded: 99:00.0 (defective) and its PIX partner 9A:00.0.
export CUDA_VISIBLE_DEVICES=\
GPU-e833048e-2953-e92f-5c83-9081030596ad,\
GPU-016df175-986b-a632-ea79-341ee9d53780,\
GPU-6f8408fc-8f6e-10cc-f64a-7f3c113c5397,\
GPU-8371f9c8-04bc-6ee4-ea1f-fd953079027a,\
GPU-47893b29-f25b-1270-b9b9-45fd94d4a8f2,\
GPU-e518f8d7-d862-409f-dddf-6f96f227be4c
```

For reference, the two excluded cards: `99:00.0` is
`GPU-4c2698bd-24de-38c8-1dde-a62030124dbf`, its partner `9A:00.0` is
`GPU-2454d2b3-d22e-9e32-4230-03363aa87b1b`. UUIDs are stable across reboots;
re-read them with `nvidia-smi -L` if a card is ever replaced.

With that set, `pcieburn` sees six devices and needs no `--gpus`. Use `sudo -E`
so the wrapper inherits it (already the case in every command below).

**Label these as their own arm.** Add `-6gpu` to every tag and never compare a
six-GPU run against an eight-GPU one. Note the exclusion and its reason in the
sidecar.

Unaffected: the `--gpus 0,1` PIX-pair cells in section 6, provided they use a
healthy pair.

## 0c. Added to the arrival procedure

`mgx_preflight.sh` section 2b now fails a blocking **arrival gate** on any
nonzero aggregate uncorrectable ECC count, any row-remap failure, any pending
repair, or any bank at `None` remap availability. Run it on arrival for every
borrowed machine, before any load:

```sh
sudo ./mgx_preflight.sh
```

Aggregate ECC counters are an acceptance check on the part's whole history.
Volatile counters are what you read after a run. Do not substitute one for the
other.

## 0d. Per-run readout added

ECC is a run variable on this platform and is absent from the fleet, so capture
it per run alongside the existing telemetry:

```sh
nvidia-smi -q -d ECC          > "$D/ecc_before.txt"     # $D = the run dir
nvidia-smi -q -d ROW_REMAPPER > "$D/rowremap_before.txt"
# ... run the arm ...
nvidia-smi -q -d ECC          > "$D/ecc_after.txt"
nvidia-smi -q -d ROW_REMAPPER > "$D/rowremap_after.txt"
```

Also record `Ecc Mode: Current` in the sidecar. ECC costs memory bandwidth,
which moves duty cycle, which moves power — so if it is ever toggled, that is a
new arm, not the same arm at a different setting.

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
