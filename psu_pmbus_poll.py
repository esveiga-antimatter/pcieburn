#!/usr/bin/env python3
"""Per-PSU input/output power over the ASRock PMBus bridge.

THE COMMAND
-----------
    ipmitool raw 0x3a 0x52 0x0c <addr> 0x02 <pmbus_cmd>
             |    |    |    |     |     |    '- PMBus command code
             |    |    |    |     |     '------ bytes to read
             |    |    |    |     '------------ PSU address, 8-bit form
             |    |    |    '------------------ I2C bus
             '----'----'----------------------- ASRock OEM PMBus bridge

Addresses (8-bit): PSU1 0xb0, PSU2 0xb2, PSU3 0xb4, PSU4 0xb6.
PSU3 and PSU4 are the two the BMC does not surface.

DATA FORMAT
-----------
READ_PIN/READ_POUT/READ_IOUT are PMBus LINEAR11: a 16-bit little-endian word
holding a signed 5-bit exponent in bits 15..11 and a signed 11-bit mantissa in
bits 10..0, value = mantissa * 2**exponent.

READ_VOUT is *not* LINEAR11. It is LINEAR16: the whole 16-bit word is an
UNSIGNED mantissa and the exponent is not in the word at all -- it comes from
VOUT_MODE (0x20), a separate 1-byte register. Feeding a LINEAR16 word to the
LINEAR11 decoder reads a 12 V rail as a few tens of millivolts, silently. Do
not assume one decoder covers every register.

VOUT_MODE is configuration, not telemetry, so it is read once at startup rather
than every round, and it is not taken on trust: a bridge that answers 0x00 to a
command it does not implement decodes as a valid "Linear, exponent 0", which
would put 6190 V in the CSV without raising a single error. The exponent is
therefore confirmed against one READ_VOUT and rejected unless it places the rail
within 50% of --vout-nominal; when it is rejected, or VOUT_MODE cannot be read
at all, the exponent is derived instead as round(log2(nominal / mantissa)).
That fixes the power-of-two scale from the nominal rail and leaves the measured
variation -- the sag, which is the thing of interest -- untouched.

VERIFIED vs ASSUMED
-------------------
ASRock supplied 0x97 (READ_PIN). 0x96 (READ_POUT) and 0x8c (READ_IOUT) are the
PMBus standard codes for the other two quantities but are NOT confirmed on this
platform. Run `--validate` first: it reads IOUT over PMBus for the two supplies
the BMC *can* see and compares against the BMC's own sensors. If those agree, the
bridge, address map, endianness and decode are all correct at once -- after which
the identical decode can be trusted for PSU3/PSU4, which have no second opinion.

VOUT, IOUT, POUT and PIN are all polled by default, for all four supplies, so a
capture carries the 12 V rail and the output current alongside the power. That
costs cadence: the reads are serial at ~30 ms each, so four registers across
four supplies puts the floor on a round near 0.48 s -- hence the 0.55 s default
-- and a --interval below that is simply not met (the script says so once on
stderr, and every row still records its ACHIEVED interval).

Narrower rounds sample faster, and the right width depends on the question.
`--cmds 0x8b 0x8c` is an 8-read rail-and-current round at 4 Hz, which is what to
use when a 12 V transient is the target rather than a power envelope.
`--cmds 0x96 0x97` is the original power-only round.

VOUT does not sum. The four supplies load-share onto one common 12 V bus, which
is exactly why system_iout_a is meaningful and a voltage total would not be:
these are four measurements of ONE rail, so the informative aggregate is the
SPREAD between them. A common sag moves all four together and shows up in the
per-supply columns; one supply losing its share of the bus shows up in
vout_spread_v and nowhere else.

psuN_vout_derived_v = POUT/IOUT is kept even though READ_VOUT is now polled
directly, because the two together are the only check on the LINEAR16 exponent:
agreement confirms it, while disagreement by a clean power of two says the
exponent is wrong rather than the rail sagging. Treat it as a check and not a
second measurement -- its two reads are ~30 ms apart.

ANALYSIS: USE MEDIANS, NOT MEANS, UNDER LOAD
-------------------------------------------
Measured on cptroca25 against a paired NVML trace. At idle the instrument is
tight: per-sample efficiency median 89.7%, p5 88.2%, p95 94.0%, max 95.3%, and
system POUT reconciles with NVML to within 30 W across separate idle periods.

Under load it is not. The four PSU reads in a round are ~30 ms apart and
system_pout_w sums them, so a single stale-low reading drags the whole sum down
while it takes all four running high to lift it. The result is a one-sided low
tail: on a 183 s steady-load run, 288 of 658 summed samples (44%) came out BELOW
the concurrent GPU-only draw, which is impossible, and the mean sat 478 W below
the median.

So aggregate over a window with a MEDIAN. On that run the mean gave a
rest-of-system of -275 W (impossible) and the median gave +203 W. Per-sample
efficiency under load reaches 285%, which is likewise a pairing artifact and not
a calibration fault -- POUT and PIN within a round are not simultaneous either.

Note this is NOT the pass-level workload oscillation: the same behaviour appears
with the 93.6%-duty 8192 arm as with the 54%-duty default, so it is read-level
staleness rather than aliasing against the GEMM/collective cycle.

Residual after using medians is consistent with NVML over-reporting GPU board
power by roughly 8%, which reconciles idle and load together. Treat that as a
working estimate, not a measurement -- it rests on rest-of-system being roughly
constant, and settling it properly needs a metered PDU or a clamp on the 12 V
cables.

SAFETY
------
Only ever issue PMBus *read* codes through this bridge. The same OEM command can
write, and a PMBus write to a live supply (OPERATION, VOUT_COMMAND, ...) can
reconfigure or shut it down. This script hard-codes a read-only allowlist and
refuses anything outside it.

CONFIRMED ON HARDWARE (cptroca25, 2026-08-26). 70 rounds, 100% read success,
0.250 s mean/median/p95 interval with zero gaps -- 4 Hz steady, against the older
BMC-sensor poller's ~1 Hz with dozens of multi-second gaps per run. Per-supply
efficiency came out 89.9-93.1% (system 91.3%), which is where a CRPS unit belongs
and is strong evidence that 0x96 really is READ_POUT and that the LINEAR11 decode
is right -- four independent supplies would not all land in the correct band by
accident. PSU3 and PSU4, invisible to the BMC, read normally.

SENSOR RATE (measured on cptroca25 at --interval 0.1). The transport sustains
10 Hz at 100% success -- 80 raw IPMI reads/s -- but the readings do not. Values
change on only 19-33% of polls, with runs of 3.0-5.2 identical samples, implying
a sensor update period of 300-520 ms, i.e. ~2-3 Hz. Quantization is 1.0 W on PIN
and 0.25 W on POUT. Note the measurement was taken at idle, where a constant load
also produces repeats, so this BOUNDS the sensor at no faster than ~2-3 Hz rather
than pinning it; separating staleness from constancy needs a varying load, where
a slow sensor draws staircases whose tread length is the update period.

This is also why adding READ_IOUT to the default round costs less than it looks.
The four-register round floors out near 0.48 s, so the default backs off to
0.55 s -- 1.8 Hz against the two-register round's 4 Hz. That is now BELOW the
sensor's own ~2-3 Hz, so unlike the three-register round it does give up real
updates, and it is a deliberate trade: a wide round measures the whole envelope
at slightly under the sensor rate, a narrow one measures two channels at twice
that. Choose per question rather than leaving the default in place for a
transient hunt. The
round is wider in wall-clock, not poorer in information. There is a reason not to
push the other way either -- the moment this data matters most is a fault, when
the BMC is also logging SEL entries and handling the power event, and 80
transactions/s of extra housekeeping risks perturbing what you are trying to
capture.

The other cost is intra-round skew: 16 serial reads instead of 8 means the first
and last value in a round are ~0.48 s apart rather than ~0.24 s, which widens
the same read-level staleness the median guidance above exists to absorb.
Aggregate system_iout_a with a median for exactly the reason system_pout_w needs
one, and more so. The exception is vout_spread_v, where the point is the
excursion: take its MAXIMUM, and take the per-supply minimum, because a median
over a run that was mostly fine hides the moment worth looking at.

This is an envelope instrument. It cannot speak to the microsecond current edges
that hypotheses 9 and 12 concern; it answers whether system power did something
sustained and unusual.

Still worth running --validate once per platform: the efficiency check confirms
the command codes and the decode, but only the BMC cross-check confirms the
ADDRESS MAP, i.e. that 0xb0 is the supply the BMC calls PSU1.
"""

import argparse
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

# --- read-only allowlist. Nothing outside this set is ever sent. --------------
# Fields: column name, decoder, system-total column (None if it does not sum),
# that total's decimal places, and how many bytes the read returns.
#
# Precision tracks each register's own quantization: 1.0 W on PIN, 0.25 W on
# POUT, with IOUT and VOUT fine enough to be worth two places.
#
# VOUT does NOT sum. Summing IOUT across supplies is meaningful because these
# four load-share onto one common 12 V output bus -- but that same fact makes a
# sum of their voltages meaningless: they are four measurements of ONE rail, so
# the useful aggregate is the spread between them, not the total. On a machine
# where each supply fed a separate rail, system_iout_a would be the nonsense
# one instead.
#
# VOUT_MODE (0x20) is configuration, not telemetry: it is read once at startup
# to get the LINEAR16 exponent and never enters the polling round.
PMBUS_READS = {
    0x20: ("vout_mode",   "raw8",     None,            0, 1),
    0x79: ("status_word", "raw16",    None,            0, 2),
    0x8b: ("vout_v",      "linear16", None,            2, 2),
    0x8c: ("iout_a",      "linear11", "system_iout_a", 2, 2),
    0x96: ("pout_w",      "linear11", "system_pout_w", 1, 2),
    0x97: ("pin_w",       "linear11", "system_pin_w",  1, 2),
}
PSUS = {1: 0xb0, 2: 0xb2, 3: 0xb4, 4: 0xb6}
BUS = 0x0c

# Measured cost of one bridged PMBus read over local KCS, from the ~30 ms
# spacing between the four PSU reads of a round. Used only to warn when the
# requested interval cannot be met -- see the cadence note in poll().
SECONDS_PER_READ = 0.030

HEXLINE = re.compile(r'^\s*(?:[0-9a-fA-F]{2}\s*)+$')


def linear11(word):
    """PMBus LINEAR11 -> float. Exponent bits 15..11 signed, mantissa 10..0 signed."""
    exp = (word >> 11) & 0x1f
    if exp > 15:
        exp -= 32
    man = word & 0x7ff
    if man > 1023:
        man -= 2048
    return man * (2.0 ** exp)


def linear16(word, exponent):
    """PMBus LINEAR16 -> float. Unlike LINEAR11 the mantissa is the whole
    16-bit word and UNSIGNED, and the exponent is not in the word at all: it
    comes from VOUT_MODE. Feeding a LINEAR16 word to linear11() above would
    read a 12 V rail as something in the tens of millivolts, silently."""
    return word * (2.0 ** exponent)


def vout_mode(byte):
    """VOUT_MODE (0x20) -> (mode, exponent). Bits 7..5 select the encoding
    (0 = Linear, 1 = VID, 2 = Direct); bits 4..0 are a signed 5-bit exponent.
    Only mode 0 is a LINEAR16 rail -- the other two need a whole different
    conversion, so they are reported rather than guessed at."""
    mode = (byte >> 5) & 0x07
    exp = byte & 0x1f
    if exp > 15:
        exp -= 32
    return mode, exp


def ipmi_prefix(args):
    if args.bmc:
        # lanplus opens a network session per invocation -- far slower than local
        # KCS. Batching matters even more in this mode.
        return ["ipmitool", "-I", "lanplus", "-H", args.bmc,
                "-U", args.user, "-P", args.password]
    return (["sudo", "-n"] if args.sudo else []) + ["ipmitool"]


def build_batch(psus, cmds):
    """One ipmitool `exec` script for the whole round. Order is significant.

    Returns (lines, nbytes) -- the byte count is per command now that not every
    readable register is two bytes wide, and run_batch needs it to know whether
    a short reply is the whole answer or a truncated one.
    """
    lines, nb = [], []
    for p in psus:
        for c in cmds:
            n = PMBUS_READS[c][4]
            lines.append(f"raw 0x3a 0x52 {BUS:#04x} {PSUS[p]:#04x} {n:#04x} {c:#04x}")
            nb.append(n)
    return lines, nb


def run_batch(args, lines, nbytes=None):
    """Run the batch; return (words, err). words is None when the round is
    unreliable, and err then carries WHY -- ipmitool reports every failure on
    stderr and prints nothing to stdout, so discarding stderr (as an earlier
    version did) turns a one-line diagnosis into thousands of blank rows.

    Alignment matters. A failed read emits no stdout line, which would silently
    shift every later value onto the wrong PSU. So a round is accepted only when
    the hex-line count equals the command count -- attributing PSU4's power to
    PSU3 would be far worse than a gap in the trace.
    """
    nbytes = nbytes or [2] * len(lines)
    if args.no_batch:
        words = []
        for one, n in zip(lines, nbytes):
            w, err = _run_one(args, one, n)
            if w is None:
                return None, err
            words.append(w)
        return words, ""

    with tempfile.NamedTemporaryFile("w", suffix=".ipmi", delete=False) as fh:
        fh.write("\n".join(lines) + "\n")
        path = fh.name
    try:
        p = subprocess.run(ipmi_prefix(args) + ["exec", path],
                           capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except OSError as e:
        # ipmitool absent or not executable. This is the same class of failure
        # preflight exists to report, so return it as an error string and let
        # that ladder print -- a traceback into psu_pmbus.err told the operator
        # far less than "command not found" plus the commands to try by hand.
        return None, f"cannot run ipmitool: {e.strerror}"
    finally:
        os.unlink(path)

    words = []
    for ln in p.stdout.splitlines():
        if not HEXLINE.match(ln):
            continue
        b = [int(x, 16) for x in ln.split()]
        # Which width to expect is positional: reply k answers command k. A
        # reply shorter than its command asked for is a failed round, not a
        # value -- assembling a 16-bit word from one byte would read half a
        # rail voltage as the whole thing.
        want = nbytes[len(words)] if len(words) < len(nbytes) else 2
        if len(b) < want:
            return None, f"short response: {ln.strip()!r} (wanted {want} byte(s))"
        words.append(sum(b[i] << (8 * i) for i in range(want)))   # little-endian
    if len(words) == len(lines):
        return words, ""
    err = " | ".join(l.strip() for l in p.stderr.splitlines()[:2]) or \
          f"got {len(words)} of {len(lines)} responses, no stderr"
    return None, err


def _run_one(args, cmdline, nbytes=2):
    """One raw command, one ipmitool. Slower, but isolates which read fails."""
    try:
        p = subprocess.run(ipmi_prefix(args) + cmdline.split(),
                           capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except OSError as e:
        return None, f"cannot run ipmitool: {e.strerror}"
    for ln in p.stdout.splitlines():
        if HEXLINE.match(ln):
            b = [int(x, 16) for x in ln.split()]
            if len(b) >= nbytes:
                return sum(b[i] << (8 * i) for i in range(nbytes)), ""
    return None, " | ".join(l.strip() for l in p.stderr.splitlines()[:2]) or "no output"


def preflight(args):
    """One read before the loop. An unreachable BMC, missing sudo, or an
    unsupported OEM command should stop the run with the reason on screen, not
    produce a CSV full of read_ok=0."""
    n = PMBUS_READS[args.cmds[0]][4]
    probe = f"raw 0x3a 0x52 {BUS:#04x} {PSUS[1]:#04x} {n:#04x} {args.cmds[0]:#04x}"
    w, err = _run_one(args, probe, n)
    if w is not None:
        return True
    sys.stderr.write(
        "psu_pmbus_poll: preflight read failed, not polling.\n"
        f"  command: {' '.join(ipmi_prefix(args))} {probe}\n"
        f"  error  : {err}\n\n"
        "  Work down this ladder on the host to find the layer that breaks:\n"
        "    sudo -n ipmitool mc info                       # local IPMI at all?\n"
        "    sudo -n ipmitool sensor reading CUR_PSU1_IOUT   # BMC sensor path?\n"
        "    sudo -n ipmitool raw 0x3a 0x52 0x0c 0xb0 0x02 0x97   # OEM bridge?\n"
        "  If the third works but this script does not, re-run with --no-batch.\n")
    return False


def scan(args):
    """Probe the PMBus address range for supplies we are not polling.

    This exists because a cross-check against NVML showed the four named
    supplies reporting LESS output than the GPUs alone consume under load: the
    POUT-vs-GPU slope is 0.92 below 1 kW of GPU draw but only 0.27 above 2.5 kW,
    i.e. the four carry nearly everything at idle and about a quarter of the
    marginal load at power. That is the signature of further supplies coming
    online as load rises -- and a system total summed over four of them is then
    badly wrong in exactly the regime that matters.

    Read-only: one READ_PIN per address, which is the same command ASRock
    supplied.
    """
    print("Probing 8-bit PMBus addresses 0xb0..0xbe (7-bit 0x58..0x5f) "
          "with READ_PIN (0x97).")
    print("Known: 0xb0=PSU1 0xb2=PSU2 0xb4=PSU3 0xb6=PSU4\n")
    found = []
    for addr in range(0xb0, 0xc0, 2):
        w, err = _run_one(args, f"raw 0x3a 0x52 {BUS:#04x} {addr:#04x} 0x02 0x97")
        known = {0xb0: "PSU1", 0xb2: "PSU2", 0xb4: "PSU3", 0xb6: "PSU4"}.get(addr, "")
        if w is None:
            print(f"  {addr:#04x} {known:<5} no response   ({err[:60]})")
        else:
            v = linear11(w)
            print(f"  {addr:#04x} {known:<5} READ_PIN = {v:8.1f} W"
                  + ("   <-- NOT IN THE POLL LIST" if not known else ""))
            found.append((addr, v))
    extra = [a for a, _ in found if a not in (0xb0, 0xb2, 0xb4, 0xb6)]
    print()
    if extra:
        print(f"Found {len(extra)} supply(ies) beyond the four named: "
              + " ".join(hex(a) for a in extra))
        print("Add them to PSUS and re-take any capture whose system totals matter.")
    else:
        print("No supplies beyond the four named respond. If system POUT still")
        print("falls short of NVML under load, the shortfall is NOT a missing")
        print("address and the next suspect is the POUT register itself.")
    return 0


def decode(words, psus, cmds, vout_exp=None):
    out, i = {}, 0
    for psu in psus:
        for c in cmds:
            name, kind = PMBUS_READS[c][:2]
            w = words[i]; i += 1
            if kind in ("raw8", "raw16"):
                v = w
            elif kind == "linear16":
                # No exponent means no decode. Emitting the raw mantissa here
                # would put a number in a volts column that is off by a factor
                # of 512, which is worse than an empty cell.
                e = (vout_exp or {}).get(psu)
                v = "" if e is None else round(linear16(w, e), 2)
            else:
                v = round(linear11(w), 2)
            out[f"psu{psu}_{name}"] = v
    return out


def read_vout_exponents(args, psus):
    """Establish the LINEAR16 exponent per supply, and prove it before use.

    VOUT_MODE (0x20) is configuration, not telemetry -- it does not change while
    a supply is up -- so it is read once here and never in the polling round.

    It is not trusted on its word. A bridge that answers 0x00 to a command it
    does not implement decodes as a perfectly valid "Linear, exponent 0", which
    would turn a 12 V rail into 6190 V in the CSV without a single error. So the
    exponent is confirmed against one READ_VOUT: accepted only if it puts the
    rail within 50% of --vout-nominal, and otherwise DERIVED from the mantissa,
    since the ambiguity is only ever a power of two:

        v = mantissa * 2**e   =>   e = round(log2(nominal / mantissa))

    Deriving it that way fixes the scale from the nominal rail voltage and
    leaves the variation -- the sag or the spread, which is the thing actually
    being measured -- untouched. psuN_vout_derived_v (POUT/IOUT) is the
    independent check that it landed right.

    NOT VERIFIED on this platform: every command ASRock supplied used a read
    length of 0x02, so the bridge may refuse the 1-byte read VOUT_MODE needs.
    That path is handled, and says so.
    """
    exps, notes = {}, []
    for p in psus:
        v_w, v_err = _run_one(
            args, f"raw 0x3a 0x52 {BUS:#04x} {PSUS[p]:#04x} 0x02 0x8b", 2)
        if v_w is None or v_w == 0:
            notes.append(f"PSU{p} READ_VOUT unusable "
                         f"({'zero mantissa' if v_w == 0 else v_err[:40]}); "
                         f"vout_v will be blank")
            continue

        cand, why = None, ""
        b, err = _run_one(
            args, f"raw 0x3a 0x52 {BUS:#04x} {PSUS[p]:#04x} 0x01 0x20", 1)
        if b is None:
            why = f"VOUT_MODE unreadable ({err[:32]})"
        else:
            mode, exp = vout_mode(b)
            if mode != 0:
                why = (f"VOUT_MODE={b:#04x} is mode {mode} "
                       f"({'VID' if mode == 1 else 'Direct'}), not Linear")
            elif not (0.5 * args.vout_nominal <= linear16(v_w, exp)
                      <= 1.5 * args.vout_nominal):
                why = (f"VOUT_MODE={b:#04x} gives exponent {exp}, which reads "
                       f"the rail as {linear16(v_w, exp):.4g} V")
            else:
                cand, why = exp, f"VOUT_MODE={b:#04x} -> exponent {exp}"

        if cand is None:
            # Nearest power of two that puts the mantissa at the nominal rail.
            derived = round(math.log2(args.vout_nominal / v_w))
            exps[p] = derived
            notes.append(f"PSU{p} {why}; deriving exponent {derived} from "
                         f"mantissa {v_w} against {args.vout_nominal} V nominal "
                         f"-> {linear16(v_w, derived):.2f} V")
        else:
            exps[p] = cand
            notes.append(f"PSU{p} {why} -> {linear16(v_w, cand):.2f} V")
    return exps, notes


def bmc_iout(args):
    """The BMC's own view of PSU1/PSU2, for the cross-check."""
    try:
        p = subprocess.run(ipmi_prefix(args) + ["sensor", "reading",
                                                "CUR_PSU1_IOUT", "CUR_PSU2_IOUT"],
                           capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return {}
    vals = {}
    for ln in p.stdout.splitlines():
        parts = [x.strip() for x in ln.split("|")]
        if len(parts) >= 2:
            try:
                vals[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return vals


def validate(args):
    # The PMBus read and the BMC sensor read are separate IPMI transactions
    # hundreds of ms apart. Under an oscillating load -- pcieburn swings kilowatts
    # at a few Hz -- they straddle the transition and the delta becomes noise that
    # flips sign between runs. An earlier version reported "decode or address map
    # is wrong" on exactly that, which was a false alarm from an invalid test.
    # So: check whether the load is moving before comparing anything.
    a, _ = _run_one(args, f"raw 0x3a 0x52 {BUS:#04x} {PSUS[1]:#04x} 0x02 0x8c")
    time.sleep(0.4)
    b, _ = _run_one(args, f"raw 0x3a 0x52 {BUS:#04x} {PSUS[1]:#04x} 0x02 0x8c")
    if a is not None and b is not None:
        x, y = linear11(a), linear11(b)
        if max(x, y) > 0 and abs(x - y) / max(x, y) > 0.10:
            print(f"!! LOAD IS MOVING: PSU1 IOUT read {x:.2f} A then {y:.2f} A "
                  f"0.4 s apart ({100*abs(x-y)/max(x,y):.0f}% apart).\n"
                  "!! This comparison is INVALID while the load oscillates -- the two\n"
                  "!! readings are separate IPMI transactions and will straddle the\n"
                  "!! swing, producing large deltas of either sign that say nothing\n"
                  "!! about the decode. Stop the load and re-run at idle.\n")
            return 3
    print("Reading IOUT (0x8c) over PMBus for all four supplies...")
    words, err = run_batch(args, *build_batch([1, 2, 3, 4], [0x8c]))
    if words is None:
        print(f"  error: {err}")
        print("FAILED: batch did not return one hex line per command.")
        print("Try one command by hand to see the actual error:")
        print(f"  {' '.join(ipmi_prefix(args))} raw 0x3a 0x52 0x0c 0xb0 0x02 0x8c")
        return 2
    pm = decode(words, [1, 2, 3, 4], [0x8c])
    bmc = bmc_iout(args)

    print(f"\n  {'':<5} {'PMBus IOUT':>12} {'BMC sensor':>13} {'delta':>9}")
    ok = True
    for psu in (1, 2):
        a = pm[f"psu{psu}_iout_a"]
        b = bmc.get(f"CUR_PSU{psu}_IOUT")
        if b is None:
            print(f"  PSU{psu}  {a:>10.2f} A {'n/a':>13}")
            continue
        d = a - b
        bad = abs(d) > max(0.5, 0.05 * max(a, b))
        ok = ok and not bad
        print(f"  PSU{psu}  {a:>10.2f} A {b:>11.2f} A {d:>8.2f}"
              + ("   <-- MISMATCH" if bad else ""))
    for psu in (3, 4):
        print(f"  PSU{psu}  {pm[f'psu{psu}_iout_a']:>10.2f} A {'(BMC blind)':>13}")

    print()
    if ok:
        print("PASS: PMBus agrees with the BMC on both visible supplies, so the")
        print("      bridge, address map, endianness and LINEAR11 decode are all")
        print("      correct. PSU3/PSU4 readings can be trusted.")
    else:
        print("FAIL: PMBus and the BMC disagree on a supply both can see. If the")
        print("      load was genuinely idle, the decode or address map is wrong and")
        print("      PSU3/PSU4 have no second opinion to catch it. If anything was")
        print("      running, re-run at idle first -- non-simultaneous reads across a")
        print("      moving load produce exactly this, with no fault in the decode.")

    print("\nProbing READ_POUT (0x96) and READ_PIN (0x97). 0x97 is confirmed by")
    print("ASRock; 0x96 is the PMBus standard code but unconfirmed here:")
    w, err = run_batch(args, *build_batch([1, 2, 3, 4], [0x96, 0x97]))
    if w is None:
        print(f"  error: {err}")
        print("  batch failed -- 0x96 may be unsupported. Fall back to")
        print("  --cmds 0x97 0x8c and derive output power as IOUT * VOUT.")
        return 1
    d = decode(w, [1, 2, 3, 4], [0x96, 0x97])
    print(f"\n  {'':<5} {'POUT':>10} {'PIN':>10} {'eff':>9}")
    for psu in (1, 2, 3, 4):
        po, pi = d[f"psu{psu}_pout_w"], d[f"psu{psu}_pin_w"]
        print(f"  PSU{psu}  {po:>8.1f} W {pi:>8.1f} W "
              + (f"{100*po/pi:>7.1f}%" if pi > 0 else f"{'n/a':>8}"))
    print("\nSanity: efficiency should be ~88-95% under load. Anything outside")
    print("that suggests 0x96 is not READ_POUT on this platform.")
    return 0 if ok else 1


def poll(args):
    cmds, psus = args.cmds, sorted(PSUS)
    lines, nb = build_batch(psus, cmds)

    # Cadence. Each bridged read costs ~30 ms and they are serial, so the floor
    # on a round is len(psus) * len(cmds) * SECONDS_PER_READ. Asking for 0.25 s
    # with three registers is asking for something the bridge cannot do; say so
    # once at the top rather than letting the reader discover it later from the
    # interval_s column. Widening the round also widens the skew BETWEEN
    # registers within it, which is why the derived VOUT below is a sanity
    # check and not a measurement.
    floor = len(psus) * len(cmds) * SECONDS_PER_READ
    if args.interval < floor:
        sys.stderr.write(
            f"psu_pmbus_poll: {len(psus)}x{len(cmds)} reads/round needs about "
            f"{floor:.2f}s; --interval {args.interval:g} will not be met. "
            f"The achieved interval is recorded per row.\n")

    cols = ["timestamp", "interval_s", "read_ok", "err"]
    cols += [f"psu{p}_{PMBUS_READS[c][0]}" for p in psus for c in cmds]
    # One system total per register that has one, in the order the registers
    # were requested -- so adding a register to the allowlist adds its column
    # without another branch here.
    sums = [(c, PMBUS_READS[c][2], PMBUS_READS[c][3])
            for c in cmds if PMBUS_READS[c][2]]
    cols += [name for _, name, _ in sums]
    # The 12 V rail. Four supplies feeding one bus means four measurements of
    # the same voltage, so the informative aggregate is the SPREAD, not a sum:
    # a common sag moves all four together, while one supply losing its share
    # shows up here and nowhere else.
    vout_exp = {}
    if 0x8b in cmds:
        vout_exp, notes = read_vout_exponents(args, psus)
        for note in notes:
            sys.stderr.write(f"psu_pmbus_poll: {note}\n")
        missing = [p for p in psus if p not in vout_exp]
        if missing:
            sys.stderr.write(
                f"psu_pmbus_poll: no usable VOUT for "
                f"PSU{','.join(map(str, missing))}; those columns stay blank "
                f"and vout_spread_v covers the rest.\n")
        if len(vout_exp) < len(psus):
            sys.stderr.write(
                "psu_pmbus_poll: cross-check psuN_vout_v against "
                "psuN_vout_derived_v -- they should agree closely, and a clean "
                "power-of-two ratio means the exponent is still wrong.\n")
        cols.append("vout_spread_v")

    # POUT/IOUT is the same quantity as READ_VOUT from two registers already in
    # the round -- no extra command, and no dependence on VOUT_MODE. It is kept
    # even when 0x8b IS polled, because the two together are the only check on
    # the LINEAR16 exponent: agreement confirms it, and disagreement by a clean
    # power of two says the exponent is wrong rather than the rail sagging.
    derive_vout = 0x8c in cmds and 0x96 in cmds
    if derive_vout:
        cols += [f"psu{p}_vout_derived_v" for p in psus]

    out = open(args.out, "w", buffering=1) if args.out else sys.stdout
    out.write(",".join(cols) + "\n")

    t_end = time.time() + args.duration if args.duration else None
    prev, last_err = None, None
    try:
        while t_end is None or time.time() < t_end:
            t0 = time.time()
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            words, err = run_batch(args, lines, nb)
            # Record the ACHIEVED interval, not the requested one. The previous
            # PSU poller was nominally 0.25 s and actually landed near 1 Hz,
            # which was only discovered long afterwards. A trace that carries its
            # own cadence cannot mislead the next reader that way.
            interval = "" if prev is None else f"{t0 - prev:.3f}"
            prev = t0

            if words is None:
                if err and err != last_err:
                    sys.stderr.write(f"psu_pmbus_poll: {err}\n")
                    last_err = err
                row = [ts, interval, "0", err.replace(",", ";")]
                row += [""] * (len(cols) - 4)
            else:
                d = decode(words, psus, cmds, vout_exp)
                row = [ts, interval, "1", ""]
                row += [str(d[f"psu{p}_{PMBUS_READS[c][0]}"]) for p in psus for c in cmds]
                for c, _name, dp in sums:
                    field = PMBUS_READS[c][0]
                    total = sum(d[f"psu{p}_{field}"] for p in psus)
                    row.append(f"{total:.{dp}f}")
                if 0x8b in cmds:
                    vs = [d[f"psu{p}_vout_v"] for p in psus]
                    vs = [v for v in vs if v != ""]
                    row.append(f"{max(vs) - min(vs):.2f}" if len(vs) > 1 else "")
                if derive_vout:
                    for p in psus:
                        i_a = d[f"psu{p}_iout_a"]
                        # Below ~0.5 A the quotient is dominated by IOUT
                        # quantization, so it is left blank rather than written
                        # as a wild volt figure a reader might average in.
                        row.append(f"{d[f'psu{p}_pout_w'] / i_a:.2f}"
                                   if i_a >= 0.5 else "")
            out.write(",".join(row) + "\n")

            slp = args.interval - (time.time() - t0)
            if slp > 0:
                time.sleep(slp)
    except KeyboardInterrupt:
        pass
    finally:
        if args.out:
            out.close()


def hexint(s):
    v = int(s, 0)
    if v not in PMBUS_READS:
        raise argparse.ArgumentTypeError(
            f"{s} is not in the read-only allowlist "
            f"({', '.join(hex(k) for k in sorted(PMBUS_READS))}). Writes through "
            "this bridge can reconfigure or shut down a live supply.")
    return v


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", action="store_true",
                    help="probe the PMBus address range for supplies beyond the "
                         "four ASRock named, and exit. Read-only.")
    ap.add_argument("--validate", action="store_true",
                    help="cross-check against the BMC's PSU1/PSU2 sensors and "
                         "exit. Run this before trusting any output.")
    ap.add_argument("--out", help="CSV path (default stdout)")
    ap.add_argument("--interval", type=float, default=0.55,
                    help="target seconds per round (default 0.55: the 16-read "
                         "default round floors at ~0.48s, and the slack keeps "
                         "jitter from eating the sleep. The achieved interval "
                         "is recorded per row, and a request below the floor "
                         "warns once on stderr.)")
    ap.add_argument("--duration", type=float, default=0,
                    help="seconds to poll, 0 = until killed")
    ap.add_argument("--cmds", type=hexint, nargs="+",
                    default=[0x8b, 0x8c, 0x96, 0x97],
                    help="PMBus read codes (default 0x8b READ_VOUT, 0x8c "
                         "READ_IOUT, 0x96 READ_POUT, 0x97 READ_PIN). Order is "
                         "the order they are issued in, and IOUT/POUT are kept "
                         "adjacent so the VOUT derived from them is skewed by "
                         "one read, not three. Narrower rounds sample faster: "
                         "'0x8b 0x8c' is an 8-read rail-and-current round at "
                         "4 Hz, '0x96 0x97' the original power-only one.")
    ap.add_argument("--vout-nominal", type=float, default=12.0,
                    help="nominal output rail, in volts (default 12.0). Used "
                         "only to sanity-check the LINEAR16 exponent from "
                         "VOUT_MODE, and to derive one when that register is "
                         "unreadable, absent or implausible. It fixes the "
                         "power-of-two scale and does not affect the measured "
                         "variation.")
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--bmc", help="BMC IP for lanplus; omit for local KCS (faster)")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", default=os.environ.get("IPMI_PASSWORD", ""))
    ap.add_argument("--no-sudo", dest="sudo", action="store_false",
                    help="do not prefix local ipmitool with sudo -n")
    ap.add_argument("--no-batch", action="store_true",
                    help="one ipmitool per read instead of a batched exec. "
                         "Much slower, but isolates which read fails.")
    ap.add_argument("--force", action="store_true",
                    help="poll even if the preflight read fails")
    args = ap.parse_args()
    if args.scan:
        sys.exit(scan(args))
    if args.validate:
        sys.exit(validate(args))
    if not preflight(args) and not args.force:
        sys.exit(2)
    poll(args)


if __name__ == "__main__":
    main()

