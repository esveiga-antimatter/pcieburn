#!/usr/bin/env python3
"""Per-PSU input/output power over the ASRock PMBus bridge.

WHY THIS EXISTS
---------------
The BMC exposes sensors for only two of the four supplies (CUR_PSU1_IOUT,
CUR_PSU2_IOUT). Everything downstream has therefore had to *estimate* system
power by assuming the instrumented pair is representative and that load is shared
evenly -- run_pcieburn.sh labels the column `est_system_w` for exactly that
reason. Reading all four supplies directly over PMBus replaces that estimate with
a measurement, and incidentally tests the even-sharing assumption itself.

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
bits 10..0, value = mantissa * 2**exponent. READ_VOUT is *not* LINEAR11 -- it is
LINEAR16 whose exponent comes from VOUT_MODE (0x20) -- which is why this script
does not read it. Do not assume one decoder covers every register.

VERIFIED vs ASSUMED
-------------------
ASRock supplied 0x97 (READ_PIN). 0x96 (READ_POUT) and 0x8c (READ_IOUT) are the
PMBus standard codes for the other two quantities but are NOT confirmed on this
platform. Run `--validate` first: it reads IOUT over PMBus for the two supplies
the BMC *can* see and compares against the BMC's own sensors. If those agree, the
bridge, address map, endianness and decode are all correct at once -- after which
the identical decode can be trusted for PSU3/PSU4, which have no second opinion.

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

Still worth running --validate once per platform: the efficiency check confirms
the command codes and the decode, but only the BMC cross-check confirms the
ADDRESS MAP, i.e. that 0xb0 is the supply the BMC calls PSU1.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

# --- read-only allowlist. Nothing outside this set is ever sent. --------------
PMBUS_READS = {
    0x79: ("status_word", "raw16"),
    0x8c: ("iout_a",      "linear11"),
    0x96: ("pout_w",      "linear11"),
    0x97: ("pin_w",       "linear11"),
}
PSUS = {1: 0xb0, 2: 0xb2, 3: 0xb4, 4: 0xb6}
BUS = 0x0c

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


def ipmi_prefix(args):
    if args.bmc:
        # lanplus opens a network session per invocation -- far slower than local
        # KCS. Batching matters even more in this mode.
        return ["ipmitool", "-I", "lanplus", "-H", args.bmc,
                "-U", args.user, "-P", args.password]
    return (["sudo", "-n"] if args.sudo else []) + ["ipmitool"]


def build_batch(psus, cmds):
    """One ipmitool `exec` script for the whole round. Order is significant."""
    return [f"raw 0x3a 0x52 {BUS:#04x} {PSUS[p]:#04x} 0x02 {c:#04x}"
            for p in psus for c in cmds]


def run_batch(args, lines):
    """Run the batch; return (words, err). words is None when the round is
    unreliable, and err then carries WHY -- ipmitool reports every failure on
    stderr and prints nothing to stdout, so discarding stderr (as an earlier
    version did) turns a one-line diagnosis into thousands of blank rows.

    Alignment matters. A failed read emits no stdout line, which would silently
    shift every later value onto the wrong PSU. So a round is accepted only when
    the hex-line count equals the command count -- attributing PSU4's power to
    PSU3 would be far worse than a gap in the trace.
    """
    if args.no_batch:
        words = []
        for one in lines:
            w, err = _run_one(args, one)
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
    finally:
        os.unlink(path)

    words = []
    for ln in p.stdout.splitlines():
        if not HEXLINE.match(ln):
            continue
        b = [int(x, 16) for x in ln.split()]
        if len(b) < 2:
            return None, f"short response: {ln.strip()!r}"
        words.append(b[0] | (b[1] << 8))     # PMBus is little-endian
    if len(words) == len(lines):
        return words, ""
    err = " | ".join(l.strip() for l in p.stderr.splitlines()[:2]) or \
          f"got {len(words)} of {len(lines)} responses, no stderr"
    return None, err


def _run_one(args, cmdline):
    """One raw command, one ipmitool. Slower, but isolates which read fails."""
    try:
        p = subprocess.run(ipmi_prefix(args) + cmdline.split(),
                           capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    for ln in p.stdout.splitlines():
        if HEXLINE.match(ln):
            b = [int(x, 16) for x in ln.split()]
            if len(b) >= 2:
                return b[0] | (b[1] << 8), ""
    return None, " | ".join(l.strip() for l in p.stderr.splitlines()[:2]) or "no output"


def preflight(args):
    """One read before the loop. An unreachable BMC, missing sudo, or an
    unsupported OEM command should stop the run with the reason on screen, not
    produce a CSV full of read_ok=0."""
    probe = f"raw 0x3a 0x52 {BUS:#04x} {PSUS[1]:#04x} 0x02 {args.cmds[0]:#04x}"
    w, err = _run_one(args, probe)
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


def decode(words, psus, cmds):
    out, i = {}, 0
    for psu in psus:
        for c in cmds:
            name, kind = PMBUS_READS[c]
            w = words[i]; i += 1
            out[f"psu{psu}_{name}"] = w if kind == "raw16" else round(linear11(w), 2)
    return out


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
    print("Reading IOUT (0x8c) over PMBus for all four supplies...")
    words, err = run_batch(args, build_batch([1, 2, 3, 4], [0x8c]))
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
        print("FAIL: do not trust PSU3/PSU4. A mismatch on a supply the BMC can")
        print("      see means the decode or address map is wrong, and the")
        print("      invisible supplies have no second opinion to catch it.")

    print("\nProbing READ_POUT (0x96) and READ_PIN (0x97). 0x97 is confirmed by")
    print("ASRock; 0x96 is the PMBus standard code but unconfirmed here:")
    w, err = run_batch(args, build_batch([1, 2, 3, 4], [0x96, 0x97]))
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
    lines = build_batch(psus, cmds)
    cols = ["timestamp", "interval_s", "read_ok", "err"]
    cols += [f"psu{p}_{PMBUS_READS[c][0]}" for p in psus for c in cmds]
    if 0x97 in cmds:
        cols.append("system_pin_w")
    if 0x96 in cmds:
        cols.append("system_pout_w")

    out = open(args.out, "w", buffering=1) if args.out else sys.stdout
    out.write(",".join(cols) + "\n")

    t_end = time.time() + args.duration if args.duration else None
    prev, last_err = None, None
    try:
        while t_end is None or time.time() < t_end:
            t0 = time.time()
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            words, err = run_batch(args, lines)
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
                d = decode(words, psus, cmds)
                row = [ts, interval, "1", ""]
                row += [str(d[f"psu{p}_{PMBUS_READS[c][0]}"]) for p in psus for c in cmds]
                if 0x97 in cmds:
                    row.append(f"{sum(d[f'psu{p}_pin_w'] for p in psus):.1f}")
                if 0x96 in cmds:
                    row.append(f"{sum(d[f'psu{p}_pout_w'] for p in psus):.1f}")
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
    ap.add_argument("--validate", action="store_true",
                    help="cross-check against the BMC's PSU1/PSU2 sensors and "
                         "exit. Run this before trusting any output.")
    ap.add_argument("--out", help="CSV path (default stdout)")
    ap.add_argument("--interval", type=float, default=0.25,
                    help="target seconds per round (default 0.25; the achieved "
                         "interval is recorded per row)")
    ap.add_argument("--duration", type=float, default=0,
                    help="seconds to poll, 0 = until killed")
    ap.add_argument("--cmds", type=hexint, nargs="+", default=[0x96, 0x97],
                    help="PMBus read codes (default 0x96 READ_POUT, 0x97 READ_PIN)")
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
    if args.validate:
        sys.exit(validate(args))
    if not preflight(args) and not args.force:
        sys.exit(2)
    poll(args)


if __name__ == "__main__":
    main()
