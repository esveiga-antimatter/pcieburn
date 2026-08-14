# plot_run.gp — interactive explorer for one pcieburn run.
#
# Opens a plot window with keyboard-switchable views over a shared time axis, so
# power, PSU current, PCIe throughput, AER counters and link state can be zoomed
# and compared against each other. A dashed vertical marker is drawn at the fault
# instant when the run faulted.
#
# Usage:
#   gnuplot -e "RUN='runs/20260812T114155Z-limit500'" -p pcieburn/plot_run.gp
#   gnuplot -e "RUN='...'; FOCUS=5" -p pcieburn/plot_run.gp   # start on GPU5
#   gnuplot -e "RUN='...'; GPTERM='wxt'" -p pcieburn/plot_run.gp
#   gnuplot -e "RUN='...'; PNG=1" pcieburn/plot_run.gp        # write PNGs instead
#
# The -p (--persist) flag keeps the window open after the script ends. Without
# it the window closes immediately.
#
# KEYS (in the plot window)
#   1..6   switch view          [ ]   previous / next focus GPU
#   a      autoscale (unzoom)   u     restore previous zoom
#   g      toggle grid          l     toggle log y        ?    list all hotkeys
#   right-drag or middle-drag zooms; q closes the window.
#
# Deliberately NOT multiplot: gnuplot disables mouse zoom in multiplot mode, so a
# stack of panels cannot be explored interactively. One view at a time keeps zoom,
# pan and the coordinate readout working. Use PNG=1 for a stackable set of files.
#
# Two timestamp formats coexist in a run directory, so each dataset parses its own
# via strptime() inside 'using' rather than one global 'set timefmt':
#   nvml_trace.csv          2026/08/12 11:41:58.009
#   everything else         2026-08-12T11:41:58.138Z
# strptime consumes the format and ignores the trailing fraction, so plots are
# accurate to the second — fine here, since the fastest poll is 100 ms.

if (!exists("RUN"))   { print "ERROR: pass RUN='path/to/run/dir'"; exit }
if (!exists("FOCUS")) { FOCUS = 0 }
if (!exists("PNG"))   { PNG = 0 }

TF_NVML = "%Y/%m/%d %H:%M:%S"
TF_ISO  = "%Y-%m-%dT%H:%M:%S"
TF_DMON = "%Y%m%d %H:%M:%S"

F_NVML   = RUN . "/nvml_trace.csv"
F_PSU    = RUN . "/psu_current.csv"
F_AER    = RUN . "/aer_counters.csv"
F_DMON   = RUN . "/pcie_dmon.txt"
F_EVENTS = RUN . "/events.csv"

have(f) = system(sprintf("test -s '%s' && echo 1 || echo 0", f)) + 0

if (!have(F_NVML)) { print "ERROR: no nvml_trace.csv in " . RUN . \
                           " (rerun with --with-nvml)"; exit }

# Common time window, from the NVML trace since it starts first and ends last.
T0 = system(sprintf("awk -F, 'NR==2{print $1; exit}' '%s'", F_NVML))
T1 = system(sprintf("awk -F, 'END{print $1}' '%s'", F_NVML))
X0 = strptime(TF_NVML, T0)
X1 = strptime(TF_NVML, T1)

# Fault instant, if any. rank_lost / error are what the supervisor logs when a
# rank dies or reports a collective timeout.
FAULT_TS = ""
if (have(F_EVENTS)) {
    FAULT_TS = system(sprintf( \
        "awk -F, '$3==\"error\"||$3==\"rank_lost\"{print $1; exit}' '%s'", \
        F_EVENTS))
}
HAVE_FAULT = strlen(FAULT_TS) > 8 ? 1 : 0
if (HAVE_FAULT) { XF = strptime(TF_ISO, FAULT_TS) }

TAG     = system(sprintf("awk -F, 'NR==2{print $2; exit}' '%s' 2>/dev/null", \
                         F_EVENTS))
VERDICT = system(sprintf("grep -m1 '^verdict' '%s/manifest.txt' 2>/dev/null" \
                         . " | sed 's/.*: *//'", RUN))

# ---------------------------------------------------------------- terminal
# An interactive terminal needs a display. Over ssh that means 'ssh -X', or copy
# the run directory to a machine with one — which is the usual workflow here,
# since runs get copied off the test node anyway.
HAVE_DISP = strlen(system("printf '%s%s' \"$DISPLAY\" \"$WAYLAND_DISPLAY\"")) > 0

if (PNG) {
    INTERACTIVE = 0
} else {
    if (!HAVE_DISP && !exists("GPTERM")) {
        print "WARNING: no DISPLAY/WAYLAND_DISPLAY — falling back to an ASCII"
        print "         terminal. For a real window use 'ssh -X', run this on a"
        print "         machine with a display, or pass PNG=1 for image files."
        GPTERM = "dumb"
    }
    if (!exists("GPTERM")) {
        # Prefer qt, then wxt, then x11, depending on what this gnuplot has.
        GPTERM = strstrt(GPVAL_TERMINALS, "wxt")  > 0 ? "wxt"  : \
                 strstrt(GPVAL_TERMINALS, "qt") > 0 ? "qt" : "x11"
    }
    INTERACTIVE = GPTERM eq "dumb" ? 0 : 1
}

if (PNG) {
    set terminal pngcairo size 1500,500 enhanced font "Sans,10" \
        background rgb "white"
} else {
    if (GPTERM eq "dumb") {
        set terminal dumb size 185,45
    } else {
        eval "set terminal " . GPTERM . " size 1400,760 enhanced font 'Sans,10'"
    }
}

# ---------------------------------------------------------------- common style
set xdata time
set format x "%H:%M:%S"
set xrange [X0:X1]
set xlabel "time (UTC)"
set grid xtics ytics lc rgb "#dddddd"
set key top left opaque box lc rgb "#999999" samplen 1.5
set tics nomirror out

C_FOCUS = "#d62728"
C_OTHER = "#b8b8b8"
C_A     = "#1f77b4"
C_B     = "#2ca02c"
C_C     = "#ff7f0e"

if (HAVE_FAULT) {
    set arrow 99 from XF, graph 0 to XF, graph 1 nohead \
        lc rgb "#d62728" lw 2 dt 2 front
}

# Per-trace row selection is done with awk, NOT with a '($2==g ? $3 : 1/0)'
# filter in 'using'. An undefined point TERMINATES a line in gnuplot, and these
# files interleave one row per GPU per poll, so such a filter leaves every trace
# as isolated single points with gaps between them — which 'with lines' renders
# as nothing at all, while still autoscaling the axes off the data. Piping
# through awk keeps each trace's rows contiguous.
nvml_gpu(g)      = sprintf("< awk -F, '$2==%d' '%s'", g, F_NVML)
aer_sel(g, role) = sprintf("< awk -F, '$2==%d && $4==\"%s\"' '%s'", \
                           g, role, F_AER)
dmon_gpu(g)      = sprintf("< awk '$1 !~ /^#/ && $3==%d && $4!=\"-\"' '%s'", \
                           g, F_DMON)
ev_rank(r)       = sprintf("< awk -F, '$3==\"progress\" && $4==%d' '%s'", \
                           r, F_EVENTS)

SUB = sprintf("   [%s  %s]", TAG, VERDICT)

# ---------------------------------------------------------------- views
# Built once as command strings. FOCUS is read when the string is eval'd, not
# when it is built, so the '[' / ']' keys need no rebuild — they just re-eval.
MISSING = "set yrange [0:1]; plot 0 with lines lc rgb '#ffffff' notitle; " \
        . "set autoscale y"

RESET = "unset label 99; unset y2tics; unset y2label; unset y2range; " \
      . "set ytics nomirror; set datafile separator ','; set autoscale y; "

V1 = RESET \
   . "set ylabel 'GPU power (W)'; " \
   . "set title sprintf('1/6  GPU power — focus GPU%d', FOCUS) . SUB; " \
   . "plot for [g=0:7] nvml_gpu(g) using (strptime(TF_NVML,strcol(1))):3 " \
   . "  with lines lw 1 lc rgb C_OTHER notitle, " \
   . " nvml_gpu(FOCUS) using (strptime(TF_NVML,strcol(1))):3 " \
   . "  with lines lw 2 lc rgb C_FOCUS title sprintf('GPU%d', FOCUS)"

V2 = RESET \
   . "set ylabel 'SM clock (MHz)'; " \
   . "set title sprintf('2/6  SM clock — focus GPU%d', FOCUS) . SUB; " \
   . "plot for [g=0:7] nvml_gpu(g) using (strptime(TF_NVML,strcol(1))):4 " \
   . "  with lines lw 1 lc rgb C_OTHER notitle, " \
   . " nvml_gpu(FOCUS) using (strptime(TF_NVML,strcol(1))):4 " \
   . "  with lines lw 2 lc rgb C_FOCUS title sprintf('GPU%d', FOCUS)"

# Output current is the only power sensor on this platform with adequate range:
# PWR_*_PIN wraps at 255 W and PWR_*_POUT saturates near 510 W.
V3 = RESET \
   . "set ylabel 'PSU output current (A)'; " \
   . "set y2label 'est. system power (W)'; set y2tics; " \
   . "set title '3/6  PSU output current (the only in-range power sensor)' . SUB; " \
   . (have(F_PSU) ? \
       "plot F_PSU using (strptime(TF_ISO,strcol(1))):2 " \
     . "  with lines lw 2 lc rgb C_A title 'PSU1 IOUT', " \
     . " F_PSU using (strptime(TF_ISO,strcol(1))):3 " \
     . "  with lines lw 2 lc rgb C_B title 'PSU2 IOUT', " \
     . " F_PSU using (strptime(TF_ISO,strcol(1))):8 axes x1y2 " \
     . "  with lines lw 1 dt 3 lc rgb '#666666' title 'est. system W (y2)'" \
     : "set label 99 'no psu_current.csv — rerun with --with-psu' " \
     . "at graph 0.5, graph 0.5 center tc rgb '#999999'; " . MISSING)

V4 = RESET . "set datafile separator whitespace; " \
   . "set ylabel 'PCIe throughput (MB/s)'; " \
   . "set title sprintf('4/6  PCIe rx/tx — GPU%d', FOCUS) . SUB; " \
   . (have(F_DMON) ? \
       "plot dmon_gpu(FOCUS) using " \
     . "  (strptime(TF_DMON,strcol(1).' '.strcol(2))):4 " \
     . "  with lines lw 1.5 lc rgb C_A title sprintf('GPU%d rx', FOCUS), " \
     . " dmon_gpu(FOCUS) using " \
     . "  (strptime(TF_DMON,strcol(1).' '.strcol(2))):5 " \
     . "  with lines lw 1.5 lc rgb C_C title sprintf('GPU%d tx', FOCUS)" \
     : "set label 99 'no pcie_dmon.txt — rerun with --with-dmon' " \
     . "at graph 0.5, graph 0.5 center tc rgb '#999999'; " . MISSING)

# Counters are cumulative and, on every failure observed so far, sit at exactly
# zero for the whole run and then burst in the last few seconds. The floor is
# pinned at 0 with a minimum ceiling of 1 so a clean all-zero run still draws.
V5 = RESET \
   . "set ylabel 'AER count (cumulative)'; set yrange [0:1<*]; " \
   . "set y2label 'link gen (dashed)'; set y2range [0:6]; set y2tics 1; " \
   . "set title sprintf('5/6  AER counters + link gen — GPU%d', FOCUS) . SUB; " \
   . (have(F_AER) ? \
       "plot aer_sel(FOCUS,'dev') using (strptime(TF_ISO,strcol(1))):8 " \
     . "  with steps lw 2 lc rgb C_FOCUS title 'dev Rollover', " \
     . " aer_sel(FOCUS,'rootport') using (strptime(TF_ISO,strcol(1))):6 " \
     . "  with steps lw 2 lc rgb C_A title 'rootport BadTLP', " \
     . " aer_sel(FOCUS,'dev') using (strptime(TF_ISO,strcol(1))):5 " \
     . "  with steps lw 1.5 lc rgb C_B title 'dev RxErr', " \
     . " nvml_gpu(FOCUS) using (strptime(TF_NVML,strcol(1))):7 axes x1y2 " \
     . "  with lines lw 1.5 dt 2 lc rgb '#555555' title 'link gen (y2)'" \
     : "plot nvml_gpu(FOCUS) using (strptime(TF_NVML,strcol(1))):7 axes x1y2 " \
     . "  with lines lw 1.5 dt 2 lc rgb '#555555' title 'link gen (y2)'")

# GEMM-window throughput per rank. The sawtooth is the half/single precision
# alternation; a step change partway through is tensor-op math engaging.
V6 = RESET \
   . "set ylabel 'GEMM throughput (GFLOP/s)'; " \
   . "set title sprintf('6/6  per-rank GFLOP/s — focus rank %d', FOCUS) . SUB; " \
   . (have(F_EVENTS) ? \
       "plot for [r=0:7] ev_rank(r) using (strptime(TF_ISO,strcol(1))):13 " \
     . "  with linespoints pt 7 ps 0.3 lw 1 lc rgb C_OTHER notitle, " \
     . " ev_rank(FOCUS) using (strptime(TF_ISO,strcol(1))):13 " \
     . "  with linespoints pt 7 ps 0.5 lw 2 lc rgb C_FOCUS " \
     . "  title sprintf('rank %d', FOCUS)" \
     : "set label 99 'no events.csv' at graph 0.5, graph 0.5 center " \
     . "tc rgb '#999999'; " . MISSING)

# ---------------------------------------------------------------- drive it
if (PNG) {
    set output RUN . "/plot_1_gpu_power.png";        eval V1
    set output RUN . "/plot_2_sm_clock.png";         eval V2
    set output RUN . "/plot_3_psu_current.png";      eval V3
    set output RUN . "/plot_4_pcie_throughput.png";  eval V4
    set output RUN . "/plot_5_aer_and_link.png";     eval V5
    set output RUN . "/plot_6_gflops.png";           eval V6
    unset output
    print sprintf("wrote %s/plot_1_*.png .. plot_6_gflops.png", RUN)
} else {
    CUR = V1
    eval CUR

    if (INTERACTIVE) {
        bind "1" 'CUR = V1; eval CUR'
        bind "2" 'CUR = V2; eval CUR'
        bind "3" 'CUR = V3; eval CUR'
        bind "4" 'CUR = V4; eval CUR'
        bind "5" 'CUR = V5; eval CUR'
        bind "6" 'CUR = V6; eval CUR'
        bind "]" 'FOCUS = (FOCUS + 1) % 8; eval CUR'
        bind "[" 'FOCUS = (FOCUS + 7) % 8; eval CUR'

        print ""
        print sprintf("pcieburn interactive  —  %s", RUN)
        print sprintf("  tag=%s   verdict=%s", TAG, VERDICT)
        if (HAVE_FAULT) { print sprintf("  fault marker at %s", FAULT_TS) }
        print ""
        print "  1 GPU power    2 SM clock     3 PSU current"
        print "  4 PCIe rx/tx   5 AER + link   6 GFLOP/s"
        print "  [ ] change focus GPU/rank     a autoscale   u unzoom"
        print "  g grid   l log-y   ? all hotkeys   right-drag to zoom"
        print ""
    }
}
if (HAVE_FAULT) {
    if (PNG) { print sprintf("fault marker at %s", FAULT_TS) }
} else {
    print "no fault row in events.csv — no fault marker drawn"
}
