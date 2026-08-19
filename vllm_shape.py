#!/usr/bin/env python3
"""Map the production vLLM serving profile onto pcieburn's knobs.

Megatron TP: each layer all-reduces the activation tensor twice (post-attention,
post-MLP). Message = tokens_in_step x hidden x dtype_bytes, independent of TP
degree. Per-step collective count = 2 x layers.

The GEMM "square equivalent" converts a real projection GEMM's FLOP count into
the N of an NxN pcieburn GEMM, since 2*N^3 == 2*M*K*Nout.
"""
MODELS = [
    # name, hidden, layers, ar_per_layer, tp
    ("gemma-4-26b",  2816, 30, 2, 2),
    ("gemma-4-31b",  5376, 60, 2, 2),
    ("gpt-oss-20b",  2880, 24, 0, 1),
    ("qwen3.6-27b",  5120, 64, 2, 2),
    ("qwen3.6-35b",  2048, 40, 2, 2),
]
DT = 2                 # bf16/fp16
DECODE_TOK = 64        # max-num-seqs, one token per live sequence
PREFILL_TOK = 16384    # max-num-batched-tokens

def human(b):
    for u in ("B", "KiB", "MiB", "GiB"):
        if b < 1024 or u == "GiB":
            return f"{b:.0f} {u}" if u == "B" else f"{b:.1f} {u}"
        b /= 1024

def sq_equiv(tokens, hidden, tp):
    # projection GEMM: [tokens, hidden] x [hidden, hidden/tp]
    flops = 2 * tokens * hidden * (hidden / tp)
    return (flops / 2) ** (1 / 3)

print(f"{'model':<14} {'AR/step':>7} | {'decode msg':>10} {'sq-equiv':>8} | "
      f"{'prefill msg':>11} {'sq-equiv':>8}")
print("-" * 74)
for name, h, L, arl, tp in MODELS:
    ar = arl * L
    if ar == 0:
        print(f"{name:<14} {0:>7} | {'— no collectives at all (TP1)':>50}")
        continue
    d_msg = DECODE_TOK * h * DT
    p_msg = PREFILL_TOK * h * DT
    print(f"{name:<14} {ar:>7} | {human(d_msg):>10} {sq_equiv(DECODE_TOK,h,tp):>8.0f} | "
          f"{human(p_msg):>11} {sq_equiv(PREFILL_TOK,h,tp):>8.0f}")

print()
print("GEMMs between consecutive all-reduces (Megatron layer):")
print("  attention block: QKV projection + output projection  -> all-reduce")
print("  MLP block:       up/gate + down                      -> all-reduce")
print("  => 2-3 GEMMs per collective\n")

print("pcieburn, as actually measured in the 8192-single arm:")
print("  32,430 GEMMs / 345 collectives = 94.0 GEMMs per collective")
print("  collective size 128 MiB-1 GiB, rate 0.57/s\n")

print("Decode-regime PCIe traffic per rank, for a range of step rates:")
for name, h, L, arl, tp in MODELS:
    if arl == 0:
        continue
    per_step = arl * L * DECODE_TOK * h * DT
    for rate in (20, 50):
        gbs = per_step * rate / 1e9
        print(f"  {name:<14} {rate:>3} steps/s -> {human(per_step):>9}/step  = {gbs:5.2f} GB/s egress/rank")
