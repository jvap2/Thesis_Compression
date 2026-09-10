# Joint shift-add MAC -- RTL + verification

A synthesizable Verilog implementation of the multiplier-free MAC described
in the `sec:codebook-design` addition (Eq. `eq:shift-add-mac`), plus an
exhaustive testbench that checks it bit-exact against an independent Python
golden model.

## Layout

```
rtl/
  w_decode_e2m1.v    -- decodes a 4-bit E2M1 weight code to <=2 signed powers of two
  a_decode_hwcb.v    -- decodes a 4-bit hw-codebook activation code to <=2 signed powers of two
  shift_add_mult.v   -- the multiplier replacement: combine + one-hot terms + compressor
  shift_add_mac.v     -- registered accumulator wrapper (for dot-product use)
tb/
  tb_shift_add_mult.v -- exhaustive check: all 256 (w_code,a_code) pairs vs golden
  tb_shift_add_mac.v  -- K=64 accumulated dot product vs golden
scripts/
  golden_model.py     -- independent Python reference (plain float decode+multiply,
                          scaled to Q4 fixed point), emits golden_products.hex/.csv
  dotprod_golden.py   -- random K=64 sequence + expected accumulated sum
  run_sim.sh           -- regenerates vectors and runs both testbenches
```

## Running it

Needs Icarus Verilog (`iverilog`/`vvp`) and Python 3, nothing else.

```bash
# Debian/Ubuntu, if not already installed:
sudo apt-get install iverilog

bash scripts/run_sim.sh
```

Expected output ends with:
```
shift_add_mult exhaustive check: 256/256 vectors matched
PASS: all 256 (w_code,a_code) combinations bit-exact vs golden model.
...
K=64 dot product: acc_q=-422 (0xfffffe5a)   expected=-422 (0xfffffe5a)
PASS: shift_add_mac accumulator matches golden dot product exactly.
```

Already run once in the sandbox this was written in -- both pass.

## Design notes

- **F=4 fractional bits.** Every value in both codebooks (E2M1 magnitudes
  `{0,.5,1,1.5,2,3,4,6}`, hw codebook `{0,2,4,6,8,10,12,16}/16`) is dyadic,
  and the smallest possible combined exponent across the whole product space
  is `e_i+f_j=-4` (weight exponent -1, activation exponent -3). Scaling by
  `2^4` makes every shift amount `e_i+f_j+F >= 0`, so `shift_add_mult` is a
  pure left-shift-only barrel shifter -- no right shift, no rounding, exact
  integer arithmetic throughout. `golden_model.py` asserts this exactness
  (`abs(scaled - round(scaled)) < 1e-9`) for all 256 pairs, so this isn't
  just claimed, it's checked every time the vectors are regenerated.
- **All 4 lanes computed unconditionally.** `shift_add_mult` always computes
  all 4 `(weight term i, activation term j)` lanes and zeroes the invalid
  ones via the decoders' valid bits, rather than branching on how many terms
  are actually present. That's what keeps the datapath's structure
  data-independent (no divergence), matching the property claimed for the
  other two MAC embodiments.
- **`ONE <<< sh` instead of a raw literal shift** in `shift_add_mult.v` is
  there purely to pin the shift result's bit width explicitly (avoids
  relying on Verilog's default 32-bit-literal self-determined-width rules,
  which is a classic footgun for exactly this kind of "shift a constant"
  code).
- **Golden model independence.** `golden_model.py` does NOT re-derive the
  power-of-two decomposition -- it decodes both codes to ordinary floats via
  the ordinary E2M1/codebook formulas and multiplies them directly, then
  scales to fixed point. Agreement between that and the RTL's
  shift-and-add path is a genuine check that the decomposition in
  Eq. `eq:shift-add-mac` is correct, not just that the RTL matches itself.
- **Accumulator width.** `shift_add_mac` defaults to `ACC_W=32` against
  `PROD_W=16`; that's about 16 guard bits, comfortably enough for the K in
  any realistically-sized dot product here. Size it to your actual K if you
  pipeline this into the real datapath (see the comment in `shift_add_mac.v`).

## Rough gate count (unmapped, generic cells -- illustrative only)

`yosys synth` on `shift_add_mult` (both decoders + combine + terms +
compressor, *not* including the accumulator) generic-maps to 791 primitive
gates (AND/OR/XOR/MUX/NOT family). That's not a real area number (no PDK,
no mapping to actual standard cells, no accumulator/pipeline registers
included) but it's a sanity data point: this is a small combinational block,
consistent with the paper's claim that it undercuts both a 16-bit multiplier
and the LUT x LUT product table on area. A real area/power number needs an
actual synthesis flow (Yosys+PDK or a commercial tool) against a target
library -- noted as future work in the paragraph.

## What's NOT done here (still future work, as the paragraph says)

- No pipelining / timing closure -- this is behavioral RTL verified
  functionally, not a timed, pipelined datapath.
- No real PDK synthesis, so no real area/power/timing numbers.
- No integration with the actual GEMM/systolic datapath from the rest of
  the accelerator -- this is a standalone MAC lane only.
