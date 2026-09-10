#!/usr/bin/env bash
# Regenerates golden vectors and runs both testbenches under Icarus Verilog.
# Usage: from the project root:  bash scripts/run_sim.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # project root

echo "== 1. Generating golden vectors =="
(cd scripts && python3 golden_model.py && python3 dotprod_golden.py)
cp scripts/golden_products.hex scripts/dotprod_codes.hex scripts/dotprod_expected.hex .

echo
echo "== 2. Exhaustive single-multiply check (256/256 code pairs) =="
iverilog -g2005 -o sim_mult.vvp \
    rtl/w_decode_e2m1.v rtl/a_decode_hwcb.v rtl/shift_add_mult.v \
    tb/tb_shift_add_mult.v
vvp sim_mult.vvp

echo
echo "== 3. K=64 accumulated dot-product check =="
iverilog -g2005 -o sim_mac.vvp \
    rtl/w_decode_e2m1.v rtl/a_decode_hwcb.v rtl/shift_add_mult.v rtl/shift_add_mac.v \
    tb/tb_shift_add_mac.v
vvp sim_mac.vvp

echo
echo "All simulations passed."
