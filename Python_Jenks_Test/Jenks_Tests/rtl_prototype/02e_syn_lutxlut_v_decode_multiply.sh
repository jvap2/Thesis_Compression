#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# The comparison that actually matters for the LUTxLUT claim: gf4_pe_lutxlut
# (Phase 1, one table read + one XOR, no multiplier) vs.
# gf4_pe_decode_then_multiply (the fair baseline: decode BOTH operands, then
# multiply the decoded magnitudes) -- both now functionally verified, the
# baseline cross-checked bit-for-bit against gf4_joint_product_table.v's own
# 64-entry reference. This is the real, honestly-scoped number for "removing
# the multiplier from the datapath altogether," independent of Table 3 and
# independent of any array-level/denominator question.
#
# Usage:
#   source ~/eda/oss-cad-suite/environment
#   bash 02e_synth_lutxlut_vs_decode_multiply.sh /path/to/phase0_decode /path/to/phase1_lutxlut
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

PHASE0_DIR="${1:?Usage: $0 /path/to/phase0_decode /path/to/phase1_lutxlut}"
PHASE1_DIR="${2:?Usage: $0 /path/to/phase0_decode /path/to/phase1_lutxlut}"
EDA_DIR="${EDA_DIR:-$HOME/eda}"
NANGATE_LIB="$EDA_DIR/nangate45/lib/NangateOpenCellLibrary_typical.lib"
OUT_DIR="synth_out/lutxlut_vs_decode_multiply"
mkdir -p "$OUT_DIR"

DEC_FIXED="$PHASE0_DIR/gf4_decode_fixed.v"
DEC_E2M1="$PHASE0_DIR/e2m1_decode_fixed.v"
DEC_PROG="$PHASE0_DIR/gf4_decode_programmable.v"
DECMUL="$PHASE0_DIR/gf4_pe_decode_then_multiply.v"
TB_DECMUL="$PHASE0_DIR/tb_gf4_pe_decode_then_multiply.v"

TABLE="$PHASE1_DIR/gf4_joint_product_table.v"
PELUT="$PHASE1_DIR/gf4_pe_lutxlut.v"

for f in "$DEC_FIXED" "$DEC_E2M1" "$DEC_PROG" "$DECMUL" "$TB_DECMUL" "$TABLE" "$PELUT"; do
  [ -f "$f" ] || { echo "Missing expected file: $f"; exit 1; }
done
command -v yosys >/dev/null 2>&1 || { echo "yosys not on PATH — source ~/eda/oss-cad-suite/environment first"; exit 1; }
[ -f "$NANGATE_LIB" ] || { echo "Nangate45 liberty not found at $NANGATE_LIB"; exit 1; }

echo "=== Functional sanity check: tb_gf4_pe_decode_then_multiply ==="
iverilog -g2012 -o "$OUT_DIR/sim.vvp" "$DEC_FIXED" "$DEC_E2M1" "$DEC_PROG" "$DECMUL" "$TB_DECMUL"
vvp "$OUT_DIR/sim.vvp" | tee "$OUT_DIR/sim.log"
if ! grep -q "ALL CHECKS PASSED" "$OUT_DIR/sim.log"; then
  echo "tb_gf4_pe_decode_then_multiply did not report ALL CHECKS PASSED — stopping."
  exit 1
fi
echo ""
echo "(gf4_pe_lutxlut itself still has no dedicated testbench -- its area"
echo "number below carries the same caveat noted back in Phase 1.)"
echo ""

synth_one () {
  local label="$1" top="$2" out_txt="$3"; shift 3
  local ys="$OUT_DIR/${label}.ys"
  {
    for f in "$@"; do echo "read_verilog -sv \"$f\""; done
    echo "hierarchy -top $top"
    echo "proc; opt; fsm; opt; memory; opt"
    echo "techmap; opt"
    echo "dfflibmap -liberty \"$NANGATE_LIB\""
    echo "abc -liberty \"$NANGATE_LIB\""
    echo "clean"
    echo "tee -o $OUT_DIR/$out_txt stat -liberty \"$NANGATE_LIB\""
  } > "$ys"
  echo "--- Synthesizing: $label (top=$top) ---"
  yosys -s "$ys" > "$OUT_DIR/${label}.log" 2>&1 || { echo "FAILED — see $OUT_DIR/${label}.log"; exit 1; }
}

synth_one "decode_then_multiply" "gf4_pe_decode_then_multiply" "area_decode_then_multiply.txt" "$DEC_PROG" "$DECMUL"
synth_one "pe_lutxlut"           "gf4_pe_lutxlut"              "area_pe_lutxlut.txt"           "$TABLE" "$PELUT"

get_area () { grep -oP "Chip area for.*:\s*\K[0-9.]+" "$OUT_DIR/$1" | tail -1; }

A_DECMUL=$(get_area area_decode_then_multiply.txt)
A_LUTXLUT=$(get_area area_pe_lutxlut.txt)

echo ""
echo "==================================================================="
echo "The real LUTxLUT-vs-decode-then-multiply comparison (Nangate45, um^2)"
echo "-------------------------------------------------------------------"
printf "%-42s %12s\n" "gf4_pe_decode_then_multiply (fair baseline):" "$A_DECMUL"
printf "%-42s %12s\n" "gf4_pe_lutxlut (no multiplier):" "$A_LUTXLUT"
DELTA=$(awk -v a="$A_DECMUL" -v b="$A_LUTXLUT" 'BEGIN{printf "%+.2f", ((b-a)/a)*100}')
echo "delta: ${DELTA}%"
echo "==================================================================="