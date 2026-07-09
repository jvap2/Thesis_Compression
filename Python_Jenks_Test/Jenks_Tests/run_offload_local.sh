#!/usr/bin/env bash
# run_offload_local.sh — rebuild the iso-energy table under the UNIFIED offload
# protocol (block-sequential CPU-offload calibration) for every model that fits
# this workstation (RTX 4080 16GB + 31GB RAM). The big models (13B/70B/Llama)
# run on the L4 VM with the same OFFLOAD=1 path, so the whole paper table is one
# consistent protocol.
#
# OFFLOAD=1 keeps the model on CPU and streams one decoder block to the GPU at a
# time; peak GPU = one block, so even opt-6.7b calibrates in ~1GB and evals in
# ~13GB (fits the 4080). Calib settings are left at defaults (CALIB_SEQLEN=2048,
# num_calib_batches=4) to match the VM big-model runs exactly.
set -u
cd "$(dirname "$0")"
export OFFLOAD=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=iso_offload_local.log
: > "$LOG"

run() {  # $1 = model id   $2 = retain (1 main / 0 ablation)
  echo "===================================================================" | tee -a "$LOG"
  echo "=== $(date '+%F %T')  MODEL=$1  RETAIN=$2  OFFLOAD=1 ===" | tee -a "$LOG"
  RETAIN="$2" python3 -u iso_energy_125m.py "$1" 2>&1 | tee -a "$LOG"
  echo "--- exit ${PIPESTATUS[0]} for MODEL=$1 RETAIN=$2 ---" | tee -a "$LOG"
}

# main rows (retention ON) for the whole OPT sweep
for M in facebook/opt-125m facebook/opt-1.3b facebook/opt-2.7b facebook/opt-6.7b; do
  run "$M" 1
done
# ablation rows (retention OFF) where the in-GPU table had them.
# NOTE: opt-6.7b ablation is NOT run here — its RETAIN=0 eval quantizes fc2, whose
# 16384-wide GF4 activation-quant tensor OOMs the 16GB 4080. It runs on the 24GB
# VM instead (run_offload_big.sh).
run facebook/opt-2.7b 0

echo | tee -a "$LOG"
echo "DONE. iso_results.csv rebuilt under OFFLOAD=1." | tee -a "$LOG"
