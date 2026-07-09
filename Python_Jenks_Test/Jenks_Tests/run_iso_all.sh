#!/usr/bin/env bash
# Unified W4A4 results: run every model through the SAME iso_energy protocol so
# the paper table is apples-to-apples. Each run appends to iso_results.csv and
# writes timeloop_gf4/models/<model>_hwsim.json.
#
#   MAIN  (RETAIN=1): outlier-layer retention ON  -> the headline W4A4 numbers.
#   ABLATION (RETAIN=0): retention OFF -> the "without retention" table that
#                        motivates §outlier (NVFP4/GF4 collapse on big models).
#
# Run order is small->large so early results land fast; big models use a shorter
# calibration sequence to fit a 24GB GPU. Gated models (LLaMA) need a one-time
#   huggingface-cli login
# before running. Pythia/OPT are open.
#
# Usage:   ./run_iso_all.sh            # all of it
#          ./run_iso_all.sh main       # only the RETAIN=1 main table
#          ./run_iso_all.sh ablation   # only the RETAIN=0 ablation
set -u
cd "$(dirname "$0")"
# PyTorch 2.9 renamed the alloc-config env var; set both (old torch ignores the
# new name, new torch warns on the old one but honors PYTORCH_ALLOC_CONF).
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WHAT="${1:-all}"
LOG=iso_runs.log

run() {  # $1 model  $2 retain  $3 calib_seqlen
  echo "=== $(date '+%F %T')  MODEL=$1  RETAIN=$2  CALIB_SEQLEN=$3 ===" | tee -a "$LOG"
  RETAIN="$2" CALIB_SEQLEN="$3" python3 iso_energy_125m.py "$1" 2>&1 | tee -a "$LOG"
}

if [ "$WHAT" = all ] || [ "$WHAT" = main ]; then
  echo "########## MAIN (retention ON) ##########" | tee -a "$LOG"
  run facebook/opt-125m         1 2048
  run facebook/opt-1.3b         1 2048
  run facebook/opt-2.7b         1 2048
  run EleutherAI/pythia-1b      1 2048
  run EleutherAI/pythia-1.4b    1 2048
  run meta-llama/Llama-3.2-3B   1 2048
  run meta-llama/Llama-2-7b-hf  1 512
  run facebook/opt-6.7b         1 512
fi

if [ "$WHAT" = all ] || [ "$WHAT" = ablation ]; then
  echo "########## ABLATION (retention OFF) ##########" | tee -a "$LOG"
  run facebook/opt-2.7b         0 2048
  run meta-llama/Llama-3.2-3B   0 2048
  run meta-llama/Llama-2-7b-hf  0 512
  run facebook/opt-6.7b         0 512
fi

echo "done. -> iso_results.csv ; assemble with: python3 assemble_iso_table.py"
