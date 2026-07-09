#!/usr/bin/env bash
# run_offload_7b.sh — the 7B tier, sized for a ~32GB-RAM box (the local RTX 4080
# workstation). opt-6.7b ablation (main already done) + Llama-2-7b both regimes.
# GPU peak is one block (streaming eval auto-triggers), so 16GB VRAM is fine; the
# limit is CPU RAM and these models are ~13-14GB (run sequentially, freed between).
# The 13B tier is too big for 32GB RAM -> run_offload_13b.sh on the VM in parallel.
set -u
cd "$(dirname "$0")"
export OFFLOAD=1
export EVAL_OFFLOAD="${EVAL_OFFLOAD:-auto}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=iso_offload_7b.log
: > "$LOG"
if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARN: HF_TOKEN not set — gated Llama-2-7b will fail to download." | tee -a "$LOG"
fi

run() {  # $1 = model id   $2 = retain
  echo "===================================================================" | tee -a "$LOG"
  echo "=== $(date '+%F %T')  MODEL=$1  RETAIN=$2  OFFLOAD=1  EVAL_OFFLOAD=$EVAL_OFFLOAD ===" | tee -a "$LOG"
  RETAIN="$2" python3 -u iso_energy_125m.py "$1" 2>&1 | tee -a "$LOG"
  echo "--- exit ${PIPESTATUS[0]} for MODEL=$1 RETAIN=$2 ---" | tee -a "$LOG"
}

run facebook/opt-6.7b        0    # ablation (main already done locally)
run meta-llama/Llama-2-7b-hf 1
run meta-llama/Llama-2-7b-hf 0

echo | tee -a "$LOG"
echo "DONE (7B tier). Rows appended to iso_results.csv." | tee -a "$LOG"
