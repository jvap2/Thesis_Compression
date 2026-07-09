#!/usr/bin/env bash
# run_offload_13b.sh — the 13B tier of the unified offload table, for the L4 VM.
# opt-13b + Llama-2-13b, both regimes, under the same OFFLOAD=1 protocol as the
# rest of the table so the rows are apples-to-apples.
#
#   OFFLOAD=1          weight-quant streams one decoder block to the GPU at a time
#                      (model in CPU RAM) -> calibrates in ~1-2GB of VRAM.
#   EVAL_OFFLOAD=auto  a 13B model (~26GB) won't fit the 24GB L4, so eval uses
#                      block-major streaming automatically; watch for the
#                      "eval: model=..GiB gpu=..GiB -> STREAMING offload" line.
#
# RAM: the model lives in CPU RAM during BOTH calibration and streaming eval, at
# ~2 bytes/param -> ~26GB for 13B. A g2-standard-8 (32GB) is too tight once you
# add activation/Python overhead; use >=48GB RAM: resize the VM to g2-standard-12
# (48GB) or g2-standard-16 (64GB). Still 1 L4, so within a 1-GPU quota.
#   gcloud compute instances stop  <vm> --zone=<zone>
#   gcloud compute instances set-machine-type <vm> --machine-type=g2-standard-16 --zone=<zone>
#   gcloud compute instances start <vm> --zone=<zone>
#
# PREREQS on the VM:
#   - the offload code synced (FP_Quantization_Experiments/bit_split.py + iso_energy_125m.py)
#   - Llama is GATED -> export a FRESH HF token first:  export HF_TOKEN=hf_xxx
#   - disk with room for the downloads (~26GB opt-13b + ~26GB llama-13b)
set -u
cd "$(dirname "$0")"
export OFFLOAD=1
export EVAL_OFFLOAD="${EVAL_OFFLOAD:-auto}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=iso_offload_13b.log
: > "$LOG"
if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARN: HF_TOKEN not set — gated Llama-2-13b will fail to download." | tee -a "$LOG"
  echo "      export HF_TOKEN=hf_xxx  before running for the Llama row." | tee -a "$LOG"
fi

run() {  # $1 = model id   $2 = retain (1 main / 0 ablation)
  echo "===================================================================" | tee -a "$LOG"
  echo "=== $(date '+%F %T')  MODEL=$1  RETAIN=$2  OFFLOAD=1  EVAL_OFFLOAD=$EVAL_OFFLOAD ===" | tee -a "$LOG"
  RETAIN="$2" python3 -u iso_energy_125m.py "$1" 2>&1 | tee -a "$LOG"
  echo "--- exit ${PIPESTATUS[0]} for MODEL=$1 RETAIN=$2 ---" | tee -a "$LOG"
}

run facebook/opt-13b          1
run facebook/opt-13b          0
run meta-llama/Llama-2-13b-hf 1
run meta-llama/Llama-2-13b-hf 0

BUNDLE="$HOME/gf4_offload_13b.tar.gz"
tar czf "$BUNDLE" iso_results.csv iso_offload_13b.log \
    timeloop_gf4/models/*_hwsim.json 2>/dev/null
echo | tee -a "$LOG"
echo "DONE (13B tier). Bundled -> $BUNDLE" | tee -a "$LOG"
echo "Fetch from your laptop (adjust zone):" | tee -a "$LOG"
echo "  gcloud compute scp gf4-l4b:~/gf4_offload_13b.tar.gz \\" | tee -a "$LOG"
echo "    ~/Desktop/Code/TIME_ML/Python_Jenks_Test/Jenks_Tests/ --zone=us-west4-a" | tee -a "$LOG"
