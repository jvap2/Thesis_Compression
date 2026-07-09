#!/usr/bin/env bash
# run_offload_big.sh — extend the UNIFIED offload table to the big models on the
# L4 VM: the opt-6.7b ablation (OOM'd on the local 4080 at eval), then 13B and
# Llama, all under the SAME OFFLOAD=1 protocol as the local small-model rows so
# the whole paper table is one consistent method.
#
#   OFFLOAD=1        weight-quant streams one decoder block to the GPU at a time
#                    (model in CPU RAM) -> 13B/30B/70B calibrate in ~1-2GB of VRAM.
#   EVAL_OFFLOAD=auto  eval uses block-major streaming when the model won't fit the
#                    GPU (>~13B on 24GB), else evaluates in-GPU. 6.7B/7B eval in-GPU;
#                    13B+ stream. Force with EVAL_OFFLOAD=1 / disable with =0.
#
# RAM: the model lives in CPU RAM during calib AND streaming eval. Rule of thumb
# ~2 bytes/param: 13B~26GB, 30B~60GB, 70B~140GB. Pick the machine type to match:
#   opt-13b / Llama-2-13b  -> g2-standard-24  (96GB RAM) is plenty
#   opt-66b / Llama-2-70b  -> g2-standard-48  (192GB RAM) + ~250GB disk for weights
#
# PREREQS on the VM:
#   - deps installed (vm_startup.sh: transformers datasets accelerate ...)
#   - Llama is GATED -> export a FRESH HF token first:  export HF_TOKEN=hf_xxx
#   - run from the Jenks_Tests dir:                     bash run_offload_big.sh
set -u
cd "$(dirname "$0")"

export OFFLOAD=1
export EVAL_OFFLOAD="${EVAL_OFFLOAD:-auto}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=iso_offload_big.log
: > "$LOG"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARN: HF_TOKEN not set — the gated Llama models will fail to download." | tee -a "$LOG"
  echo "      export HF_TOKEN=hf_xxx  before running for the Llama rows." | tee -a "$LOG"
fi

run() {  # $1 = model id   $2 = retain (1 main / 0 ablation)
  echo "===================================================================" | tee -a "$LOG"
  echo "=== $(date '+%F %T')  MODEL=$1  RETAIN=$2  OFFLOAD=1  EVAL_OFFLOAD=$EVAL_OFFLOAD ===" | tee -a "$LOG"
  RETAIN="$2" python3 -u iso_energy_125m.py "$1" 2>&1 | tee -a "$LOG"
  echo "--- exit ${PIPESTATUS[0]} for MODEL=$1 RETAIN=$2 ---" | tee -a "$LOG"
}

# 1) finish the local gap: opt-6.7b ablation (main was done on the 4080)
run facebook/opt-6.7b 0

# 2) 13B tier (fits any g2 with >=32GB RAM; eval streams)
run facebook/opt-13b        1
run facebook/opt-13b        0

# 3) Llama-2 (gated). 7B evals in-GPU; 13B streams.
run meta-llama/Llama-2-7b-hf   1
run meta-llama/Llama-2-7b-hf   0
run meta-llama/Llama-2-13b-hf  1
run meta-llama/Llama-2-13b-hf  0

# 4) 70B tier — ONLY on a 192GB-RAM box (g2-standard-48). Uncomment there:
# run facebook/opt-66b          1
# run facebook/opt-66b          0
# run meta-llama/Llama-2-70b-hf 1
# run meta-llama/Llama-2-70b-hf 0

BUNDLE="$HOME/gf4_offload_results.tar.gz"
tar czf "$BUNDLE" iso_results.csv iso_offload_big.log \
    timeloop_gf4/models/*_hwsim.json 2>/dev/null
echo | tee -a "$LOG"
echo "DONE. Bundled -> $BUNDLE" | tee -a "$LOG"
echo "Fetch from your laptop (adjust zone):" | tee -a "$LOG"
echo "  gcloud compute scp gf4-l4b:~/gf4_offload_results.tar.gz \\" | tee -a "$LOG"
echo "    ~/Desktop/Code/TIME_ML/Python_Jenks_Test/Jenks_Tests/ --zone=us-west4-a" | tee -a "$LOG"
