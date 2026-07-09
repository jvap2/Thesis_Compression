#!/usr/bin/env bash
# run_a100_big.sh — finish the two big models that OOM'd on the 24GB L4, on an
# A100-40GB (a2-highgpu-1g). Runs each in BOTH regimes so the paper table +
# ablation are complete and on the same protocol:
#   opt-6.7b     RETAIN=1 (main) + RETAIN=0 (ablation)
#   llama-2-7b   RETAIN=1 (main) + RETAIN=0 (ablation)
#
# Each run appends a row to iso_results.csv and writes timeloop_gf4/models/<m>_hwsim.json,
# then everything is tar'd to ~/gf4_big_results.tar.gz for one-command fetch.
#
# PREREQS on the A100:
#   - deps installed (vm_startup.sh does this: transformers==4.46.3 datasets accelerate ...)
#   - Llama is GATED -> export your HF token first:   export HF_TOKEN=hf_xxx
#   - run from the Jenks_Tests dir:                    bash run_a100_big.sh
#
# Optional: CALIB_SEQLEN=1024 bash run_a100_big.sh   # if the 7B OOMs during
# calibration on a 40GB card, shorten the calib sequence (default = full 2048).

set -u
cd "$(dirname "$0")"

# torch alloc config (2.9 renamed the var; set both)
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=iso_big.log
: > "$LOG"   # truncate log at start

# Gated-model sanity check (warn, don't abort — opt-6.7b is open and will still run)
if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARN: HF_TOKEN not set — meta-llama/Llama-2-7b-hf (gated) will fail to download." | tee -a "$LOG"
  echo "      export HF_TOKEN=hf_xxx  (or huggingface-cli login) before running for Llama." | tee -a "$LOG"
fi

run() {  # $1 = model id   $2 = retain (1 main / 0 ablation)
  echo "===================================================================" | tee -a "$LOG"
  echo "=== $(date '+%F %T')  MODEL=$1  RETAIN=$2  CALIB_SEQLEN=${CALIB_SEQLEN:-2048} ===" | tee -a "$LOG"
  RETAIN="$2" python3 iso_energy_125m.py "$1" 2>&1 | tee -a "$LOG"
  echo "--- exit ${PIPESTATUS[0]} for MODEL=$1 RETAIN=$2 ---" | tee -a "$LOG"
}

for MODEL in facebook/opt-6.7b meta-llama/Llama-2-7b-hf; do
  run "$MODEL" 1   # main:     retention ON  (the headline W4A4 numbers)
  run "$MODEL" 0   # ablation: retention OFF (the collapse evidence)
done

# Bundle results for fetching back to the laptop.
BUNDLE="$HOME/gf4_big_results.tar.gz"
tar czf "$BUNDLE" iso_results.csv iso_big.log timeloop_gf4/models/*_hwsim.json 2>/dev/null
echo | tee -a "$LOG"
echo "DONE. Bundled -> $BUNDLE" | tee -a "$LOG"
echo "Fetch from your laptop with:" | tee -a "$LOG"
echo "  gcloud compute scp gf4-a100:~/gf4_big_results.tar.gz \\" | tee -a "$LOG"
echo "    ~/Desktop/Code/TIME_ML/Python_Jenks_Test/Jenks_Tests/ --zone=us-central1-a" | tee -a "$LOG"
