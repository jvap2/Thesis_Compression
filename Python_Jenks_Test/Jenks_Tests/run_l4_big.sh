#!/usr/bin/env bash
# run_l4_big.sh — run the two big models (opt-6.7b, llama-2-7b) on a 24GB L4,
# using LEAN=1 (frees each layer's original weight during weight-quant, so the
# resident peak is ~1x model instead of ~2x). No A100 / no GPU quota needed.
# Each model in BOTH regimes (main RETAIN=1 + ablation RETAIN=0) so the paper
# table + ablation are complete on the same protocol.
#
# LEAN=1 is numerically identical to the normal path (it only changes memory
# management); it just skips an orig-vs-quant sanity print.
#
# PREREQS on the L4:
#   - deps installed (transformers==4.46.3 datasets accelerate ... ; vm_startup.sh)
#   - Llama is GATED -> export your HF token first:   export HF_TOKEN=hf_xxx
#   - run from the Jenks_Tests dir:                    bash run_l4_big.sh
#
# If the 7B still OOMs during the CALIBRATION forward (weight-quant is fixed by
# LEAN, but activation collection also uses memory), shorten the calib sequence:
#   CALIB_SEQLEN=1024 bash run_l4_big.sh     # or 512

set -u
cd "$(dirname "$0")"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LEAN=1                                  # <-- the memory fix (fits 24GB)

LOG=iso_big_l4.log
: > "$LOG"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARN: HF_TOKEN not set — meta-llama/Llama-2-7b-hf (gated) will fail to download." | tee -a "$LOG"
  echo "      export HF_TOKEN=hf_xxx  before running for the Llama rows." | tee -a "$LOG"
fi

run() {  # $1 = model id   $2 = retain (1 main / 0 ablation)
  echo "===================================================================" | tee -a "$LOG"
  echo "=== $(date '+%F %T')  MODEL=$1  RETAIN=$2  LEAN=1  CALIB_SEQLEN=${CALIB_SEQLEN:-2048} ===" | tee -a "$LOG"
  RETAIN="$2" python3 iso_energy_125m.py "$1" 2>&1 | tee -a "$LOG"
  echo "--- exit ${PIPESTATUS[0]} for MODEL=$1 RETAIN=$2 ---" | tee -a "$LOG"
}

for MODEL in facebook/opt-6.7b meta-llama/Llama-2-7b-hf; do
  run "$MODEL" 1   # main:     retention ON
  run "$MODEL" 0   # ablation: retention OFF
done

BUNDLE="$HOME/gf4_big_results.tar.gz"
tar czf "$BUNDLE" iso_results.csv iso_big_l4.log timeloop_gf4/models/*_hwsim.json 2>/dev/null
echo | tee -a "$LOG"
echo "DONE. Bundled -> $BUNDLE" | tee -a "$LOG"
echo "Fetch from your laptop with:" | tee -a "$LOG"
echo "  gcloud compute scp gf4-l4:~/gf4_big_results.tar.gz \\" | tee -a "$LOG"
echo "    ~/Desktop/Code/TIME_ML/Python_Jenks_Test/Jenks_Tests/ --zone=us-central1-a" | tee -a "$LOG"
