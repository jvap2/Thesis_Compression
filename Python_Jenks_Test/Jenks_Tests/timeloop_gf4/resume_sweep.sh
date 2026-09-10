#!/usr/bin/env bash
# Resume ONLY the two models the reboot interrupted, APPENDING to the results file
# (never truncating the 12 already-finished models). Mirrors overnight_sweep's block.
set -u
cd "$(dirname "$0")"
export HOME=/work
RES=sparse_sweep_results.txt
for m in vgg19_cifar100_90_sparsity vgg19_test; do
  echo "==================================================================" | tee -a "$RES"
  echo "### $m  $(date +%F_%H:%M:%S)  (resume)" | tee -a "$RES"
  bash run_pareto_sweep.sh "$m" 2>&1 | tee "log_${m}.txt" \
    | grep -E "^L[0-9]|^conv|ENERGY|THROUGHPUT" | tee -a "$RES"
done
echo "==================================================================" | tee -a "$RES"
echo "resume DONE $(date)" | tee -a "$RES"
