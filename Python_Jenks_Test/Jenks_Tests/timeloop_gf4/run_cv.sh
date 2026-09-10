#!/usr/bin/env bash
# GF4 CV energy sweep: run every cv_* problem through the GF4-Engine (4-bit LUT-FP4)
# and the NVFP4 baseline (4-bit, no LUT), DENSE (no skip_zeros — GF4 is precision not
# sparsity). Energy-first mapping (map_e). Resumable: skips shapes already done.
# Run INSIDE the timeloop container:  bash /work/run_cv.sh
set -u
cd /work
export HOME=/work
COMPS="components/fp4_mac.yaml components/lut_fp4_mac.yaml components/lut_regfile.yaml components/lut_sram.yaml"

run_one(){ # $1=engine_arch  $2=engine_name  $3=prob_path
  local arch=$1 name=$2 prob=$3
  local base; base=$(basename "$prob" .yaml)
  local o="out/$name/$base"
  if [ -f "$o/timeloop-mapper.stats.txt" ]; then return; fi
  mkdir -p "$o"
  timeloop-mapper "arch/$arch" $COMPS "$prob" map_e.yaml -o "$o" > "$o/log.txt" 2>&1 \
    || echo "  [warn] $name/$base failed (see $o/log.txt)"
}

n=0; tot=$(ls prob/cv_*.yaml 2>/dev/null | wc -l)
for prob in prob/cv_*.yaml; do
  n=$((n+1))
  echo "[$n/$tot] $(basename "$prob")"
  run_one gf4_engine.yaml   cv_gf4   "$prob"
  run_one nvfp4_engine.yaml cv_nvfp4 "$prob"
done
echo "CV SWEEP DONE  ($(date))"
