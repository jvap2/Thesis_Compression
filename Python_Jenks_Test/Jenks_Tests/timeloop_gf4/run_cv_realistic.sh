#!/usr/bin/env bash
# Faithful CV sweep: gf4 (int4 decode-per-MAC), nvfp4 (int4 baseline),
# gf4_real (int4 decode + fp16 multiply) over every cv_* layer.
# Skips shapes already done so it is resumable.
set -u
IMG=timeloopaccelergy/accelergy-timeloop-infrastructure:latest
COMPS="/work/components/lut_fp4_mac.yaml /work/components/lut_fp16_mac.yaml \
/work/components/fp4_mac.yaml /work/components/lut_regfile.yaml \
/work/components/lut_sram.yaml /work/mapper/mapper.yaml"

run_one() {
  local arch=$1 name=$2 prob=$3 base
  base=$(basename "$prob" .yaml)
  if [ -f "out/$name/$base/timeloop-mapper.stats.txt" ]; then
    echo "skip $name/$base"; return
  fi
  mkdir -p "out/$name/$base"
  if docker run --rm -v "$PWD:/work" -w "/work/out/$name/$base" "$IMG" \
       timeloop-mapper "/work/arch/$arch" $COMPS "/work/$prob" \
       > "out/$name/$base/log.txt" 2>&1; then
    echo "ok $name/$base"
  else
    echo "FAIL $name/$base"
  fi
}

for prob in prob/cv_*.yaml; do
  run_one gf4_engine.yaml        gf4       "$prob"
  run_one nvfp4_engine.yaml      nvfp4     "$prob"
  run_one gf4_real_engine.yaml   gf4_real  "$prob"
done
echo "CV sweep done -> python3 postprocess.py"
