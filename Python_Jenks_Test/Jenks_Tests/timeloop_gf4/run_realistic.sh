#!/usr/bin/env bash
# Realistic-multiply run: adds the gf4_real engine (16-bit multiply on decoded
# magnitudes) alongside the existing gf4 (int4+LUT) and nvfp4 (int4) engines, so
# postprocess can report BOTH the decode-only overhead and the realistic
# decode+wider-multiply overhead.
#
# Run in the timeloop docker (same as run.sh):
#   nohup bash run_realistic.sh > realistic.out 2>&1 &
# Then aggregate (see note at bottom / postprocess).
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="timeloopaccelergy/accelergy-timeloop-infrastructure:latest"
DOCKER="${DOCKER:-docker}"

run_one() {  # $1=engine_arch  $2=engine_name  $3=prob_file  $4=out_subdir
  local arch="$1" name="$2" prob="$3" out="$4"
  mkdir -p "$HERE/out/$name/$out"
  $DOCKER run --rm -v "$HERE:/work" -w "/work/out/$name/$out" "$IMG" \
    timeloop-mapper \
      "/work/arch/$arch" \
      /work/components/lut_fp4_mac.yaml \
      /work/components/lut_fp16_mac.yaml \
      /work/components/fp4_mac.yaml \
      /work/components/lut_regfile.yaml \
      /work/components/lut_sram.yaml \
      /work/mapper/mapper.yaml \
      "/work/prob/$prob" \
    > "$HERE/out/$name/$out/log.txt" 2>&1 \
    || echo "  [warn] $name/$out failed (see out/$name/$out/log.txt)"
}

for prob in "$HERE"/prob/*.yaml; do
  base="$(basename "$prob" .yaml)"
  echo "=== $base ==="
  run_one gf4_engine.yaml      gf4      "$(basename "$prob")" "$base"
  run_one nvfp4_engine.yaml    nvfp4    "$(basename "$prob")" "$base"
  run_one gf4_real_engine.yaml gf4_real "$(basename "$prob")" "$base"
done
echo "done. aggregate with: python3 postprocess.py  (add the gf4_real engine there)"
