#!/usr/bin/env bash
# Run Timeloop+Accelergy for both engines (GF4 LUT-FP4 vs NVFP4) over every
# problem shape, via the official Docker image.  Outputs land in out/<engine>/<shape>/.
#
# Requires Docker access (see README — add user to the `docker` group, or run
# this script with sudo).  Pull once:
#   docker pull timeloopaccelergy/accelergy-timeloop-infrastructure:latest
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="timeloopaccelergy/accelergy-timeloop-infrastructure:latest"
DOCKER="${DOCKER:-docker}"   # set DOCKER='sudo docker' if not in docker group

run_one() {  # $1=engine_arch  $2=engine_name  $3=prob_file  $4=out_subdir
  local arch="$1" name="$2" prob="$3" out="$4"
  mkdir -p "$HERE/out/$name/$out"
  $DOCKER run --rm -v "$HERE:/work" -w "/work/out/$name/$out" "$IMG" \
    timeloop-mapper \
      "/work/arch/$arch" \
      /work/components/lut_fp4_mac.yaml \
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
  # Only these two are simulated (simple compounds -> reliable energy+area).
  # The other decode placements are priced analytically in postprocess.py from
  # real access counts (wrapping STORAGE in a compound corrupts its estimate).
  run_one gf4_engine.yaml   gf4   "$(basename "$prob")" "$base"
  run_one nvfp4_engine.yaml nvfp4 "$(basename "$prob")" "$base"
done
echo "done. aggregate with: python3 postprocess.py"
