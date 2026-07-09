#!/bin/bash
# GCE startup script for the GF4 A100 VM (Deep Learning image, common-cu12*).
# Runs as ROOT on every boot. Idempotent. Attach with:
#   gcloud compute instances create ... \
#     --metadata-from-file startup-script=vm_startup.sh \
#     --metadata install-nvidia-driver=True,REPO_URL=<optional git url>,HF_TOKEN=<optional>
#
# What it does:
#   1) waits for the NVIDIA driver (the DL image installs it on first boot),
#   2) pip-installs deps INTO the image's conda env (not system python),
#   3) optionally git-clones REPO_URL and/or logs in to HF with HF_TOKEN,
#   4) writes /opt/gf4/READY when done so you can tell setup finished.
#
# Watch progress:  sudo tail -f /var/log/gf4-startup.log
set -uo pipefail
exec > >(tee -a /var/log/gf4-startup.log) 2>&1
echo "=== gf4 startup $(date) ==="

# --- read optional instance metadata --------------------------------------
meta() { curl -s -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null; }
REPO_URL="$(meta REPO_URL)"
HF_TOKEN="$(meta HF_TOKEN)"

# --- pick the image's python/pip (Deep Learning image uses conda) ----------
if [ -x /opt/conda/bin/pip ]; then
  PIP=/opt/conda/bin/pip; PY=/opt/conda/bin/python
else
  PIP="$(command -v pip3 || command -v pip)"; PY="$(command -v python3)"
fi
echo "using PY=$PY  PIP=$PIP"

# --- 1) wait for the GPU driver (up to ~10 min on first boot) ---------------
for i in $(seq 1 60); do
  if nvidia-smi >/dev/null 2>&1; then echo "GPU ready:"; nvidia-smi -L; break; fi
  echo "  waiting for NVIDIA driver ($i/60)..."; sleep 10
done

# --- 2) python deps (DL image already has torch+cuda; add the rest) ---------
# Pin transformers: the newest releases eagerly import torchaudio (Parakeet/RNNT
# loss), which ABI-mismatches the image's torch -> "undefined symbol". 4.46.3
# loads OPT/Llama fine and never touches torchaudio.
echo "installing python deps..."
$PIP install -q -U "transformers==4.46.3" datasets accelerate triton huggingface_hub || \
  echo "WARN: pip install had errors (check log)"
$PIP uninstall -y -q torchaudio 2>/dev/null || true   # belt-and-suspenders

# --- 3) optional: clone repo + HF login ------------------------------------
mkdir -p /opt/gf4
if [ -n "$REPO_URL" ]; then
  echo "cloning $REPO_URL -> /opt/gf4/repo"
  git clone "$REPO_URL" /opt/gf4/repo 2>&1 || echo "WARN: clone failed (private? use scp instead)"
fi
if [ -n "$HF_TOKEN" ]; then
  echo "configuring HF token (for gated models)"
  HF_TOKEN="$HF_TOKEN" $PY - <<'EOF' 2>/dev/null || echo "WARN: HF login skipped"
import os
from huggingface_hub import login
login(os.environ["HF_TOKEN"], add_to_git_credential=False)
EOF
fi

# --- 4) done marker ---------------------------------------------------------
echo "TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;8.9" >> /etc/environment
touch /opt/gf4/READY
echo "=== gf4 startup DONE $(date) ===  (deps installed; ls /opt/gf4/READY to confirm)"
