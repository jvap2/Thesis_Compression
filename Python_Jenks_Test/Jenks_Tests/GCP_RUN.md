# Running on Google Cloud (A100) — no notebook needed

On a GCE VM the repo is importable, so you run the `.py` files directly (skip the
Colab notebook entirely).

## 0. One-time: get A100 quota  ← the real blocker
New projects have **0 GPU quota**. Request it BEFORE creating the VM:
- Console → IAM & Admin → **Quotas** → filter `NVIDIA_A100_GPUS` (per-region) and
  `GPUS_ALL_REGIONS` → request limit **1** in a region that has A100
  (us-central1, us-east1, europe-west4, asia-northeast1).
- Approval is minutes–hours. Without it, VM creation fails with a quota error.

## 1. Create the VM (Deep Learning image = CUDA + PyTorch preinstalled)
1× A100 40GB is plenty for 6.7B/7B. Run from your laptop (with `gcloud` installed)
or Cloud Shell:

```bash
gcloud compute instances create gf4-a100 \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --image-family=common-cu121-debian-11 --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=200GB --boot-disk-type=pd-ssd \
  --metadata-from-file startup-script=vm_startup.sh \
  --metadata=install-nvidia-driver=True
# Cheaper: add  --provisioning-model=SPOT  (can be preempted; fine for one-off runs)
```

**The `vm_startup.sh` script** (attached above) runs on boot as root and:
waits for the GPU driver, pip-installs deps into the image's conda env, and
optionally clones a repo / logs in to HF if you pass extra metadata:
`--metadata=install-nvidia-driver=True,REPO_URL=<git url>,HF_TOKEN=<token>`.
Watch it with `sudo tail -f /var/log/gf4-startup.log`; it touches
`/opt/gf4/READY` when finished, so deps are installed before you even SSH in.
(If you `scp` the repo instead of cloning, omit `REPO_URL`.)

A100 80GB instead: `--machine-type=a2-ultragpu-1g --accelerator=type=nvidia-a100-80gb,count=1`.

## 2. SSH in and get the code
```bash
gcloud compute ssh gf4-a100 --zone=us-central1-a
nvidia-smi                       # confirm the A100 + driver are up
# copy the repo up (from your laptop, in another terminal):
#   gcloud compute scp --recurse ~/Desktop/Code/TIME_ML gf4-a100:~/ --zone=us-central1-a
# or git clone your repo.
```

## 3. Deps + run  (cd into Python_Jenks_Test/Jenks_Tests)
```bash
ls /opt/gf4/READY                # startup script done? deps already installed.
# If you skipped the startup script, install manually:
#   pip install -U transformers datasets accelerate triton
huggingface-cli login            # ONLY for gated Llama; OPT/GPT need no token

# (a) full driver table (has the fc2/down_proj skip fix):
MODEL=6.7b python3 FP_QuantNetworkTest_LLM.py
#     keys: 125m 1.3b 2.7b 6.7b 13b | llama-1b llama-3b llama-7b

# (b) iso-energy A/B + hwsim export (one model per run):
python3 iso_energy_125m.py facebook/opt-6.7b
#     writes timeloop_gf4/models/<model>_hwsim.json  -> feeds the energy sim
```

## 4. Pull results back to your machine
```bash
# from your laptop:
gcloud compute scp gf4-a100:~/TIME_ML/Python_Jenks_Test/Jenks_Tests/timeloop_gf4/models/*_hwsim.json \
   ~/Desktop/Code/TIME_ML/Python_Jenks_Test/Jenks_Tests/timeloop_gf4/models/ --zone=us-central1-a
# then locally: cd timeloop_gf4 && python3 gen_problems.py && ./run.sh && python3 postprocess.py
#               python3 ../hw_sim_gf4.py
```

## 5. STOP THE VM (A100 ≈ $3–4/hr on-demand)
```bash
gcloud compute instances stop gf4-a100 --zone=us-central1-a   # stop billing for compute
gcloud compute instances delete gf4-a100 --zone=us-central1-a # or delete entirely
```

## Alternative: Vertex AI Workbench (managed Jupyter, closest to Colab)
Console → Vertex AI → Workbench → **Create** → add an A100 → open JupyterLab →
upload `Colab/FP_Quant.ipynb` and run as in Colab. Same A100 quota applies.
(The `.py`-on-a-VM path above is cheaper and simpler for headless runs.)

## Notes
- First run JIT-compiles CUDA kernels (a few min) — normal.
- `TORCH_CUDA_ARCH_LIST` is set inside the driver/scripts (A100=8.0 included).
- The Deep Learning image already has a matching CUDA/PyTorch, so you usually only
  `pip install` transformers/datasets/triton.
