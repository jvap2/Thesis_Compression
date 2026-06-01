#!/bin/bash
#SBATCH --job-name=gsm_elemsgd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_gsm_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_NETID@nd.edu

# --- pass --model on the command line to target one experiment, e.g.:
#   sbatch submit_gsm.sh --model ResNet56
#   sbatch submit_gsm.sh --model VGG19/CIFAR10
# omit to run all experiments sequentially.

module load python/3.10
module load cuda/12.1

# Adjust to wherever your conda env lives on CRC
source activate gsm_env   # or: source /afs/crc.nd.edu/user/n/NETID/envs/gsm_env/bin/activate

SCRATCH=/scratch365/$USER
DATA_ROOT=$SCRATCH/datasets
OUT_DIR=$SCRATCH/gsm_elem_sgd_results

mkdir -p "$OUT_DIR"

python train_gsm_elem_sgd.py \
    --data_root "$DATA_ROOT" \
    --out_dir   "$OUT_DIR" \
    "$@"
