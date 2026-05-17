#!/bin/bash
#SBATCH --account=bgqz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=slime-30B-A3B
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=/work/hdd/bgqz/bzhang31/logs/30B-A3B_%j.log
#SBATCH --constraint=projects&work

set -ex

module reset
module load python/miniforge3_pytorch/2.11.0 cuda/12.9.0 cudnn/9.3.0.75 nccl-ofi-plugin/1.18.0-cuda129
source /sw/user/python/miniforge3-pytorch-2.11.0/etc/profile.d/conda.sh
conda activate base
unset PYTHONNOUSERSITE
source /work/nvme/bgqz/bzhang31/envs/slime/bin/activate

export WANDB_API_KEY="$(cat /u/bzhang31/wandb_api.txt)"

# Compute MASTER_ADDR from SLURM
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR=$MASTER_ADDR"
echo "SLURM_NODELIST=$SLURM_JOB_NODELIST"

# Spawn one task per node; each runs the same script with SLURM_PROCID set
srun --kill-on-bad-exit=1 bash /projects/bgqz/bzhang31/slime/scripts/run-qwen3-30B-A3B-deltaai.sh
