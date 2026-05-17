#!/bin/bash
#SBATCH --account=bgqz-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=slime-Think-2507-4n
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=/work/hdd/bgqz/bzhang31/logs/Think2507_4n_%j.log
#SBATCH --constraint=projects&work

set -ex

module reset
module load python/miniforge3_pytorch/2.11.0 cuda/12.9.0 cudnn/9.3.0.75 nccl-ofi-plugin/1.18.0-cuda129
source /sw/user/python/miniforge3-pytorch-2.11.0/etc/profile.d/conda.sh
conda activate base
unset PYTHONNOUSERSITE
source /work/nvme/bgqz/bzhang31/envs/slime/bin/activate

export WANDB_API_KEY="$(cat /u/bzhang31/wandb_api.txt)"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR=$MASTER_ADDR"
echo "SLURM_NODELIST=$SLURM_JOB_NODELIST"

# Force the high-speed network for cross-node Gloo/NCCL (required on DeltaAI multi-node;
# default IPv6 link-local times out)
export GLOO_SOCKET_IFNAME=hsn0
export TP_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_IFNAME=hsn0

HF_DIR=/work/nvme/bgqz/bzhang31/models/Qwen3-30B-A3B-Thinking-2507
TD_DIR=${HF_DIR}_torch_dist

# Step 1: HF -> torch_dist conversion (only if not already done)
# The sbatch script runs on node 0, which has 4 GPUs. We invoke torchrun directly
# (no `srun ... bash -c "..."` wrapping) so the parent shell expands MODEL_ARGS
# cleanly — wrapping mangled bracket-containing values like --moe-layer-freq.
if [ ! -d "${TD_DIR}" ] || [ -z "$(ls -A ${TD_DIR} 2>/dev/null)" ]; then
  echo "===> Converting ${HF_DIR} -> ${TD_DIR}"
  SCRIPT_DIR=/projects/bgqz/bzhang31/slime/scripts
  source "${SCRIPT_DIR}/models/qwen3-30B-A3B.sh"

  cd /projects/bgqz/bzhang31/slime
  torchrun --nproc_per_node 4 tools/convert_hf_to_torch_dist.py \
    --hf-checkpoint ${HF_DIR} \
    --save ${TD_DIR} \
    --megatron-to-hf-mode bridge \
    "${MODEL_ARGS[@]}" \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1
  echo "===> Conversion finished"
else
  echo "===> torch_dist checkpoint already exists at ${TD_DIR}, skipping conversion"
fi

# Step 2: launch training across all 4 nodes
echo "===> Launching training (4 nodes, 16 GH200)"
srun --jobid=$SLURM_JOB_ID --kill-on-bad-exit=1 \
  bash /projects/bgqz/bzhang31/slime/scripts/run-qwen3-30B-A3B-deltaai.sh
