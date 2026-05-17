#!/bin/bash
# Qwen3-30B-A3B-Thinking-2507 GRPO on dapo-math-17k, 4 ghx4 nodes (16 GH200).
# Adapted from scripts/run-qwen3-30B-A3B.sh for DeltaAI paths.
# Multi-node version expects: MASTER_ADDR set, RANK and WORLD_RANK via sbatch.
# Ref-load + KL loss dropped (kl-loss-coef was 0 anyway) to free ~60 GB host RAM.

pkill -9 sglang 2>/dev/null || true
sleep 2
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 2

set -ex

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: $HAS_NVLINK"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Qwen3-30B-A3B-Thinking-2507 uses rope_theta=10000000 (10M); base is 1M.
export MODEL_ARGS_ROTARY_BASE=10000000
source "${SCRIPT_DIR}/models/qwen3-30B-A3B.sh"

NUM_NODES=4
NUM_GPUS_PER_NODE=4
NUM_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))  # 16
MODEL_BASE=/work/nvme/bgqz/bzhang31/models/Qwen3-30B-A3B-Thinking-2507

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_BASE}
   --ref-load ${MODEL_BASE}_torch_dist
   --load /work/nvme/bgqz/bzhang31/checkpoints_live/Qwen3-30B-A3B-Thinking-2507_slime/
   --save /work/nvme/bgqz/bzhang31/checkpoints_live/Qwen3-30B-A3B-Thinking-2507_slime/
   --save-interval 100
)

# Compute-measurement run: short, just need throughput numbers
ROLLOUT_ARGS=(
   --prompt-data /work/hdd/bgqz/bzhang31/datasets/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 20
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-response-len 32768
   --rollout-temperature 1

   --global-batch-size 64
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 5
   --eval-prompt-data aime2026 /work/hdd/bgqz/bzhang31/datasets/aime-2026/aime-2026.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 32768
   --eval-temperature 0.6
   --eval-top-p 0.95
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 32768
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-deltaai
   --wandb-group qwen3-30B-A3B-Thinking-2507-grpo-math
   --wandb-key ${WANDB_API_KEY}
)

SGLANG_ARGS=(
   --num-gpus-per-node ${NUM_GPUS_PER_NODE}
   --rollout-num-gpus-per-engine ${NUM_GPUS_PER_NODE}
   --sglang-mem-fraction-static 0.7
   --sglang-attention-backend flashinfer
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# Multi-node ray: head on rank 0, workers join
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-6379}

if [ "${SLURM_PROCID:-0}" -eq 0 ]; then
    ray start --head --node-ip-address ${MASTER_ADDR} \
        --port=${MASTER_PORT} \
        --num-gpus ${NUM_GPUS_PER_NODE} \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 --dashboard-port=8265
    sleep 10  # give workers a moment to connect

    RUNTIME_ENV_JSON="{
      \"env_vars\": {
        \"PYTHONPATH\": \"/work/nvme/bgqz/bzhang31/src/Megatron-LM/\",
        \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
        \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
        \"GLOO_SOCKET_IFNAME\": \"hsn0\",
        \"TP_SOCKET_IFNAME\": \"hsn0\",
        \"NCCL_SOCKET_IFNAME\": \"hsn0\",
        \"FLASHINFER_WORKSPACE_BASE\": \"/tmp/${USER}/flashinfer\",
        \"TRITON_CACHE_DIR\": \"/tmp/${USER}/triton_cache\",
        \"TORCHINDUCTOR_CACHE_DIR\": \"/tmp/${USER}/torchinductor_cache\"
      }
    }"

    ray job submit --address="http://127.0.0.1:8265" \
       --runtime-env-json="${RUNTIME_ENV_JSON}" \
       -- python3 train.py \
       --actor-num-nodes ${NUM_NODES} \
       --actor-num-gpus-per-node ${NUM_GPUS_PER_NODE} \
       --colocate \
       ${MODEL_ARGS[@]} \
       ${CKPT_ARGS[@]} \
       ${ROLLOUT_ARGS[@]} \
       ${OPTIMIZER_ARGS[@]} \
       ${GRPO_ARGS[@]} \
       ${WANDB_ARGS[@]} \
       ${PERF_ARGS[@]} \
       ${EVAL_ARGS[@]} \
       ${SGLANG_ARGS[@]} \
       ${MISC_ARGS[@]}
else
    # Worker node: wait for head, then join
    sleep 15
    ray start --address=${MASTER_ADDR}:${MASTER_PORT} --num-gpus ${NUM_GPUS_PER_NODE}
    # Worker stays alive; head node coordinates the job
    sleep infinity
fi
