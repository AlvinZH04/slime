#!/bin/bash
# Smoke test: Qwen3.5-4B GRPO on dapo-math-17k, single ghx4 node (4 GH200).
# Uses slime_plugins.models.qwen3_5 plugin (hybrid attention + MTP).

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
source "${SCRIPT_DIR}/models/qwen3.5-4B.sh"

NUM_GPUS=4
MODEL_BASE=/work/nvme/bgqz/bzhang31/models/Qwen3.5-4B

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_BASE}
   --ref-load ${MODEL_BASE}_torch_dist
   --load /work/nvme/bgqz/bzhang31/checkpoints_live/Qwen3.5-4B_slime/
   --save /work/nvme/bgqz/bzhang31/checkpoints_live/Qwen3.5-4B_slime/
   --save-interval 100
)

ROLLOUT_ARGS=(
   --prompt-data /work/hdd/bgqz/bzhang31/datasets/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 20
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 16384
   --rollout-temperature 1

   --global-batch-size 128
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 5
   --eval-prompt-data aime2026 /work/hdd/bgqz/bzhang31/datasets/aime-2026/aime-2026.jsonl
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 16384
   --eval-temperature 0.6
   --eval-top-p 0.95
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
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
   --wandb-group qwen3.5-4B-grpo-math
   --wandb-key ${WANDB_API_KEY}
)

SGLANG_ARGS=(
   --num-gpus-per-node ${NUM_GPUS}
   --rollout-num-gpus-per-engine ${NUM_GPUS}
   --sglang-mem-fraction-static 0.6
   --sglang-attention-backend flashinfer
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/work/nvme/bgqz/bzhang31/src/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"FLASHINFER_WORKSPACE_BASE\": \"/tmp/${USER}/flashinfer\",
    \"TRITON_CACHE_DIR\": \"/tmp/${USER}/triton_cache\",
    \"TORCHINDUCTOR_CACHE_DIR\": \"/tmp/${USER}/torchinductor_cache\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${NUM_GPUS} \
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
