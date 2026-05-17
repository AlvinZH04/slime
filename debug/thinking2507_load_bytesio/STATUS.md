# Status — Qwen3-30B-A3B-Thinking-2507 GRPO on DeltaAI (4 × GH200)

**Last update:** 2026-05-17

## What's solved

1. **Multi-node infra (4 × GH200 per node).** Three stacked slime bugs were fixed locally so cross-node training works:
   - `placement_group.sort_key` made deterministic (was producing wrong `base_gpu_id` due to non-deterministic hostname resolution).
   - Mandatory `--num-gpus-per-node 4` (slime default 8 mis-packs engines on the same node).
   - Cross-node Gloo/NCCL forced onto Slingshot `hsn0` (default IPv6 link-local doesn't route).

2. **The `BytesIO has no len()` checkpoint-load failure** (the big one) is **resolved**. Root cause: `tools/convert_hf_to_torch_dist.py --megatron-to-hf-mode bridge` produces a torch_dist whose MoE expert weights fail to reshape on load (~39 600 "from model not in state dict, will skip" warnings before the crash). The same script with `bridge` worked for base `Qwen3-30B-A3B` but not for the Thinking-2507 variant.
   **Fix:** drop the `--megatron-to-hf-mode bridge` flag — the default `raw` produces a torch_dist Megatron loads correctly.

3. **End-to-end GRPO training validated.** With the raw-converted torch_dist + 4 nodes × 4 GH200 + `--rollout-max-response-len 32768` + `--rollout-batch-size 16`:
   - 2 full GRPO steps completed (step 1: rollout 244 s + train 116 s, TFlops 56; step 2: rollout 371 s + train 97 s, TFlops 78), then a host-RAM OOM cut step 3.
   - 6–8 min / step wall time at this scale.

4. **Auxiliary fixes** documented in [DIFFICULTIES_AND_FIXES.md](DIFFICULTIES_AND_FIXES.md): SLURM `--gpu-bind`, stale srun interconnect cleanup, `flashinfer` JIT-cache on Lustre, `bash -c` mangling bash arrays, `--rotary-base` override for the Thinking variant's `rope_theta=10000000`, `--ref-load` is required even with `--kl-loss-coef=0` (slime falls back through it), and the convert script's silent `PP = world_size` auto-override.

## What's still open

**Host-RAM accumulation across GRPO iterations** — the only blocker to sustained training.

- Each iteration leaks ~50–80 GB of host RAM that doesn't release between rollouts.
- Per-node Megatron actors occupy ~83 GB × 4 = 332 GB baseline; after 2–3 iterations the node crosses Ray's 95 % memory-usage threshold (≈ 815 GB / 858 GB) and a worker is killed.
- Steps-before-OOM observed:
  - 2 nodes, `rollout-batch=16` → **1 step**
  - 4 nodes, `rollout-batch=16` → **2 steps + 3rd rollout**
  - 4 nodes, `rollout-batch=8` → (untested past startup; got cut by alloc timeout)

**Suspected sources** (ordered by likelihood):

1. sglang's `release_memory_occupation` not fully releasing KV cache after each rollout. Worth trying `enable_memory_saver=False` on the sglang engine to see if the leak stops.
2. Megatron's optimizer-cpu-offload pipeline (`--overlap-cpu-optimizer-d2h-h2d`) retaining intermediate buffers across iterations.
3. Async dist-ckpt save buffers held after the first `save_checkpoint`.

**Suggested first diagnostic step.** Add per-actor `psutil.Process().memory_info().rss` printouts at three points per iteration — start of rollout, end of rollout, end of train — and compare values across iterations. The actor that grows by tens of GB step-over-step is the leak.

## Working recipe (current best)

```bash
# Convert (don't pass TP, don't pass --megatron-to-hf-mode):
MODEL_ARGS_ROTARY_BASE=10000000 source scripts/models/qwen3-30B-A3B.sh
torchrun --nproc_per_node 4 tools/convert_hf_to_torch_dist.py \
  --hf-checkpoint $HF_DIR \
  --save        $TD_DIR \
  "${MODEL_ARGS[@]}"

# Train (4 nodes × 4 GH200):
#   NUM_NODES=4, TP=4, EP=8, PP=1, CP=1, ETP=1
#   --rollout-batch-size 8 (recommended for memory headroom; 16 also works for short runs)
#   --global-batch-size 64 (= rollout_batch * n_samples / num_steps_per_rollout)
#   --rollout-max-response-len 32768
#   --max-tokens-per-gpu 32768
#   --ref-load $TD_DIR   (required even though --use-kl-loss is off)
#
# Mandatory env (per-node + in ray runtime_env_json):
export GLOO_SOCKET_IFNAME=hsn0
export TP_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_IFNAME=hsn0
export FLASHINFER_WORKSPACE_BASE=/tmp/$USER/flashinfer
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache
```

## Hard numbers (32K gen, 4 nodes, batch=16)

| Phase            | Wall  | Notes |
|---|---|---|
| Engine boot      | ~7 m  | Includes sglang weight load and cuda-graph capture |
| Checkpoint load  | ~30 s | Megatron actor loading the torch_dist |
| First `update_weights` (Megatron→sglang) | ~17 s | Constant per iter |
| Rollout (32 prompts × 8 samples = 256 traces) | 240–370 s | Length-dependent; truncation 20–27 % |
| Train step       | 100–200 s | TFlops 56–78 (warmup effect) |
| **Total per step** | **6–8 min** | |

AIME-2026 eval pass@1 = 0 % at 32 K eval cap — **not a model-quality issue**, but Thinking-mode wraps reasoning in `<think>…</think>` then writes the answer *after* the closing tag. With median eval response = cap (32 K), the answer never gets emitted. The Qwen team recommends 81 920 tokens for AIME-class math; the next step once memory is stable is to bump the eval cap (training rollouts at temp 1.0 only truncate 20–27 % and produce useful reward signal).

## Repo layout (this branch)

| Path | What it is |
|---|---|
| `debug/thinking2507_load_bytesio/README.md`              | original BytesIO bug report |
| `debug/thinking2507_load_bytesio/DIFFICULTIES_AND_FIXES.md` | full chronological postmortem (13 issues) |
| `debug/thinking2507_load_bytesio/STATUS.md`              | this file |
| `debug/thinking2507_load_bytesio/{env,config,scripts,logs}/` | exact versions, HF config, launch scripts, error/convert logs |
| `slime/ray/placement_group.py`                            | sort_key patch (multi-node bug #1) |
| `scripts/run-qwen3-30B-A3B-deltaai.sh`                    | current working launch script |
| `README_DELTAAI.md`                                       | cluster-side runbook + multi-node bug list |

## Quick start for the collaborator

```bash
git clone -b deltaai-debug-thinking2507 git@github.com:AlvinZH04/slime_debug.git
cd slime_debug
cat debug/thinking2507_load_bytesio/DIFFICULTIES_AND_FIXES.md       # full context
cat debug/thinking2507_load_bytesio/STATUS.md                       # this file
cat debug/thinking2507_load_bytesio/env.md                          # exact versions
cat debug/thinking2507_load_bytesio/logs/error_traceback.log        # the BytesIO crash
```

Wandb runs of the working A2/A3 attempts:
- A2 (2-node, 1 step): https://wandb.ai/bzhang90-johns-hopkins-university/slime-deltaai/runs/a5iterbx
- A3 (4-node, 2 steps): https://wandb.ai/bzhang90-johns-hopkins-university/slime-deltaai/runs/dm5peho2
