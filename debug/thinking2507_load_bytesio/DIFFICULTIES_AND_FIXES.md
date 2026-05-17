# Difficulties Encountered and How We Solved Them — Qwen3-30B-A3B-Thinking-2507 GRPO on DeltaAI

This document is a chronological postmortem of getting GRPO training of `Qwen/Qwen3-30B-A3B-Thinking-2507` working end-to-end with long-context (32K) rollouts on NCSA DeltaAI (Grace Hopper aarch64, 4 GH200 / node). Each entry: **symptom → diagnosis → fix**.

It's intended for the collaborator who is debugging the remaining memory-leak issue, and for our future selves when the next model variant breaks.

---

## 1. SLURM allocation: interactive partition refuses non-exclusive jobs

**Symptom.** `salloc … --partition=ghx4-interactive` without `--exclusive` returned `Requested node configuration is not available`.

**Diagnosis.** `scontrol show partition ghx4-interactive` revealed `OverSubscribe=NO`. The partition only places one job per node regardless of how many CPUs/memory are free. Non-exclusive requests are simply not schedulable.

**Fix.** Always include `--exclusive --mem=0` on `ghx4-interactive`. There is no faster path even when nodes have idle resources.

---

## 2. SLURM: single-task allocation gets only one GPU

**Symptom.** Ray came up with `ValueError: Attempting to start raylet with 4 GPU, but CUDA_VISIBLE_DEVICES contains ['0']` even though we requested 4 GPUs / node.

**Diagnosis.** Default `--gpu-bind=verbose,closest` with `--ntasks-per-node=1` pins the lone task to the GPU closest to the rank, so the process only sees 1 GPU.

**Fix.** Pass `--gpu-bind=none` on every `salloc`/`sbatch`. Ray then sees all 4 GPUs.

---

## 3. SLURM: stale srun steps wedge the Slingshot interconnect

**Symptom.** After a failed run, `srun --jobid=… --overlap` on the same node started returning `Error configuring interconnect`. Other nodes in the same allocation worked.

**Diagnosis.** A previous srun step had not torn down its interconnect endpoints. `sacct -j <jobid>` showed steps in `RUNNING` long after the wrapping bash had exited.

**Fix.** `scancel --signal=KILL <jobid>.<stepid>` for each stale step, then retry srun. If one specific node is permanently bad (we hit this with `gh006`), add `--exclude=gh006` to the next `salloc`.

---

## 4. slime placement-group: `sort_key` returns non-deterministic order for hostnames

**Symptom.** With 2 nodes × 4 GPUs, the sglang engine on the head node tried to set `cuda:5` and crashed with `invalid device ordinal`. Placement-group log showed gh069's GPUs sorted as `[2, 0, 1, 3]` instead of `[0, 1, 2, 3]`.

**Diagnosis.** In `slime/ray/placement_group.py:sort_key`, the IP-parsing path falls back to `socket.gethostbyname` for hostname-style identifiers; for `"gh069"`, the resolution can return different IP strings under DNS pressure, breaking within-node ordering. `engine 0` then received `base_gpu_id = reordered_gpu_ids[0] = 2`, and `tp_size=4` mapped TP ranks 0-3 to physical GPUs 2,3,4,5. GPU 4 doesn't exist on a 4-GPU node.

**Fix.** Replace the IP-conversion path with a direct string-comparison sort:
```python
def sort_key(x):
    index, node_identifier, gpu_id = x
    return (str(node_identifier), int(gpu_id))
```
Groups bundles by raw node-identifier string and orders by `gpu_id` within each node. Deterministic.

Patch is in [`slime/ray/placement_group.py`](../../slime/ray/placement_group.py) on this branch.

---

## 5. slime: `--num-gpus-per-node` defaults to 8

**Symptom.** Two engines reported the same `host` and `dist_init_addr` even when their actors lived on different physical nodes. The TCP rendezvous timed out.

**Diagnosis.** Slime's CLI default for `--num-gpus-per-node` is 8 (8-GPU H100/H800 box). Our nodes have 4. With the default, `num_engines_per_node = 8 // 4 = 2`, so slime packed both engines onto a single node. They competed for the same 4 GPUs.

**Fix.** Pass **`--num-gpus-per-node 4`** through `SGLANG_ARGS` (the help string explicitly says to do this when you have <8 GPUs/node).

---

## 6. Cross-node Gloo binds to IPv6 link-local addresses

**Symptom.** Megatron's `MegatronTrainRayActor.init` hung for 10 minutes then died with `Gloo connectFullMesh … timed out connecting, remote=[fe80::240:a6ff:fe8f:f72d]:13750`.

**Diagnosis.** Without an explicit interface, PyTorch distributed picks the first non-loopback NIC, which on DeltaAI is an IPv6 link-local that doesn't route across nodes. Slingshot's `hsn0` is the right interface for cross-node traffic.

**Fix.** Export these three env vars (and propagate them in ray's `RUNTIME_ENV_JSON`):
```bash
export GLOO_SOCKET_IFNAME=hsn0
export TP_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_IFNAME=hsn0
```

---

## 7. flashinfer JIT cache fails on Lustre

**Symptom.** sglang scheduler crashed mid-init with `OSError: [Errno 37] No locks available`. Manifested only on 4-node runs; 2-node was fine.

**Diagnosis.** `flashinfer` JIT-compiles attention kernels on first use and uses `fcntl.flock` on the cache directory at `~/.cache/flashinfer/`. The home filesystem is Lustre, which does support flock but with limited concurrency. With 4 nodes hammering the same cache file simultaneously, locks fail.

**Fix.** Per-node local-tmp cache:
```bash
export FLASHINFER_WORKSPACE_BASE=/tmp/${USER}/flashinfer
export TRITON_CACHE_DIR=/tmp/${USER}/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/${USER}/torchinductor_cache
```
Also pre-create these directories on each compute node before launch:
```bash
srun --jobid=$JOBID --overlap --ntasks=$NNODES --ntasks-per-node=1 \
  bash -c "mkdir -p /tmp/${USER}/{flashinfer,triton_cache,torchinductor_cache}"
```
Each node compiles its own kernels, but there's no contention. Add the three env vars to `RUNTIME_ENV_JSON` so ray actors inherit them.

---

## 8. `bash -c "…${MODEL_ARGS[@]}…"` mangles bash arrays containing brackets

**Symptom.** The first sbatch attempt at the convert step crashed with `TypeError: '<=' not supported between instances of 'int' and 'NoneType'` — `args.num_layers` was `None`. Strings dumped under `set -ex` showed `--disable-bias-linear'` (extra quote) and `--moe-layer-freq '[1,1,…]'` re-quoted as part of a single blob containing the rest of the arch flags, so argparse only saw the first flag.

**Diagnosis.** Wrapping `${MODEL_ARGS[@]}` inside `bash -c "…"` causes the outer shell to expand the array and re-quote elements that look like glob patterns. The single quotes around `--moe-layer-freq '[1,1,…]'` swallowed everything between them.

**Fix.** Drop the `bash -c` indirection — the sbatch script already runs on the head node, just invoke `torchrun` directly so the parent shell expands `MODEL_ARGS` cleanly.

---

## 9. `Qwen3-30B-A3B-Thinking-2507` config has `rope_theta=10000000`, not `1000000`

**Symptom.** Train startup failed with `AssertionError: rope_theta in hf config 10000000 is not equal to rotary_base 1000000, please check the config.`

**Diagnosis.** Slime's `scripts/models/qwen3-30B-A3B.sh` defaults `rotary-base` to 1M (the base model's value). Thinking-2507 uses 10M.

**Fix.** The model-args file already has an env override hook: `--rotary-base "${MODEL_ARGS_ROTARY_BASE:-1000000}"`. In our launcher:
```bash
export MODEL_ARGS_ROTARY_BASE=10000000
source "${SCRIPT_DIR}/models/qwen3-30B-A3B.sh"
```

---

## 10. slime convert script silently overrides `--pipeline-model-parallel-size`

**Symptom.** Convert with `--tensor-model-parallel-size 4 --pipeline-model-parallel-size 1` on 4 GPUs failed with `AssertionError: world size (4) is not divisible by total_model_size (total_model_size=16)`.

**Diagnosis.** `tools/convert_hf_to_torch_dist.py` has an auto-override:
```python
if args.pipeline_model_parallel_size == 1 and world_size > 1:
    pp_size = world_size
    ...
```
…which silently sets `PP = world_size`. With our `TP=4` and the auto-set `PP=4`, `total_model_size = 4 * 4 = 16` but `world=4`.

**Fix.** Don't pass `--tensor-model-parallel-size` to the convert script. Let it default to `TP=1`, auto-set `PP=world_size`. The training side reshards on load.

---

## 11. Dropping `--ref-load` (because `--kl-loss-coef=0`) breaks slime's bootstrap path

**Symptom.** Without `--ref-load`, train startup failed with `AssertionError` in `setup_model_and_optimizer`, on `assert args.load is not None or args.pretrained_checkpoint is not None`.

**Diagnosis.** Slime's argument-preprocessing path: if `args.load` (the live-checkpoint dir) doesn't exist, slime falls back to `args.load = args.ref_load`. With `--ref-load` removed, that fallback set `args.load = None`, which then fails Megatron's check. The reference policy is used as the bootstrap weight source even when KL loss is disabled.

**Fix.** Always keep `--ref-load <torch_dist_path>`. The host RAM savings from not loading the ref aren't worth breaking the bootstrap flow.

---

## 12. THE BIG ONE: `--megatron-to-hf-mode bridge` produces a torch_dist that fails MoE-expert reshape on load

**Symptom.** After all the above fixes, training startup got as far as the Megatron actor loading the converted checkpoint, then emitted ~39,600 warnings of `decoder.layers.<N>.mlp.experts.experts.linear_fc2.weight from model not in state dict, will skip` followed by:
```
TypeError: object of type '_io.BytesIO' has no len()
  at megatron/core/dist_checkpointing/strategies/torch.py:439
     assert len(tensors) == len(rename_mapping[k])
```
Reproduced on both 2 nodes and 4 nodes — confirmed Thinking-2507-specific.

**Diagnosis.** The convert script in `bridge` mode serializes the MoE expert weights in a way that Megatron's `_replace_sharded_keys_with_state_dict_keys` can't iterate (`tensors` is a `_io.BytesIO` instead of `list[Tensor]`). The same script with `bridge` mode worked for the *base* `Qwen3-30B-A3B` — but failed for the Thinking variant despite identical architecture. We do not yet know whether the bug is in `mbridge` (the bridge backend), in slime's `slime_plugins.mbridge`, or in Megatron-LM's loader.

**Fix.** Drop `--megatron-to-hf-mode bridge` from the convert command — the parser's default is `"raw"`, which produces a checkpoint Megatron loads correctly. Concrete diff:
```diff
-torchrun --nproc_per_node 4 tools/convert_hf_to_torch_dist.py \
-  --hf-checkpoint $HF_DIR \
-  --save $TD_DIR \
-  --megatron-to-hf-mode bridge \
-  "${MODEL_ARGS[@]}"
+torchrun --nproc_per_node 4 tools/convert_hf_to_torch_dist.py \
+  --hf-checkpoint $HF_DIR \
+  --save $TD_DIR \
+  "${MODEL_ARGS[@]}"
```

The torch_dist directory is the same size (~57 GB) and contains the same number of `.distcp` files in either mode, but the state-dict layout differs.

**Open follow-up:** the slime gpt-oss-20B example explicitly uses `bridge` for MoE. That example may be broken for newer Megatron, or our `mbridge==0.15.1` is the wrong version, or Thinking-2507 has some metadata that trips bridge specifically. The collaborator should narrow this down.

---

## 14. `sgl_kernel.flash_attn` raises at module-import time when FA3 isn't built

**Symptom.** Trying to load `Qwen3.5-4B` (or any Qwen3.5 variant) failed with `ValueError: Qwen3_5ForConditionalGeneration has no SGlang implementation`, even though `sglang/srt/models/qwen3_5.py` is present and `EntryClass` lists the class. Earlier log entry showed why: `Ignore import error when loading sglang.srt.models.qwen3_5: Can not import FA3 in sgl_kernel.`

**Diagnosis.** `sgl_kernel/flash_attn.py` line 6-11 has a top-level `try: from sgl_kernel import flash_ops; except: raise ImportError("Can not import FA3 in sgl_kernel…")`. Our DeltaAI sgl-kernel build excludes FA3 (memory limit at build time, gated by `SGL_KERNEL_ENABLE_FA3=OFF`), so this raise fires. Anything that imports `sgl_kernel.flash_attn` cascades — including `sglang/srt/layers/attention/flashattention_backend.py`, which is transitively imported when sglang's `ModelRegistry` scans `sglang.srt.models.qwen3_5`. The registry's `try/except` then silently skips qwen3_5 and the model never gets registered.

**Fix.** Patch `flash_attn.py` to convert the top-level raise into a placeholder so the module imports cleanly. We use `--sglang-attention-backend flashinfer`, so FA3 functions are never actually invoked at runtime:
```python
try:
    from sgl_kernel import flash_ops
    _FA3_AVAILABLE = True
except Exception:
    flash_ops = None
    _FA3_AVAILABLE = False
```
Patch is applied to the live install at `/work/nvme/bgqz/bzhang31/envs/slime/lib/python3.12/site-packages/sgl_kernel/flash_attn.py`. After this, `from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration` succeeds and the architecture lookup hits the native sglang implementation instead of falling back to transformers.

**Validated.** Qwen3.5-4B GRPO smoke ran 3 full steps on alloc 2295059 (1 node × 4 GH200), no FA3-related errors. TFlops ramped from 23 (step 1 warmup) → 78 → 121 (steady state). Same patch is required for any Qwen3.5 variant (4B/9B/27B/35B-A3B).

---

## 13. Host-RAM accumulation across GRPO iterations (still open)

**Symptom.** Training runs successfully for 1-3 GRPO steps, then a Ray worker is killed with:
```
OutOfMemoryError: … Memory on the node was 815-825 GB / 858 GB (~95%), exceeds threshold 0.95
```

Per-attempt step budget before OOM:
| Config | Steps completed before OOM |
|---|---|
| 2 nodes, rollout-batch=16 | 1 |
| 4 nodes, rollout-batch=16 | 2 (+ rollout 3) |
| 4 nodes, rollout-batch=8 | (testing in A4) |

The trainer actors take ~83 GB each, so 4 actors per node baseline is 332 GB. Each iteration appears to leak ~50-80 GB of additional host RAM that doesn't release between rollouts. By step 2-3 we cross the 95% threshold and Ray kills a worker.

**Diagnosis.** Not yet root-caused. Likely candidates: (a) sglang's KV cache not fully releasing after `release_memory_occupation`, (b) slime's optimizer-cpu-offload pipeline accumulating intermediate buffers across `--overlap-cpu-optimizer-d2h-h2d` cycles, (c) Megatron dist_ckpt async-save buffers retained after the first save.

**Workarounds tried.**
- 4 nodes vs 2 nodes: doubles the effective host-RAM headroom; bought us 1 extra step.
- Reduce `--rollout-batch-size` 32 → 16 → 8: each halving cuts the per-step KV+activation buffer roughly proportionally. Currently testing 8.

**Real fix (not yet implemented).** Find which actor is leaking. Suggested first steps for the collaborator:
1. Add per-actor `psutil.Process().memory_info().rss` printouts at start-of-rollout, end-of-rollout, end-of-train. Compare across steps.
2. Force `gc.collect()` and `ctypes.CDLL("libc.so.6").malloc_trim(0)` between iterations in the train loop.
3. Try `enable_memory_saver=False` on the sglang engine (vs the current `True`) — current setting keeps weights on GPU but moves activations/KV to CPU under pressure, which may be what accumulates.

---

## What works today (recipe)

```
# Convert (don't pass TP, don't pass --megatron-to-hf-mode):
torchrun --nproc_per_node 4 tools/convert_hf_to_torch_dist.py \
  --hf-checkpoint .../Qwen3-30B-A3B-Thinking-2507 \
  --save        .../Qwen3-30B-A3B-Thinking-2507_torch_dist \
  "${MODEL_ARGS[@]}"   # with MODEL_ARGS_ROTARY_BASE=10000000

# Train (4 nodes × 4 GH200):
NUM_NODES=4, TP=4, EP=8, PP=1, CP=1, ETP=1
--rollout-batch-size 8 (or 16 if you only need 2 steps)
--global-batch-size 64
--rollout-max-response-len 32768
--max-tokens-per-gpu 32768

# Mandatory env:
GLOO_SOCKET_IFNAME=hsn0  TP_SOCKET_IFNAME=hsn0  NCCL_SOCKET_IFNAME=hsn0
FLASHINFER_WORKSPACE_BASE=/tmp/$USER/flashinfer
TRITON_CACHE_DIR=/tmp/$USER/triton_cache

# Mandatory slime args:
--num-gpus-per-node 4
--ref-load .../Qwen3-30B-A3B-Thinking-2507_torch_dist
```

GRPO step time on this config: rollout 240-370s + train 100-200s = **6-8 min/step**.
