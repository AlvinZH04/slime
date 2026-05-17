# Slime on DeltaAI — Operations Cheat Sheet

User: `bzhang31`, account `bgqz-dtai-gh`. Goal: RL post-training of Qwen3-30B-A3B / Qwen3.6-35B-A3B with slime; this doc covers cluster mechanics, not slime internals.

Companion docs: `/u/bzhang31/slime_plan.md` (install recipe), `/u/bzhang31/deltaai_jobs_reference.md` (cluster reference), `CLAUDE.md` (session handoff state).

---

## Quick start (activate the env)

From any login or compute node where you have an allocation:

```bash
module reset
module load python/miniforge3_pytorch/2.11.0 cuda/12.9.0 cudnn/9.3.0.75 nccl-ofi-plugin/1.18.0-cuda129
source /sw/user/python/miniforge3-pytorch-2.11.0/etc/profile.d/conda.sh
conda activate base
unset PYTHONNOUSERSITE
source /work/nvme/bgqz/bzhang31/envs/slime/bin/activate
```

That brings up `(slime) (base)` with: torch 2.9.1+cu129, sglang 0.5.11 (editable, v0.5.9 source + slime patches), sgl-kernel 0.3.21 (sm_90, FA3 disabled), flash-attn 2.7.4.post1, TransformerEngine 2.10.0, apex (cuda_ext), Megatron-LM (pinned commit + slime patches), transformers 5.8.1.

CUDA_HOME, CUDNN_HOME, CPATH/LD_LIBRARY_PATH, PYTHONPATH for Megatron-LM, and HF/wandb caches are all pre-baked into the venv activate.

Smoke test (verify on a GPU node):
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expect: True NVIDIA GH200 120GB
```

---

## Checking jobs and nodes

```bash
# what I have running/queued
squeue -u $USER -o "%.10i %.13P %.18j %.8T %.5D %.10M %.10L %R"

# detail (incl. nodelist, alloc TRES, partition, dependencies)
scontrol show job <JOBID>

# live resource usage of a running job
sstat -j <JOBID> --format=JobID,MaxRSS,AveCPU

# job history (resolves: did the build OOM? hit time? exit how?)
sacct -u $USER --starttime YYYY-MM-DD --format=JobID,JobName,State,Elapsed,ExitCode,AllocTRES%50

# partition state (idle/mixed/draining counts)
sinfo -p ghx4 -o "%P %a %D %T %C %m"
sinfo -p ghx4-interactive -o "%P %a %D %T %C %m"

# my account / QOS
sacctmgr show assoc user=$USER format=Account,Partition,QOS%30
```

---

## Attaching to a running allocation

```bash
# slurm-native (gets cgroup + GPU binding)
srun --jobid=<JOBID> --overlap --pty bash -l

# direct ssh (lighter, no cgroup binding)
ssh <nodename>     # e.g. ssh gh014
```

`--overlap` lets multiple shells share the single task slot — required if Claude Code is already running srun on the alloc.

**Important — `nvidia-smi` from plain `ssh` won't see your processes.** Direct `ssh gh<NNN>` puts you in a session OUTSIDE the SLURM cgroup namespace, so `nvidia-smi` shows "No running processes found" even when training is actively using all 4 GPUs (memory column still reflects real usage, but the process list is empty). To see processes:
```bash
srun --jobid=<JOBID> --overlap --pty bash -l
nvidia-smi             # now lists your sglang/python PIDs
# or live-track GPU usage as model loads:
watch -n 2 nvidia-smi
```

---

## Allocations

**Interactive / debug builds** (8 node-h budget on this account):
```bash
salloc --no-shell -A bgqz-dtai-gh -p ghx4-interactive \
       --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 \
       --gpus=1 --mem=200g --time=02:00:00 \
       --constraint="projects&work" \
       --job-name=<name>
# returns: salloc: Granted job allocation <JOBID>
# 2h max wall on ghx4-interactive
# billing: partial-node × 2x interactive multiplier
```

Use `--no-shell` so the parent doesn't get stuck in an interactive child; reach the node with `srun --jobid=... --overlap --pty bash`.

**Production runs** (counts against main SU, not 8 node-h debug budget):
```bash
sbatch --account=bgqz-dtai-gh --partition=ghx4 ...
# up to 48h wall, billed by the whole node × hours
```

Release early:
```bash
scancel <JOBID>
```

**Constraint note:** the docs reference `projects&worknvme` and `workhdd`; the live cluster only exposes `work` (single feature mounting `/work/nvme` and `/work/hdd` together). Use `--constraint="projects&work"`.

**Module note:** `cuda/12.9.0` is deprecated; loads `cudatoolkit/25.5_12.9` (NVHPC-bundled). For source builds we override `CUDA_HOME=/sw/user/cudatoolkits/installs/cuda-12.9` (standalone toolkit) — the venv activate already does this.

---

## Storage layout (canonical paths)

| Path | Size / Quota | Use for |
|---|---|---|
| `/u/bzhang31` | 100 GB | HOME — code, configs, this README, `wandb_api.txt` |
| `/projects/bgqz/bzhang31/` | 500 GB shared (allocation-wide) | slime source, kept-checkpoints |
| `/work/nvme/bgqz/bzhang31/` | 1 TB shared, fast NVMe | venv, hot training I/O, models, live checkpoints |
| `/work/hdd/bgqz/bzhang31/` | 1 TB shared, bulk | datasets, wandb cache, logs |

Quota check: `lfs quota -uh $USER /work` (per-user) and `quota -s` (also shows allocation quotas under `delta_bgqz`).

Current allocation usage as of build completion: `/projects/bgqz` ~440 GB / 500 GB soft (tight). `/work/nvme/bgqz` ~530 GB after the build + Qwen3.5-9B + torch_dist. 35B-A3B will add ~70 GB HF + 70 GB torch_dist + rotating checkpoints.

---

## Workflow paths

**Smoke / dev runs**
- 1 node ghx4 (4 × GH200), 4-GPU dense, ~30-60 min per ~10 GRPO steps
- Scripts: `scripts/sbatch-qwen3.5-9B-smoke.sh` → calls `scripts/run-qwen3.5-9B-deltaai-smoke.sh` → sources `scripts/models/qwen3.5-9B.sh`
- Output: `/work/hdd/bgqz/bzhang31/logs/smoke_<JOBID>.log`, plus wandb (project `slime-deltaai-smoke`)

**Real run targets** (for 30B / 35B compute estimation)
- Qwen3-30B-A3B: slime ships `scripts/run-qwen3-30B-A3B.sh` and `scripts/models/qwen3-30B-A3B.sh` natively. Plan: 2 nodes (8 GH200), EP=8 / TP=1, ~25 s/step expected → ~3000 steps in ~21 h on 2 nodes.
- Qwen3.6-35B-A3B: not shipped with slime. Architecture is also `qwen3_*` hybrid linear/full attention + MoE, multimodal config (similar to qwen3.5-9B). Will need a `slime_plugins.models.qwen3_6` plugin and a `scripts/models/qwen3.6-35B-A3B.sh` model-args file; the qwen3_5 plugin is the closest template.

The smoke gives measured `train/sec_per_step`, `rollout/tokens_per_sec`, `train/tokens_per_sec` via wandb — feed those into `slime_plan.md` Part 3 extrapolation (≈2× factor 9B-dense → 3B-active) to refine cost.

---

## Wandb

API key at `/u/bzhang31/wandb_api.txt` (mode 0600). Never `cat` it into chat output. In sbatch scripts:
```bash
export WANDB_API_KEY="$(cat /u/bzhang31/wandb_api.txt)"
```

Cache and run dirs go to `/work/hdd/bgqz/bzhang31/{wandb,wandb_cache}` (set by venv activate).

If the cluster is firewalled from wandb.ai (test this), fall back to `WANDB_MODE=offline` and `wandb sync` from a network-enabled node.

---

## Gotchas we hit during install (so you don't have to)

1. **sglang pins `torch==2.9.1`** at the v0.5.9 commit. `pip install -e python[all]` silently pulls a **CPU-only** aarch64 torch wheel from PyPI. Recovery: `pip install torch==2.9.1+cu129 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129`.

2. **`cuda/12.9.0` module is deprecated** and points at NVHPC's bundled CUDA; CMake `find_package(CUDA)` doesn't see it. Set `CUDA_HOME=/sw/user/cudatoolkits/installs/cuda-12.9` explicitly (now in venv activate).

3. **cmake 4.x rejects old dlpack** (`cmake_minimum_required < 3.5`). Pin `cmake>=3.27,<4`. Already installed in venv.

4. **sgl-kernel CMakeLists adds 5 CUDA arches on aarch64+cu12.9** (sm_90, sm_90a, sm_100a, sm_120a, sm_101a). Memory-blows-up at MAX_JOBS=8. Patched locally to gate sm_100/120/101 on `SGL_KERNEL_ENABLE_SM100A=ON` and FA3/sm_90a on `SGL_KERNEL_ENABLE_FA3=ON` (both default OFF). FA3 is disabled in our build — slime falls back to FA2 hopper kernels at runtime.

5. **`cudnn/9.3.0.75` module sets `INCLUDE` not `CPATH`** — TE 2.10 build fails with `cudnn.h: not found`. Already fixed in venv activate (`CPATH=$CUDNN_HOME/include:$CPATH`).

6. **Bundled TE 2.13 in base conda is built against cu130** — fails to import once we install cu129 torch (`libcublas.so.13` missing). We use TE 2.10 from source instead; `transformer_engine_cu12` 2.10 ships as a 286 MB pre-built aarch64 wheel, only the small `transformer_engine_torch` binding builds from source (~5 min).

7. **Base conda's apex is cu130-linked** — `amp_C` fails. Force-rebuild apex in venv: `pip install --force-reinstall --no-deps -v --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 4" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4`.

8. **Qwen3.5-9B requires `transformers>=5.x`** (`model_type: qwen3_5` was added in v5). We upgraded to 5.8.1. sglang complains about its pinned `transformers==4.57.1` but still works at runtime.

9. **DAPO-Math-17k now ships as parquet**, not jsonl as the slime READMEs reference. Update `--prompt-data` accordingly.

10. **NCSA-recommended `HF_HUB_ENABLE_HF_TRANSFER=1`** fails because base conda's `huggingface_hub` is shadowed but `hf_transfer` is only in our venv — mismatch. Just don't set the flag; standard download is fast enough for 18 GB models.

---

## Files added under `scripts/` (local, not upstream)

These are excluded via `.git/info/exclude`:
- `scripts/models/qwen3.5-9B.sh` — derived from `qwen3.5-27B.sh` + Qwen3.5-9B HF `text_config`
- `scripts/run-qwen3-4B-deltaai-smoke.sh` — single-node smoke run, working
- `scripts/sbatch-qwen3-4B-smoke.sh` — sbatch wrapper for 4B
- `scripts/run-qwen3-30B-A3B-deltaai.sh` — multi-node 30B-A3B run
- `scripts/sbatch-qwen3-30B-A3B.sh` — sbatch wrapper for 30B-A3B
- `README_DELTAAI.md` — this file

---

## Starting a training run (the runbook)

### Step 1: Get an allocation

**Interactive (preferred for debug, faster to land):**
```bash
# 1-node smoke (4B)
salloc --no-shell -A bgqz-dtai-gh -p ghx4-interactive \
       --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 \
       --gpus-per-node=4 --gpu-bind=none --mem=0 --exclusive \
       --time=02:00:00 --constraint="projects&work" \
       --job-name=slime-4B

# 2-node real (30B-A3B)
salloc --no-shell -A bgqz-dtai-gh -p ghx4-interactive \
       --nodes=2 --ntasks-per-node=1 --cpus-per-task=64 \
       --gpus-per-node=4 --gpu-bind=none --mem=0 --exclusive \
       --time=02:00:00 --constraint="projects&work" \
       --job-name=slime-30B-int
```

`ghx4-interactive` has `OverSubscribe=NO` — every node is single-tenant, so you **must** use `--exclusive`. Non-exclusive requests are rejected. Wait depends on how fast nodes drain; "Resources" reason means SLURM already earmarked nodes (visible in `scontrol show job <jobid>` → `SchedNodeList=...`), it's just waiting for them to clear.

`--gpu-bind=none` is required for `--ntasks-per-node=1`. Default (`--gpu-bind=verbose,closest`) gives the single task only the "closest" GPU, then ray fails with `CUDA_VISIBLE_DEVICES contains ['0']`.

**Production (longer runs):**
```bash
sbatch /projects/bgqz/bzhang31/slime/scripts/sbatch-qwen3-30B-A3B.sh
```

### Step 2: Launch on the allocation

For an **sbatch** job, the wrapper handles everything. For an **interactive salloc**, run from a login node (NOT inside the alloc):

```bash
LOG=/work/hdd/bgqz/bzhang31/logs/<run>_$(date +%s).log
JOBID=<your alloc jobid>
NODELIST=<comma-separated, e.g. gh[069,119]>
{
  module reset
  module load python/miniforge3_pytorch/2.11.0 cuda/12.9.0 cudnn/9.3.0.75 nccl-ofi-plugin/1.18.0-cuda129
  source /sw/user/python/miniforge3-pytorch-2.11.0/etc/profile.d/conda.sh
  conda activate base
  unset PYTHONNOUSERSITE
  source /work/nvme/bgqz/bzhang31/envs/slime/bin/activate
  export WANDB_API_KEY="$(cat /u/bzhang31/wandb_api.txt)"
  export MASTER_ADDR=$(scontrol show hostnames "$NODELIST" | head -n 1)
  # Force the right NIC for cross-node Gloo/NCCL (multi-node only)
  export GLOO_SOCKET_IFNAME=hsn0
  export TP_SOCKET_IFNAME=hsn0
  export NCCL_SOCKET_IFNAME=hsn0
  srun --jobid=$JOBID --overlap --ntasks=<N nodes> --ntasks-per-node=1 --kill-on-bad-exit=1 \
    bash /projects/bgqz/bzhang31/slime/scripts/run-<run-name>.sh
} > "$LOG" 2>&1 &
```

For 4B (single node) you don't need the socket-iface exports — Gloo's defaults work on one node.

### Step 3: Watch the log

```bash
tail -F $LOG | grep -E "Ports for engine|successfully loaded checkpoint|Timer update_weights end|Eval aime|elapsed_steps|rollout/|wandb.*View run|Traceback|invalid device|OOM|RuntimeError|FAILED|EXIT="
```

**Expected timeline for 30B-A3B on 2 nodes × 4 GH200 (measured 2026-05-16):**

| Wall time | Marker | Notes |
|---|---|---|
| 0:00 | `MASTER_ADDR=...` | Env exported |
| 0:30 | `Router launched at <IP>:<port>` | sglang router up |
| ~1:30 | `Ports for engine 0: {'host': ...}` and `engine 1: {'host': ...}` | **Hosts must differ** — one per physical node. Same host = `--num-gpus-per-node` bug |
| 2:00–6:00 | `server_args=ServerArgs(...)` per engine; `Capture decode cuda graph` | sglang loads weights and traces cuda graphs |
| 6:00–8:00 | `[Gloo] Rank 0 is connected to 7 peer ranks` | Megatron all-ranks rendezvous succeeds — confirms `*_SOCKET_IFNAME=hsn0` worked |
| ~8:00 | `successfully loaded checkpoint from .../_torch_dist` | Megatron actor loads the model and the ref policy (loaded twice) |
| ~9:00 | `Timer update_weights end (elapsed: ~20s)` | First Megatron→sglang weight sync done |
| ~9:30 | `Eval aime2026: 0/240` starts | Initial baseline eval (30 prompts × 8 samples). ~4 min at ~6K tok/s/node |
| ~13:00 | `Eval aime2026: 240/240` done; `rollout/` log lines | First training rollout starts |
| ~16:00 | First `elapsed_steps=1` | First GRPO step complete |

**Total: ~15-20 min from launch to first measured step.** Budget accordingly when picking interactive (2h cap) — that leaves ~100 min of actual training, plus eval re-runs at step 5/10/15/...

Watch the wandb run URL (`wandb: 🚀 View run at https://wandb.ai/...`) for live `train/sec_per_step`, `rollout/tokens_per_sec`, `eval/aime2026/acc`.

**Benign transient warnings to ignore (we hit all of these):**
- `Ignore import error when loading sglang.srt.models.step3_vl...: Can not import FA3` — our sgl-kernel is built without FA3; flashinfer fallback works
- `UserWarning: Environment variable SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK will be deprecated` — repeated once per TP rank, harmless
- `(raylet) ... Workers (tasks / actors) killed due to memory pressure (OOM)` — one ray worker dying does NOT necessarily kill the job; ray restarts the actor. Only act if the message repeats or the Megatron/sglang processes themselves die
- `(TP, PP) mismatch after resume ((4, 1) vs (1, 1) from checkpoint): RNG state will be ignored` — the torch_dist checkpoint was made at TP=1, we're loading at TP=4; weights re-shard correctly, only RNG state is ignored
- `UserWarning: barrier(): using the device under current context` — ignorable torch warning

### Step 4: Tear down

Interactive: `scancel <jobid>` (or it auto-expires at 2h).

Inter-run cleanup on a still-living alloc (e.g., to restart after a crash):
```bash
srun --jobid=$JOBID --overlap --ntasks=<N nodes> --ntasks-per-node=1 bash -c \
  "pkill -9 sglang 2>/dev/null; ray stop --force 2>/dev/null; pkill -9 ray 2>/dev/null; pkill -9 python 2>/dev/null; sleep 2"
```

---

## Slime patches required for DeltaAI multi-node (4 GPUs/node)

Upstream slime defaults assume 8 GPUs/node single-box. We hit 3 stacked bugs going to 2 nodes × 4 GPUs:

| Bug | Symptom | Fix |
|---|---|---|
| `--num-gpus-per-node` defaults to 8 | `num_engines_per_node = 8/4 = 2` → both engines packed on one node; sglang `nnodes` mis-computed → TP=8 attempted on a 4-GPU node | Add **`--num-gpus-per-node 4`** to SGLANG_ARGS in the run script (see help string in `slime/utils/arguments.py`) |
| `placement_group.sort_key` returns non-deterministic order for hostnames | Engine 0 gets `base_gpu_id=2`, TP=4 maps to GPUs 2,3,4,5; 4,5 don't exist on a 4-GPU node | Patched [`slime/ray/placement_group.py:sort_key`](slime/ray/placement_group.py) to sort by `(str(node_identifier), int(gpu_id))` directly |
| Gloo defaults to IPv6 link-local for cross-node init | `Gloo connectFullMesh ... remote=[fe80::...]` timeout | Export `GLOO_SOCKET_IFNAME=hsn0`, `NCCL_SOCKET_IFNAME=hsn0`, `TP_SOCKET_IFNAME=hsn0` (in the launcher AND in the ray `RUNTIME_ENV_JSON` `env_vars` block in the run script) |

`hsn0` is the Slingshot high-speed interface (172.28.81.x range). It routes between compute nodes and supports the high bandwidth NCCL wants. `bond0` (172.28.60.x) works too for control-plane Gloo but is 1500-MTU, so prefer hsn0 for everything.

For **single-node** runs (smoke 4B), only `--num-gpus-per-node 4` is strictly required — the other two don't manifest with one node.

### `--num-gpus-per-node` is mandatory anytime per-node GPUs ≠ 8

This includes the smoke 4B script. If you fork a new model script, copy the `--num-gpus-per-node ${NUM_GPUS_PER_NODE}` line in `SGLANG_ARGS` — without it slime mis-counts and engines collide.

### Model-parallel choice for 30B-A3B on 2 nodes (8 GH200)

```
--tensor-model-parallel-size 4    # within-node TP=4
--expert-model-parallel-size 8    # EP=8 spans both nodes
--expert-tensor-parallel-size 1   # expert weights not TP-sharded (each GPU holds full experts for its EP shard)
--pipeline-model-parallel-size 1
--context-parallel-size 1
```

This is the same as upstream `scripts/run-qwen3-30B-A3B.sh` for an 8×H100 box. EP=8 spans nodes via NCCL; rollout is per-node engines (DP=2 across nodes) which avoids cross-node TP for inference.
