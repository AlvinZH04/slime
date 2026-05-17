# slime + Qwen3-30B-A3B-Thinking-2507: `BytesIO has no len()` during torch_dist load on 4 nodes

## TL;DR

Training `Qwen/Qwen3-30B-A3B-Thinking-2507` with [THUDM/slime](https://github.com/THUDM/slime) fails when loading the `torch_dist` checkpoint at `MegatronTrainRayActor.init`. The error:

```
TypeError: object of type '_io.BytesIO' has no len()
```

…raised from `megatron/core/dist_checkpointing/strategies/torch.py:439` inside `_replace_sharded_keys_with_state_dict_keys`. Just before this, the log emits **39,600** lines of `decoder.layers.<N>.mlp.experts.experts.linear_fc2.weight from model not in state dict, will skip` (one per MoE expert weight × per rank).

## What does work — for context

The *base* model `Qwen/Qwen3-30B-A3B` (same `model_type=qwen3_moe`, same 48 layers / 128 experts) loads and trains successfully on this stack on **2 nodes × 4 GH200**. So the slime + Megatron-LM combination is broadly working; the failure is specific to the *Thinking-2507* variant **and/or** to the 4-node configuration we now want to use.

## Configurations

### Convert (`tools/convert_hf_to_torch_dist.py`)

```
torchrun --nproc_per_node 4 tools/convert_hf_to_torch_dist.py \
  --hf-checkpoint .../Qwen3-30B-A3B-Thinking-2507 \
  --save .../Qwen3-30B-A3B-Thinking-2507_torch_dist \
  --megatron-to-hf-mode bridge \
  "${MODEL_ARGS[@]}"
```

where `MODEL_ARGS` is the upstream slime [`scripts/models/qwen3-30B-A3B.sh`](scripts/model_args.sh), but with `--rotary-base 10000000` (matching Thinking-2507's `rope_theta`).

The convert script auto-sets `pipeline_model_parallel_size = world_size = 4`, prints `Using pipeline model parallel size: 4, decoder last pipeline num layers: 12`, then successfully saves the checkpoint (~57 GB):

```
successfully saved checkpoint from iteration       1 to .../Qwen3-30B-A3B-Thinking-2507_torch_dist [ t 1/1, p 1/4 ]
```

### Train (`train.py` + 4 nodes × 4 GPU = 16 GPUs)

```
python3 train.py \
  --actor-num-nodes 4 --actor-num-gpus-per-node 4 --colocate \
  --tensor-model-parallel-size 4 \
  --pipeline-model-parallel-size 1 \
  --expert-model-parallel-size 8 \
  --expert-tensor-parallel-size 1 \
  --hf-checkpoint .../Qwen3-30B-A3B-Thinking-2507 \
  --ref-load .../Qwen3-30B-A3B-Thinking-2507_torch_dist \
  --load .../checkpoints_live/Qwen3-30B-A3B-Thinking-2507_slime/ \
  --save .../checkpoints_live/Qwen3-30B-A3B-Thinking-2507_slime/ \
  ${MODEL_ARGS[@]} \
  ...
```

Full launch script: [scripts/run_train.sh](scripts/run_train.sh).

`--load` directory doesn't exist (first run), so slime falls through to `args.load = args.ref_load`, i.e. it loads the `torch_dist` we just made. That's where the error is raised.

The conversion is done at `TP=1, PP=4, EP=1` (script auto-sets PP=world_size), the train expects `TP=4, PP=1, EP=8`. Megatron's distributed-checkpoint reshape should handle this; for the *base* model it does. For Thinking-2507 it does not.

## Reproduction

1. Clone slime at the pinned commit and apply our DeltaAI patches (sort_key, socket NICs, num-gpus-per-node — see [`env.md`](env.md) for the slime HEAD).
2. Download Qwen3-30B-A3B-Thinking-2507 (~62 GB HF).
3. Run the convert command in [scripts/run_train.sh](scripts/run_train.sh) (or equivalently use the `sbatch` wrapper); confirm it produces a 57 GB `torch_dist/` directory.
4. Launch training across 4 nodes × 4 GPUs with the args above.
5. Observe the `MegatronTrainRayActor.init` failure with `BytesIO has no len()` after the cluster of "from model not in state dict, will skip" warnings.

## Artifacts in this repo

| File | Contents |
|------|----------|
| [README.md](README.md) | This file |
| [env.md](env.md) | Exact `torch`/`transformers`/`sglang`/`flashinfer`/slime/Megatron-LM HEADs |
| [config.json](config.json) | The HF config of Qwen3-30B-A3B-Thinking-2507 as we used it |
| [scripts/run_train.sh](scripts/run_train.sh) | Our exact training launch script |
| [scripts/model_args.sh](scripts/model_args.sh) | Sourced `MODEL_ARGS` array (matches upstream `scripts/models/qwen3-30B-A3B.sh`) |
| [logs/convert_success.log](logs/convert_success.log) | Successful convert (key lines) + listing of saved `torch_dist/` |
| [logs/error_traceback.log](logs/error_traceback.log) | The actual training-time traceback |
| [logs/missing_keys_count.txt](logs/missing_keys_count.txt) | Count of `from model not in state dict, will skip` warnings emitted before the error |

## What we have tried

- Convert with `--megatron-to-hf-mode bridge` and consistent `--rotary-base 10000000` (Thinking-2507's `rope_theta`). Result: torch_dist saves, training still fails with `BytesIO`.
- Convert with `--rotary-base 1000000` (the slime default in `scripts/models/qwen3-30B-A3B.sh`). Result: same train-time error.
- Dropping `--ref-load` (since `--kl-loss-coef=0`). Result: slime overrides `args.load = args.ref_load = None`, which then fails Megatron's `assert args.load is not None or args.pretrained_checkpoint is not None`. So `--ref-load` must remain set.
- Restoring `--ref-load`: training proceeds to `setup_model_and_optimizer` and then hits the `BytesIO` error.

## Hypotheses

1. **MoE-EP reshape bug** in `_replace_sharded_keys_with_state_dict_keys` when converting a torch_dist saved at `EP=1` into a training run at `EP=8`. For the base model the same reshape works; maybe Thinking-2507's expert weights are serialized differently (Megatron stores them as `_io.BytesIO` for some reason instead of as `list[Tensor]`).
2. **Version mismatch** between the Megatron-LM that wrote the checkpoint and the Megatron-LM (or its slime-patched dist_ckpt strategy) that's loading it.
3. **`--megatron-to-hf-mode bridge` produces an incompatible state-dict for Thinking-2507** (some renamed key path triggers the `BytesIO` branch in the strategy). Whether `--megatron-to-hf-mode raw` works is unverified — that path was not tried because slime's gpt-oss-20B convert example uses `bridge`.

## Open question for the collaborator

In `megatron/core/dist_checkpointing/strategies/torch.py:439`, why is `tensors` ever a `_io.BytesIO`? It looks like sharded-tensor metadata for MoE experts gets routed through an `ObjectStorageMetadata`-style path that returns BytesIO instead of `list[Tensor]`. Is this expected on certain checkpoint layouts and the loader code should special-case it? Or is the checkpoint being mis-serialized?
