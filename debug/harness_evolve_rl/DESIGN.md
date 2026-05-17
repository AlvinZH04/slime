# harness-evolve × slime: RL Pipeline Design

**Status:** draft, 2026-05-17. Pre-implementation. Pairs with the parallel Qwen3.5-4B math RL smoke that's validating slime infra.

## Goal

Train an open-weight LLM (eventually Qwen3-30B-A3B-Thinking-2507; smaller models for iteration) via GRPO so that, given a fixed-shape context bundle describing a software test harness and its prior evaluation results, the model emits a *diff that measurably improves the harness's primary score on held-out tasks*.

This replaces the existing 3-stage explore→diagnose→implement proposer in `harness_evolve.proposer` with a single-shot proposer trained via RL — eliminating both the inference token waste and the training-surface bloat that `heuristic_context_v2.py:1-30` calls out.

## Context builder: v1 vs v2

The harness-evolve codebase ships two heuristic context builders:

| | `heuristic_context.py` (v1) | `heuristic_context_v2.py` (v2) |
|---|---|---|
| Selection | **5 rule-based strategies** (parent lineage, top-k by score, failed traces, best improvers, recent candidates) | Sample K failed + K successful traces from parent, + N sibling scores |
| Output shape | Variable — depends on what each strategy finds | Fixed shape per parent (seeded by `hash((parent_id, run_dir))`) |
| Designed for | **Comprehensive context** for the orchestrator pipeline | Stripped-down, deterministic, "training-surface-friendly" for SFT/RL |
| Use case | Production proposer (orchestrator) | Bootstrap distillation / RL-friendly canonical context |

**Per user (2026-05-17): v1 is more comprehensive.** For RL training we will start with **v1** because (a) richer context = stronger reward signal early when the policy is weak, (b) we lose nothing by training on v1 and *inferring* with v2 later if we decide the shaping argument matters.

The trade-off in `heuristic_context_v2.py:1-30` (training-surface bloat) only bites if the model has to learn *which strategy chose what*. v1 already commits to a fixed set of 5 strategies; the strategies themselves are deterministic, so the training surface is the same shape every time. v1's variable *output volume* is the real concern — we'll cap context length at slime's `--rollout-max-prompt-len` and the trained model will adapt.

## End-to-end loop (one GRPO iteration)

```
                  ┌─────────────────────────────────────────────┐
                  │  PROMPT DATASET (slime --prompt-data)       │
                  │  N rows; each row =                         │
                  │    prompt: heuristic_v2 context bundle      │
                  │           (parent harness + scores +        │
                  │            sampled traces + sibling scores) │
                  │    label: {parent_id, task_family,          │
                  │            baseline_score, task_config}     │
                  └─────────────────────────────────────────────┘
                                       │
                                       ▼
              ┌─────────────────────────────────────────┐
              │  ROLLOUT (slime + sglang)               │
              │  k samples / prompt at temp 1.0         │
              │  each sample = model's diagnose + diff  │
              │  (SEARCH/REPLACE blocks)                │
              └─────────────────────────────────────────┘
                                       │
                                       ▼
              ┌─────────────────────────────────────────┐
              │  REWARD FUNCTION                        │
              │  (slime --custom-rm-path …)             │
              │  per sample, run in worker pool:        │
              │    1. parse SEARCH/REPLACE blocks       │
              │    2. apply diff → candidate harness    │
              │    3. exec evaluator on task_config →   │
              │       EvalResult.score_primary          │
              │    4. reward = new_score - baseline OR  │
              │       new_score (configurable)          │
              │  shaping: format penalty, exec penalty  │
              └─────────────────────────────────────────┘
                                       │
                                       ▼
              ┌─────────────────────────────────────────┐
              │  GRPO ADVANTAGE + POLICY UPDATE         │
              │  (slime + Megatron) — standard          │
              └─────────────────────────────────────────┘
```

## Concrete pieces to build

### A. Prompt dataset (`tools/build_harness_evolve_prompts.py`)

Reads from a directory of prior `run_dir/`s (`/projects/bgqz/bzhang31/harness-evolve/runs/...`) and emits a slime-style JSONL:

```json
{
  "prompt":  "<chat-template-formatted heuristic_v2 bundle>",
  "label":   {
    "parent_id": "...",
    "task_family": "sudoku_easy_3x3",
    "task_config": {...},
    "baseline_score": 0.42,
    "harness_repo_root": "/work/hdd/.../harness-evolve",
    "parent_harness_path": "runs/.../harness.py"
  }
}
```

The prompt is built by calling `harness_evolve.heuristic_context.HeuristicContextBuilder().build_context(ContextBuildRequest(parent_id=..., run_dir=...))` and chat-template-wrapping its output with the proposer system prompt (request a diagnose + SEARCH/REPLACE diff). Reuses code from `harness_evolve.proposer`'s implement stage.

`label` is a JSON-encoded blob — slime supports arbitrary label payloads.

Expected dataset size: ~5–10 k prompts initially. Reusing the existing Reasoning-Gym SFT-split runs.

### B. Reward function (`slime_plugins/rewards/harness_evolve.py`)

A slime custom reward function — module-level `compute_reward(...)` that slime imports via `--custom-rm-path`:

```python
def compute_reward(prompts, completions, labels, **kwargs):
    """
    prompts:     list[str]   length = batch
    completions: list[str]   length = batch  (model's output)
    labels:      list[dict]  length = batch  (the JSON-encoded labels)
    Returns:     list[float] length = batch  (per-sample reward)
    """
    rewards = []
    with ProcessPoolExecutor(max_workers=NUM_EVAL_WORKERS) as pool:
        futures = [pool.submit(_score_one, c, l) for c, l in zip(completions, labels)]
        for f in futures:
            rewards.append(f.result())
    return rewards

def _score_one(completion: str, label: dict) -> float:
    # 1. Parse SEARCH/REPLACE blocks from completion.
    diff_ok, new_code = _apply_search_replace(label["parent_harness_path"], completion)
    if not diff_ok:
        return FORMAT_PENALTY        # e.g. -0.1

    # 2. Write candidate harness to a temp dir.
    work = _stage_candidate(label, new_code)

    # 3. Run the evaluator subprocess (harness_evolve.eval_interface).
    try:
        eval_result = _run_eval(work, label["task_config"], timeout_sec=300)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return EXEC_PENALTY          # e.g. -0.2

    # 4. Reward shaping.
    delta = eval_result.score_primary - label["baseline_score"]
    return max(-1.0, min(1.0, delta))  # clip to [-1, 1]
```

Worker pool runs evaluators in subprocesses so we get parallelism within one rollout batch. Worker count depends on the evaluator's resource footprint (CPU-bound for most reasoning-gym families). Per-eval timeout (e.g. 300 s) prevents a single slow harness from blocking the batch.

### C. Slime training script (`scripts/run-harness-evolve-rl-smoke.sh`)

Almost a copy of `run-qwen3.5-4B-deltaai-smoke.sh` but:

```bash
ROLLOUT_ARGS=(
  --prompt-data /work/hdd/bgqz/bzhang31/datasets/harness_evolve_prompts.jsonl
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rollout-shuffle
  --rm-type custom
  --custom-rm-path slime_plugins.rewards.harness_evolve.compute_reward
  --num-rollout 20
  --rollout-batch-size 8
  --n-samples-per-prompt 8
  --rollout-max-response-len 16384
  --rollout-temperature 1.0
  --global-batch-size 64
  --balance-data
)

EVAL_ARGS=(
  # No AIME — evaluate on a held-out harness slice
  --eval-interval 5
  --eval-prompt-data harness_eval /work/hdd/bgqz/bzhang31/datasets/harness_evolve_eval.jsonl
  --n-samples-per-eval-prompt 8
  --eval-max-response-len 16384
  --eval-temperature 0.6
  --eval-top-p 0.95
)
```

Everything else (TP, EP, optimizer, slime patches) carries over from the Qwen3.5-4B smoke.

## Open design questions

1. **What's the prompt format?** Aider-style fenced SEARCH/REPLACE blocks vs. unified-diff vs. full-file rewrites. The existing `harness_evolve.proposer.implement` stage uses *something* — should mirror it so the trained model is drop-in compatible at inference time.

2. **Reward shape: delta vs. absolute.**
   - `delta` = new − baseline: zero-centered, low variance early, but the *sign* of the gradient depends on a noisy baseline estimate.
   - `absolute` = new_score: easy to interpret, but slows convergence because the policy gets credit for things it didn't change.
   The standard GRPO mean-subtraction over k samples partly absorbs this — start with absolute, revisit if reward variance is too low.

3. **Off-policy vs. on-policy harness evaluation.** Running the evaluator on every completion is expensive (some reasoning-gym families take 30–60 s per harness). Two compromises:
   - Cache by `harness_hash`: if the model emits the same diff twice, score it once. The HistoryStore already does this for the orchestrator path; we can reuse it.
   - Trial-budget shaping: run a cheap k=1 verification first, only run full k=K if it doesn't crash.

4. **How many task_families per training mini-batch?** If a batch is all sudoku, the policy specializes on sudoku. Mix families per rollout to encourage transfer. Need a `mix_strategy` in the dataset builder.

5. **Reasoning-gym SFT-split as the bootstrap dataset** — confirmed in earlier session. Need to flatten it to slime JSONL format.

## Why not just SFT first?

The user explicitly chose RL-first (2026-05-17 conversation): "My original goal is to do SFT first and then RL, but I want to first try out RL so we know if we even need SFT at all." Rationale: if the base model is already good enough to occasionally produce a working diff, RL can exploit that without burning data on demonstrations. Worst case: RL diverges (reward variance too high, format collapse), we add SFT later as a warm-start.

## Tiny POC (next concrete deliverable)

Before scaling, validate the wiring end-to-end:

1. Single prompt (one parent_id from an existing reasoning-gym run).
2. Single task_family (sudoku_easy_3x3 — cheapest evaluator).
3. k = 4 samples / prompt.
4. 3 GRPO steps.
5. Qwen3-4B (smaller, single-node) or Qwen3.5-4B once the smoke validates the infra.
6. Inspect:
   - Are the completions parseable as SEARCH/REPLACE? (format penalty rate)
   - Do any completions actually compile + run the evaluator? (exec-success rate)
   - Is the reward variance non-zero across samples? (training signal)
   - Does the loss decrease across the 3 steps? (sanity)

This is a few hours of work on a single node, no GH-h burned on the 30B path.

## Dependencies on the 30B path

None blocking. The harness-evolve RL phase can proceed even while the 30B host-RAM leak is unresolved — we use a 4B model for the POC and only scale up after the small-model loop is closing on real reward signal. The 30B work and the harness-evolve work are deliberately decoupled.

## File layout for the implementation

```
slime/
├── slime_plugins/
│   └── rewards/
│       └── harness_evolve.py          # B. reward function
├── tools/
│   └── build_harness_evolve_prompts.py # A. prompt builder
└── scripts/
    └── run-harness-evolve-rl-smoke.sh # C. training script
```

All three exist as **stubs** to write. Order: A (dataset) → B (reward) → C (training).
