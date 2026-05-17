"""Build a slime-compatible JSONL prompt dataset from harness-evolve runs.

Each row has:
    prompt: str  — chat-template-ready proposer prompt (heuristic context bundle
                   + SEARCH/REPLACE instructions)
    label:  dict — JSON-encoded blob the reward function needs at scoring time:
                   { parent_id, task_family, task_config, baseline_score,
                     harness_repo_root, parent_harness_path }

Usage:
    python tools/build_harness_evolve_prompts.py \\
        --runs-root /projects/bgqz/bzhang31/harness-evolve/runs \\
        --harness-repo-root /projects/bgqz/bzhang31/harness-evolve \\
        --out /work/hdd/bgqz/bzhang31/datasets/harness_evolve_prompts.jsonl \\
        [--context-builder v1 | v2] [--min-baseline-score 0.0]
        [--include-task-family sudoku_easy_3x3 ...]

Slime ingestion side:
    --prompt-data <path-to-this-jsonl>
    --input-key prompt
    --label-key label
    --apply-chat-template       (model's chat template wraps the prompt string)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add harness-evolve to sys.path so we can import its context builders.
# Adjust --harness-repo-root if your checkout lives elsewhere.
def _add_harness_evolve_to_path(harness_repo_root: Path) -> None:
    if str(harness_repo_root) not in sys.path:
        sys.path.insert(0, str(harness_repo_root))


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Proposer system prompt (Aider-style SEARCH/REPLACE format).
# Pulled (with light adaptation) from
# harness_evolve.proposer.IMPLEMENTATION_SYSTEM_PROMPT_SEARCH_REPLACE so the
# trained model produces outputs the existing parser at
# slime_plugins/rewards/harness_evolve.py can score.
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_SEARCH_REPLACE = """\
You are an expert software engineer improving an evaluation harness.

Output your edits as one or more SEARCH/REPLACE blocks in this EXACT format:

<<<<<<< SEARCH
<exact text from parent harness>
=======
<replacement text>
>>>>>>> REPLACE

Rules:
- Each SEARCH block must match the parent harness EXACTLY (whitespace, casing,
  trailing punctuation). If the SEARCH text is not present byte-for-byte, the
  edit cannot be applied.
- Output ONLY the SEARCH/REPLACE blocks. No prose, no commentary, no markdown
  fences.
- Multiple blocks are applied in order. Each block must reference a different
  region.
"""


def _render_prompt(context_text: str, task_family: str, baseline_score: float,
                   parent_scores_text: str) -> str:
    """Concatenate the heuristic context bundle into a single prompt string.

    The string is later passed through the model's chat template via slime's
    `--apply-chat-template`, which will wrap it with the model's special
    user/assistant tokens.
    """
    return (
        f"{SYSTEM_PROMPT_SEARCH_REPLACE}\n\n"
        f"---\n\n"
        f"Task family: {task_family}\n"
        f"Parent baseline primary score: {baseline_score:.4f}\n\n"
        f"Parent scores breakdown:\n{parent_scores_text}\n\n"
        f"---\n\n"
        f"Context bundle (parent harness, sibling scores, sampled traces):\n\n"
        f"{context_text}\n\n"
        f"---\n\n"
        f"Now emit one or more SEARCH/REPLACE blocks that improve the harness."
    )


def _iter_parents_from_run_dir(run_dir: Path):
    """Walk a harness-evolve run_dir and yield (parent_id, parent_record) pairs.

    A `parent_record` is the dict slime needs for one prompt:
        { parent_id, task_family, task_config, baseline_score,
          parent_harness_path, parent_scores_text, parent_artifacts, ... }

    TODO: wire to the actual schema of harness-evolve's history_store /
    ExperimentIndex. The fields below match what `heuristic_context.py` and
    `heuristic_context_v2.py` expect on their request object.
    """
    raise NotImplementedError(
        "_iter_parents_from_run_dir: implement against harness-evolve's "
        "ExperimentIndex / history_store schema. See harness_evolve/context_index.py "
        "and harness_evolve/history_store.py."
    )


def build_one_row(
    parent_record: dict,
    harness_repo_root: Path,
    context_builder,  # HeuristicContextBuilder instance (v1 or v2)
    context_request_cls,  # ContextBuildRequest
    experiment_index,  # ExperimentIndex
    run_dir: Path,
) -> dict:
    request = context_request_cls(
        run_dir=run_dir,
        index=experiment_index,
        parent_id=parent_record["parent_id"],
        task_family=parent_record["task_family"],
        task_config=parent_record["task_config"],
        candidate_table=parent_record.get("candidate_table", ""),
        parent_artifacts=parent_record.get("parent_artifacts", {}),
        parent_scores=parent_record.get("parent_scores", {}),
        parent_failed_traces=parent_record.get("parent_failed_traces", []),
        iteration_metadata=parent_record.get("iteration_metadata", {}),
        instructions=parent_record.get("instructions", ""),
    )
    result = context_builder.build_context(request)

    # Render the materialized context bundle as a single string we can feed
    # to the model. `result.table` is the candidate table; `result.selected`
    # is the list of (path, body) tuples chosen by the builder.
    context_text_parts = [result.table] if result.table else []
    for path, body in result.selected:
        context_text_parts.append(f"### {path}\n```\n{body}\n```")
    if result.insight:
        context_text_parts.append(f"\n### Insight\n{result.insight}")
    context_text = "\n\n".join(context_text_parts)

    prompt = _render_prompt(
        context_text=context_text,
        task_family=parent_record["task_family"],
        baseline_score=float(parent_record.get("parent_scores", {}).get("score_primary", 0.0)),
        parent_scores_text=json.dumps(parent_record.get("parent_scores", {}), indent=2),
    )

    label = {
        "parent_id": parent_record["parent_id"],
        "task_family": parent_record["task_family"],
        "task_config": parent_record["task_config"],
        "baseline_score": float(parent_record.get("parent_scores", {}).get("score_primary", 0.0)),
        "harness_repo_root": str(harness_repo_root),
        "parent_harness_path": parent_record["parent_harness_path"],
    }

    return {"prompt": prompt, "label": label}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True,
                        help="Directory containing one or more harness-evolve run_dirs.")
    parser.add_argument("--harness-repo-root", type=Path, required=True,
                        help="Root of the harness-evolve repo (so we can resolve "
                             "parent_harness_path relative to it).")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output JSONL path.")
    parser.add_argument("--context-builder", choices=["v1", "v2"], default="v1",
                        help="Which heuristic context builder to use. v1 = more "
                             "comprehensive (5 rule-based strategies), v2 = "
                             "fixed-shape stripped-down. Per user (2026-05-17), "
                             "start with v1 for richer reward signal.")
    parser.add_argument("--include-task-family", nargs="*", default=None,
                        help="If set, only include rows whose task_family is "
                             "in this list.")
    parser.add_argument("--min-baseline-score", type=float, default=None,
                        help="Skip parents below this baseline (avoid wasting "
                             "rollouts on harnesses that are too far gone).")
    parser.add_argument("--max-baseline-score", type=float, default=None,
                        help="Skip parents above this baseline (avoid ceiling "
                             "cases with no room to improve).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    _add_harness_evolve_to_path(args.harness_repo_root)

    # Import here so the import error doesn't fire when this file is read by
    # tooling on a machine that doesn't have harness-evolve installed.
    from harness_evolve.context_builder import ContextBuildRequest
    from harness_evolve.context_index import ExperimentIndex
    if args.context_builder == "v1":
        from harness_evolve.heuristic_context import HeuristicContextBuilder
    else:
        from harness_evolve.heuristic_context_v2 import HeuristicContextBuilder

    builder = HeuristicContextBuilder()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = n_skipped = 0
    with open(args.out, "w") as f:
        for run_dir in sorted(p for p in args.runs_root.iterdir() if p.is_dir()):
            try:
                experiment_index = ExperimentIndex.load(run_dir)  # TODO: confirm API
            except Exception as e:
                logger.warning(f"skip {run_dir}: cannot load index: {e}")
                continue

            for parent_id, parent_record in _iter_parents_from_run_dir(run_dir):
                baseline = float(parent_record.get("parent_scores", {}).get("score_primary", 0.0))

                if args.include_task_family and parent_record["task_family"] not in args.include_task_family:
                    n_skipped += 1
                    continue
                if args.min_baseline_score is not None and baseline < args.min_baseline_score:
                    n_skipped += 1
                    continue
                if args.max_baseline_score is not None and baseline > args.max_baseline_score:
                    n_skipped += 1
                    continue

                row = build_one_row(
                    parent_record=parent_record,
                    harness_repo_root=args.harness_repo_root,
                    context_builder=builder,
                    context_request_cls=ContextBuildRequest,
                    experiment_index=experiment_index,
                    run_dir=run_dir,
                )
                f.write(json.dumps(row) + "\n")
                n_written += 1

    logger.info(f"wrote {n_written} rows, skipped {n_skipped} -> {args.out}")


if __name__ == "__main__":
    main()
