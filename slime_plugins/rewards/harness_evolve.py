"""Slime custom reward function for the harness-evolve RL pipeline.

Wired via:  --rm-type custom --custom-rm-path slime_plugins.rewards.harness_evolve.compute_reward

Per-sample contract (slime calls one task per completion):
    args:    parsed CLI argparse Namespace
    sample:  slime Sample object with .prompt, .response, .label, .metadata
    returns: float reward in [-1, 1]

Sample.label is the JSON-encoded dict produced by tools/build_harness_evolve_prompts.py:
    {
      "parent_id": "...",
      "task_family": "sudoku_easy_3x3",
      "task_config": {...},
      "baseline_score": 0.42,
      "harness_repo_root": "/work/.../harness-evolve",
      "parent_harness_path": "runs/.../harness.py",
    }

Reward shape:
    parse-failure         -> FORMAT_PENALTY (-0.1)
    apply-failure         -> APPLY_PENALTY  (-0.15)
    evaluator-exec failure -> EXEC_PENALTY  (-0.2)
    timeout               -> TIMEOUT_PENALTY (-0.2)
    success               -> reward = clip(new_score - baseline_score, -1.0, 1.0)
                             or, if HARNESS_EVOLVE_REWARD_SHAPE=absolute, new_score

Tunables via env (read at import time):
    HARNESS_EVOLVE_EVAL_TIMEOUT_SEC   default 300
    HARNESS_EVOLVE_REWARD_SHAPE       default "delta" (or "absolute")
    HARNESS_EVOLVE_EVAL_PYTHON        default sys.executable
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants / tunables
# ──────────────────────────────────────────────────────────────────────────────

FORMAT_PENALTY = -0.10
APPLY_PENALTY = -0.15
EXEC_PENALTY = -0.20
TIMEOUT_PENALTY = -0.20

EVAL_TIMEOUT_SEC = int(os.environ.get("HARNESS_EVOLVE_EVAL_TIMEOUT_SEC", "300"))
REWARD_SHAPE = os.environ.get("HARNESS_EVOLVE_REWARD_SHAPE", "delta").lower()
EVAL_PYTHON = os.environ.get("HARNESS_EVOLVE_EVAL_PYTHON", sys.executable)

assert REWARD_SHAPE in ("delta", "absolute"), f"Unknown REWARD_SHAPE: {REWARD_SHAPE}"


# SEARCH/REPLACE parser — mirror of harness_evolve.proposer._SEARCH_REPLACE_BLOCK_RE.
# Pulled in locally so this module doesn't hard-depend on the harness-evolve
# package being importable from the rollout worker.
_SEARCH_REPLACE_BLOCK_RE = re.compile(
    r"<<<<<<<\s*SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>>\s*REPLACE",
    re.DOTALL,
)


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point (slime calls this)
# ──────────────────────────────────────────────────────────────────────────────

async def compute_reward(args, sample, **kwargs) -> float:
    """Slime's --custom-rm-path target. Returns a float in [-1, 1]."""
    label = _decode_label(sample.label)
    if label is None:
        logger.warning("harness_evolve: label is not a JSON dict — returning 0")
        return 0.0

    # Run the expensive eval in a thread (subprocess is sync I/O) so we don't
    # block the rollout's asyncio loop.
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _score_one_sync, sample.response, label)


# ──────────────────────────────────────────────────────────────────────────────
# Sync inner logic — lives in a thread
# ──────────────────────────────────────────────────────────────────────────────

def _score_one_sync(response: str, label: dict) -> float:
    # 1. Parse the response for SEARCH/REPLACE blocks.
    blocks = _parse_search_replace_blocks(response)
    if not blocks:
        return FORMAT_PENALTY

    # 2. Apply the diff to the parent harness.
    parent_path = Path(label["harness_repo_root"]) / label["parent_harness_path"]
    try:
        parent_src = parent_path.read_text()
    except OSError as e:
        logger.warning(f"harness_evolve: cannot read parent harness {parent_path}: {e}")
        return EXEC_PENALTY

    new_src, apply_errors = _apply_search_replace_blocks(parent_src, blocks)
    if apply_errors:
        return APPLY_PENALTY

    # 3. Stage the candidate harness in a tmp dir, run the evaluator.
    try:
        score = _run_eval_subprocess(label, new_src)
    except subprocess.TimeoutExpired:
        return TIMEOUT_PENALTY
    except subprocess.CalledProcessError as e:
        logger.info(f"harness_evolve: evaluator exited non-zero ({e.returncode})")
        return EXEC_PENALTY
    except Exception as e:
        logger.warning(f"harness_evolve: evaluator unhandled exception: {e}")
        return EXEC_PENALTY

    # 4. Shape the reward.
    if REWARD_SHAPE == "delta":
        reward = score - float(label.get("baseline_score", 0.0))
    else:
        reward = score
    return max(-1.0, min(1.0, reward))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _decode_label(label):
    if isinstance(label, dict):
        return label
    if isinstance(label, str):
        try:
            return json.loads(label)
        except json.JSONDecodeError:
            return None
    return None


def _parse_search_replace_blocks(text: str) -> list[tuple[str, str]]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
    return [(m.group(1), m.group(2)) for m in _SEARCH_REPLACE_BLOCK_RE.finditer(cleaned)]


def _apply_search_replace_blocks(
    source: str, blocks: list[tuple[str, str]]
) -> tuple[str, list[str]]:
    """Apply blocks in order. Returns (new_source, errors)."""
    text = source
    errors: list[str] = []
    for i, (search, replace) in enumerate(blocks, 1):
        if search in text:
            text = text.replace(search, replace, 1)
            continue
        norm_search = re.sub(r"[ \t]+", " ", search).strip()
        norm_text = re.sub(r"[ \t]+", " ", text)
        if norm_search and norm_search in norm_text:
            errors.append(f"block {i}: whitespace-only match")
        else:
            errors.append(f"block {i}: SEARCH not found")
    return text, errors


def _run_eval_subprocess(label: dict, candidate_src: str) -> float:
    """Stage the candidate harness in a tmp dir and run the harness-evolve evaluator.

    Returns the float score from EvalResult.score_primary, or raises on failure.

    TODO: wire to the real evaluator CLI. For now this is the contract:
        - Take the candidate source.
        - Write to a tmp harness.py.
        - Invoke `python -m harness_evolve.eval ...` (or whatever entry point
          the harness-evolve team finalizes).
        - Parse the stdout JSON for score_primary.

    Until the CLI is wired, this raises NotImplementedError so the wiring is
    obvious in logs (and we don't silently train against zero reward).
    """
    raise NotImplementedError(
        "harness_evolve._run_eval_subprocess: wire to harness-evolve evaluator CLI. "
        "See debug/harness_evolve_rl/DESIGN.md section B."
    )

    # Reference implementation sketch (uncomment + adapt once CLI is finalized):
    # with tempfile.TemporaryDirectory(prefix="harness_eval_") as tmp:
    #     candidate_path = Path(tmp) / "harness.py"
    #     candidate_path.write_text(candidate_src)
    #     task_config_path = Path(tmp) / "task_config.json"
    #     task_config_path.write_text(json.dumps(label["task_config"]))
    #     result_path = Path(tmp) / "result.json"
    #     subprocess.run(
    #         [EVAL_PYTHON, "-m", "harness_evolve.eval",
    #          "--harness", str(candidate_path),
    #          "--task-family", label["task_family"],
    #          "--task-config", str(task_config_path),
    #          "--out", str(result_path)],
    #         cwd=label["harness_repo_root"],
    #         check=True,
    #         capture_output=True,
    #         timeout=EVAL_TIMEOUT_SEC,
    #     )
    #     with open(result_path) as f:
    #         eval_result = json.load(f)
    #     return float(eval_result["score_primary"])
