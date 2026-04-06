# experiments/phase1_replicate.py
# STATUS: Phase 1 — reproduce Stranieri & Stella (2023) 1P1W benchmarks
# DEPENDS ON: config.py, envs/scimai_env.py, policies/heuristics.py,
#             policies/agents.py, results/tables.py
# TEST: python -c "from experiments.phase1_replicate import run_phase1"

"""
Phase 1: Replicate the SCIMAI-Gym paper results.

Target benchmark: Stranieri & Stella (2023), 1P1W scenario, Exp 1.

Published results:
  (s,Q) = 1226 ± 71
  PPO   = 1213 ± 68
  A3C   =  870 ± 67  (we use A2C as substitute)
  Oracle= 1474 ± 45

Pipeline:
  1. Evaluate (s,Q) heuristic over 100 episodes
  2. Evaluate greedy oracle over 100 episodes
  3. Train PPO (300k steps) → evaluate over 100 episodes
  4. Train A2C (300k steps) → evaluate over 100 episodes
  5. Print comparison table
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.monitor import Monitor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    EVAL_CONFIG,
    MODELS_DIR,
    PAPER_RESULTS,
    PPO_CONFIG,
    A2C_CONFIG,
    SEED,
    TRAIN_CONFIG,
)
from envs.scimai_env import ScimaiEnv
from policies.agents import (
    create_a2c_agent,
    create_ppo_agent,
    evaluate_agent,
    load_agent,
    train_agent,
)
from policies.heuristics import OraclePolicy, SQPolicy

logger = logging.getLogger(__name__)


def run_phase1(
    total_timesteps: int | None = None,
    eval_only: bool = False,
    quick_test: bool = False,
) -> dict[str, Any]:
    """
    Run the full Phase 1 replication experiment.

    Args:
        total_timesteps: Override training budget.
        eval_only: If True, skip training and load saved models.
        quick_test: If True, use reduced timesteps for smoke testing.

    Returns:
        Dictionary with all results for table generation.
    """
    if quick_test:
        timesteps = TRAIN_CONFIG["quick_test_timesteps"]
    else:
        timesteps = total_timesteps or TRAIN_CONFIG["total_timesteps"]

    n_eval = EVAL_CONFIG["n_eval_episodes"]

    print("\n" + "=" * 70)
    print("  PHASE 1: REPLICATION — Stranieri & Stella (2023)")
    print(f"  Training budget: {timesteps:,} timesteps | Eval episodes: {n_eval}")
    print("=" * 70)

    results: dict[str, Any] = {}

    # ── 1. Evaluate (s,Q) Heuristic ──────────────────────────────────────
    from config import SQ_CONFIG
    print(f"\n▸ Evaluating (s,Q) heuristic (s={SQ_CONFIG['s']}, Q={SQ_CONFIG['Q']})...")
    env_sq = ScimaiEnv()
    sq_policy = SQPolicy()
    results["sq"] = evaluate_agent(
        sq_policy, env_sq, n_episodes=n_eval, is_heuristic=True
    )
    print(
        f"  (s,Q) profit: {results['sq']['mean_profit']:.1f} "
        f"± {results['sq']['std_profit']:.1f}"
    )

    # ── 2. Evaluate Oracle ────────────────────────────────────────────────
    print("\n▸ Evaluating greedy oracle (perfect demand foresight)...")
    env_oracle = ScimaiEnv()
    oracle = OraclePolicy(seed=SEED)
    results["oracle"] = evaluate_agent(
        oracle, env_oracle, n_episodes=n_eval, is_heuristic=True
    )
    print(
        f"  Oracle profit: {results['oracle']['mean_profit']:.1f} "
        f"± {results['oracle']['std_profit']:.1f}"
    )

    # ── 3. Train + Evaluate PPO ──────────────────────────────────────────
    print(f"\n▸ {'Loading' if eval_only else 'Training'} PPO agent...")
    env_ppo = Monitor(ScimaiEnv())
    ppo_model_path = MODELS_DIR / "phase1_ppo_final"

    if eval_only and ppo_model_path.with_suffix(".zip").exists():
        ppo_agent = load_agent(ppo_model_path, env_ppo, agent_type="ppo")
        print(f"  Loaded from {ppo_model_path}")
        results["ppo_training_rewards"] = []
    else:
        ppo_agent = create_ppo_agent(env_ppo)
        eval_env_ppo = ScimaiEnv()
        train_agent(
            ppo_agent,
            total_timesteps=timesteps,
            save_path=MODELS_DIR,
            eval_env=eval_env_ppo,
            name="phase1_ppo",
        )
        results["ppo_training_rewards"] = [
            ep_info["r"] for ep_info in ppo_agent.ep_info_buffer
        ]

    results["ppo"] = evaluate_agent(ppo_agent, env_ppo, n_episodes=n_eval)
    print(
        f"  PPO profit: {results['ppo']['mean_profit']:.1f} "
        f"± {results['ppo']['std_profit']:.1f}"
    )

    # ── 4. Train + Evaluate A2C (substitute for A3C) ─────────────────────
    print(f"\n▸ {'Loading' if eval_only else 'Training'} A2C agent (≈A3C)...")
    env_a2c = Monitor(ScimaiEnv())
    a2c_model_path = MODELS_DIR / "phase1_a2c_final"

    if eval_only and a2c_model_path.with_suffix(".zip").exists():
        a2c_agent = load_agent(a2c_model_path, env_a2c, agent_type="a2c")
        print(f"  Loaded from {a2c_model_path}")
        print("  NOTE: If A2C results look wrong, delete saved model and retrain.")
        print("  A2C config was updated (n_steps=128, gae=0.95, ent=0.01).")
        results["a2c_training_rewards"] = []
    else:
        a2c_agent = create_a2c_agent(env_a2c)
        eval_env_a2c = ScimaiEnv()
        train_agent(
            a2c_agent,
            total_timesteps=timesteps,
            save_path=MODELS_DIR,
            eval_env=eval_env_a2c,
            name="phase1_a2c",
        )
        results["a2c_training_rewards"] = [
            ep_info["r"] for ep_info in a2c_agent.ep_info_buffer
        ]

    results["a2c"] = evaluate_agent(a2c_agent, env_a2c, n_episodes=n_eval)
    print(
        f"  A2C profit: {results['a2c']['mean_profit']:.1f} "
        f"± {results['a2c']['std_profit']:.1f}"
    )

    # ── 5. Build comparison data ─────────────────────────────────────────
    results["comparison"] = _build_comparison(results)

    return results


def _build_comparison(results: dict[str, Any]) -> dict[str, Any]:
    """Build paper vs. our results comparison."""
    comparison = {
        "(s,Q)": {
            "paper_mean": PAPER_RESULTS["sq_mean"],
            "paper_std": PAPER_RESULTS["sq_std"],
            "our_mean": results["sq"]["mean_profit"],
            "our_std": results["sq"]["std_profit"],
        },
        "PPO (SB3)": {
            "paper_mean": PAPER_RESULTS["ppo_mean"],
            "paper_std": PAPER_RESULTS["ppo_std"],
            "our_mean": results["ppo"]["mean_profit"],
            "our_std": results["ppo"]["std_profit"],
        },
        "A2C (≈A3C)": {
            "paper_mean": PAPER_RESULTS["a3c_mean"],
            "paper_std": PAPER_RESULTS["a3c_std"],
            "our_mean": results["a2c"]["mean_profit"],
            "our_std": results["a2c"]["std_profit"],
        },
        "Oracle": {
            "paper_mean": PAPER_RESULTS["oracle_mean"],
            "paper_std": PAPER_RESULTS["oracle_std"],
            "our_mean": results["oracle"]["mean_profit"],
            "our_std": results["oracle"]["std_profit"],
        },
    }

    # Compute deltas and within-range flags
    for name, data in comparison.items():
        delta_pct = abs(data["our_mean"] - data["paper_mean"]) / max(
            abs(data["paper_mean"]), 1e-6
        ) * 100
        data["delta_pct"] = delta_pct
        data["within_range"] = delta_pct <= 20.0  # ±20% tolerance

    return comparison
