# experiments/phase2_extend.py
# STATUS: Phase 2 — disruption experiments with LLM-augmented PPO
# DEPENDS ON: config.py, envs/disruption_env.py, policies/agents.py,
#             policies/heuristics.py
# TEST: python -c "from experiments.phase2_extend import run_phase2"

"""
Phase 2: Novel extension with supply chain disruptions and LLM risk signals.

For each disruption regime (frequent_short, infrequent_long):
  1. Evaluate (s,Q) heuristic (no disruption awareness)
  2. Train + evaluate PPO Blind (base state only)
  3. Train + evaluate PPO Disruption-Aware (state includes true_disruption_flag)
  4. Train + evaluate PPO LLM-Augmented (state includes noisy risk_score)
  5. Compute metrics: avg profit, service level, stockout rate, disruption cost
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
    DISRUPTION_CONFIG,
    EVAL_CONFIG,
    MODELS_DIR,
    SEED,
    TRAIN_CONFIG,
)
from envs.disruption_env import DisruptionEnv
from policies.agents import (
    create_ppo_agent,
    evaluate_agent,
    load_agent,
    train_agent,
)
from policies.heuristics import SQPolicy

logger = logging.getLogger(__name__)


def run_phase2(
    regimes: list[str] | None = None,
    total_timesteps: int | None = None,
    eval_only: bool = False,
    quick_test: bool = False,
) -> dict[str, Any]:
    """
    Run Phase 2 disruption experiments.

    Args:
        regimes: List of disruption regimes. Default: frequent_short, infrequent_long.
        total_timesteps: Override training budget.
        eval_only: If True, load saved models instead of training.
        quick_test: Use reduced timesteps.

    Returns:
        Nested dict: {regime: {policy: metrics_dict}}.
    """
    if regimes is None:
        regimes = ["frequent_short", "infrequent_long"]

    if quick_test:
        timesteps = TRAIN_CONFIG["quick_test_timesteps"]
    else:
        timesteps = total_timesteps or TRAIN_CONFIG["total_timesteps"]

    n_eval = EVAL_CONFIG["n_eval_episodes"]

    print("\n" + "=" * 70)
    print("  PHASE 2: DISRUPTION EXTENSION — LLM-Augmented RL")
    print(f"  Regimes: {regimes}")
    print(f"  Training budget: {timesteps:,} | Eval episodes: {n_eval}")
    print("=" * 70)

    all_results: dict[str, Any] = {}

    for regime in regimes:
        print(f"\n{'─' * 70}")
        print(f"  Regime: {regime.upper().replace('_', ' ')}")
        print(f"{'─' * 70}")

        regime_results = _run_regime(
            regime=regime,
            timesteps=timesteps,
            n_eval=n_eval,
            eval_only=eval_only,
        )
        all_results[regime] = regime_results

    return all_results


def _run_regime(
    regime: str,
    timesteps: int,
    n_eval: int,
    eval_only: bool,
) -> dict[str, Any]:
    """Run all 4 policies for a single disruption regime."""
    results: dict[str, Any] = {}

    # ── 1. (s,Q) Heuristic (no disruption awareness) ─────────────────────
    print("\n  ▸ Evaluating (s,Q) heuristic...")
    env_sq = DisruptionEnv(
        disruption_regime=regime,
        include_true_flag=False,
        include_risk_score=False,
    )
    sq_policy = SQPolicy()
    results["sq"] = evaluate_agent(
        sq_policy, env_sq, n_episodes=n_eval, is_heuristic=True
    )
    _print_policy_result("(s,Q)", results["sq"])

    # ── 2. PPO Blind (base state only) ───────────────────────────────────
    print("\n  ▸ PPO Blind (no disruption info)...")
    env_blind = Monitor(DisruptionEnv(
        disruption_regime=regime,
        include_true_flag=False,
        include_risk_score=False,
    ))
    model_path = MODELS_DIR / f"phase2_{regime}_ppo_blind_final"

    if eval_only and not model_path.with_suffix(".zip").exists():
        print(f"  WARNING: No saved model at {model_path}. Training instead.")
        eval_only_this = False
    else:
        eval_only_this = eval_only

    if eval_only_this and model_path.with_suffix(".zip").exists():
        ppo_blind = load_agent(model_path, env_blind, agent_type="ppo")
        results["ppo_blind_training_rewards"] = []
    else:
        ppo_blind = create_ppo_agent(env_blind)
        eval_env = DisruptionEnv(
            disruption_regime=regime,
            include_true_flag=False,
            include_risk_score=False,
        )
        train_agent(
            ppo_blind,
            total_timesteps=timesteps,
            save_path=MODELS_DIR,
            eval_env=eval_env,
            name=f"phase2_{regime}_ppo_blind",
        )
        results["ppo_blind_training_rewards"] = [
            ep_info["r"] for ep_info in ppo_blind.ep_info_buffer
        ]

    results["ppo_blind"] = evaluate_agent(
        ppo_blind, env_blind, n_episodes=n_eval
    )
    _print_policy_result("PPO Blind", results["ppo_blind"])

    # ── 3. PPO Disruption-Aware (true flag in state) ─────────────────────
    print("\n  ▸ PPO Disruption-Aware (true flag)...")
    env_aware = Monitor(DisruptionEnv(
        disruption_regime=regime,
        include_true_flag=True,
        include_risk_score=False,
    ))
    model_path = MODELS_DIR / f"phase2_{regime}_ppo_aware_final"

    if eval_only and not model_path.with_suffix(".zip").exists():
        print(f"  WARNING: No saved model at {model_path}. Training instead.")
        eval_only_this = False
    else:
        eval_only_this = eval_only

    if eval_only_this and model_path.with_suffix(".zip").exists():
        ppo_aware = load_agent(model_path, env_aware, agent_type="ppo")
        results["ppo_aware_training_rewards"] = []
    else:
        ppo_aware = create_ppo_agent(env_aware)
        eval_env = DisruptionEnv(
            disruption_regime=regime,
            include_true_flag=True,
            include_risk_score=False,
        )
        train_agent(
            ppo_aware,
            total_timesteps=timesteps,
            save_path=MODELS_DIR,
            eval_env=eval_env,
            name=f"phase2_{regime}_ppo_aware",
        )
        results["ppo_aware_training_rewards"] = [
            ep_info["r"] for ep_info in ppo_aware.ep_info_buffer
        ]

    results["ppo_aware"] = evaluate_agent(
        ppo_aware, env_aware, n_episodes=n_eval
    )
    _print_policy_result("PPO Disrupt-Aware", results["ppo_aware"])

    # ── 4. PPO LLM-Augmented (noisy risk score in state) ─────────────────
    print("\n  ▸ PPO LLM-Augmented (noisy risk score)...")
    env_llm = Monitor(DisruptionEnv(
        disruption_regime=regime,
        include_true_flag=False,
        include_risk_score=True,
    ))
    model_path = MODELS_DIR / f"phase2_{regime}_ppo_llm_final"

    if eval_only and not model_path.with_suffix(".zip").exists():
        print(f"  WARNING: No saved model at {model_path}. Training instead.")
        eval_only_this = False
    else:
        eval_only_this = eval_only

    if eval_only_this and model_path.with_suffix(".zip").exists():
        ppo_llm = load_agent(model_path, env_llm, agent_type="ppo")
        results["ppo_llm_training_rewards"] = []
    else:
        ppo_llm = create_ppo_agent(env_llm)
        eval_env = DisruptionEnv(
            disruption_regime=regime,
            include_true_flag=False,
            include_risk_score=True,
        )
        train_agent(
            ppo_llm,
            total_timesteps=timesteps,
            save_path=MODELS_DIR,
            eval_env=eval_env,
            name=f"phase2_{regime}_ppo_llm",
        )
        results["ppo_llm_training_rewards"] = [
            ep_info["r"] for ep_info in ppo_llm.ep_info_buffer
        ]

    results["ppo_llm"] = evaluate_agent(
        ppo_llm, env_llm, n_episodes=n_eval
    )
    _print_policy_result("PPO LLM-Augmented", results["ppo_llm"])

    return results



def _print_policy_result(name: str, metrics: dict[str, Any]) -> None:
    """Print a one-line result summary for a policy."""
    print(
        f"    {name:20s} | profit: {metrics['mean_profit']:8.1f} ± {metrics['std_profit']:.1f}"
        f" | service: {metrics['mean_service_level']:.3f}"
        f" | stockout: {metrics['mean_stockout_rate']:.3f}"
    )
