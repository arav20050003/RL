# policies/agents.py
# STATUS: SB3 PPO/A2C agent wrappers — training, loading, evaluation
# DEPENDS ON: config.py, envs/scimai_env.py
# TEST: python -c "from policies.agents import create_ppo_agent;
#        from envs.scimai_env import ScimaiEnv;
#        agent = create_ppo_agent(ScimaiEnv()); print('Agent OK')"

"""
Stable-Baselines3 RL agent wrappers for PPO and A2C.

Provides factory functions, training loops with callbacks,
model persistence, and standardised evaluation.

Three PPO variants (differ by environment configuration):
  - Blind PPO: base ScimaiEnv state only
  - Disruption-Aware PPO: state includes true_disruption_flag
  - LLM-Augmented PPO: state includes noisy risk_score
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PPO_CONFIG, A2C_CONFIG, TRAIN_CONFIG, EVAL_CONFIG, MODELS_DIR, SEED

logger = logging.getLogger(__name__)


def create_ppo_agent(
    env: Any,
    config: Optional[dict] = None,
    tensorboard_log: Optional[str] = None,
) -> PPO:
    """
    Create a PPO agent with standard hyperparameters.

    Args:
        env: Gymnasium environment instance.
        config: Optional config override (defaults to PPO_CONFIG).
        tensorboard_log: Optional tensorboard log directory.

    Returns:
        Configured SB3 PPO agent (untrained).
    """
    import copy
    cfg = copy.deepcopy(config or PPO_CONFIG)
    policy = cfg.pop("policy", "MlpPolicy")

    return PPO(
        policy=policy,
        env=env,
        tensorboard_log=tensorboard_log,
        **cfg,
    )


def create_a2c_agent(
    env: Any,
    config: Optional[dict] = None,
    tensorboard_log: Optional[str] = None,
) -> A2C:
    """
    Create an A2C agent (synchronous substitute for A3C).

    A2C used as synchronous substitute for A3C. Produces comparable
    results in single-environment settings per Mnih et al. 2016.

    Args:
        env: Gymnasium environment instance.
        config: Optional config override (defaults to A2C_CONFIG).
        tensorboard_log: Optional tensorboard log directory.

    Returns:
        Configured SB3 A2C agent (untrained).
    """
    import copy
    cfg = copy.deepcopy(config or A2C_CONFIG)
    policy = cfg.pop("policy", "MlpPolicy")

    return A2C(
        policy=policy,
        env=env,
        tensorboard_log=tensorboard_log,
        **cfg,
    )


def train_agent(
    agent: Any,
    total_timesteps: Optional[int] = None,
    save_path: Optional[Path] = None,
    eval_env: Optional[Any] = None,
    eval_freq: int = 5000,
    name: str = "agent",
) -> dict[str, Any]:
    """
    Train an SB3 agent with eval and checkpoint callbacks.

    Args:
        agent: SB3 PPO or A2C agent.
        total_timesteps: Training budget (default from TRAIN_CONFIG).
        save_path: Directory to save model checkpoints.
        eval_env: Optional separate environment for evaluation during training.
        eval_freq: How often to run eval callback (steps).
        name: Agent name prefix for saved files.

    Returns:
        Dictionary with training metadata and reward history.
    """
    timesteps = total_timesteps or TRAIN_CONFIG["total_timesteps"]
    save_dir = save_path or MODELS_DIR

    callbacks = []

    # Checkpoint callback
    checkpoint_dir = Path(save_dir) / f"{name}_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks.append(
        CheckpointCallback(
            save_freq=max(timesteps // 10, 1000),
            save_path=str(checkpoint_dir),
            name_prefix=name,
        )
    )

    # Eval callback (if eval env provided)
    if eval_env is not None:
        eval_log_dir = Path(save_dir) / f"{name}_eval_logs"
        eval_log_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(eval_log_dir),
                log_path=str(eval_log_dir),
                eval_freq=eval_freq,
                n_eval_episodes=10,
                deterministic=True,
                verbose=0,
            )
        )

    callback = CallbackList(callbacks) if callbacks else None

    logger.info(f"Training {name} for {timesteps} timesteps...")
    agent.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)

    # Save final model
    final_path = Path(save_dir) / f"{name}_final"
    agent.save(str(final_path))
    logger.info(f"Model saved to {final_path}")

    return {
        "name": name,
        "timesteps": timesteps,
        "save_path": str(final_path),
    }


def load_agent(
    path: str | Path,
    env: Any,
    agent_type: str = "ppo",
) -> Any:
    """
    Load a trained SB3 agent from disk.

    Args:
        path: Path to saved model (without .zip extension).
        env: Environment to attach to.
        agent_type: "ppo" or "a2c".

    Returns:
        Loaded SB3 agent.
    """
    path = Path(path)
    if not path.suffix:
        path_str = str(path)
    else:
        path_str = str(path)

    if agent_type.lower() == "ppo":
        return PPO.load(path_str, env=env)
    elif agent_type.lower() == "a2c":
        return A2C.load(path_str, env=env)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def evaluate_agent(
    agent: Any,
    env: Any,
    n_episodes: Optional[int] = None,
    deterministic: Optional[bool] = None,
    is_heuristic: bool = False,
) -> dict[str, Any]:
    """
    Evaluate an agent (SB3 or heuristic) over multiple episodes.

    Args:
        agent: SB3 model or heuristic with .predict(obs, env=...) method.
        env: Gymnasium environment.
        n_episodes: Number of eval episodes (default from EVAL_CONFIG).
        deterministic: Whether to use deterministic actions.
        is_heuristic: If True, passes env to agent.predict() for raw state.

    Returns:
        Dictionary with aggregated metrics:
          - mean_profit, std_profit
          - mean_service_level, std_service_level
          - mean_stockout_rate, std_stockout_rate
          - episode_profits, episode_details
    """
    n_ep = n_episodes or EVAL_CONFIG["n_eval_episodes"]
    det = deterministic if deterministic is not None else EVAL_CONFIG["deterministic"]

    episode_profits: list[float] = []
    episode_service_levels: list[float] = []
    episode_stockout_rates: list[float] = []
    episode_details: list[dict] = []
    all_inventories: list[list[float]] = []
    all_actions: list[np.ndarray] = []
    episode_disruption_costs: list[float] = []
    episode_logistics_losses: list[float] = []
    episode_disruption_periods: list[int] = []

    for ep in range(n_ep):
        obs, info = env.reset(seed=SEED + ep)
        done = False
        ep_profit = 0.0
        ep_demand_total = 0.0
        ep_fulfilled_total = 0.0
        ep_stockout_total = 0.0
        inventory_trajectory = [info.get("warehouse_inv", 0.0)]

        # For oracle: pre-generate demand sequence
        if hasattr(agent, "set_demand_sequence") and hasattr(env, "generate_demand_sequence"):
            demands = env.generate_demand_sequence(seed=SEED + ep)
            agent.set_demand_sequence(demands)

        while not done:
            if is_heuristic:
                action, _ = agent.predict(obs, env=env, deterministic=det)
            else:
                action, _ = agent.predict(obs, deterministic=det)

            all_actions.append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_profit += reward
            ep_demand_total += info.get("demand", 0.0)
            ep_fulfilled_total += info.get("fulfilled", 0.0)
            ep_stockout_total += info.get("stockout", 0.0)
            inventory_trajectory.append(info.get("warehouse_inv", 0.0))

        episode_profits.append(ep_profit)
        episode_disruption_costs.append(info.get("disruption_cost", 0.0))
        episode_logistics_losses.append(info.get("total_logistics_loss", 0.0))
        episode_disruption_periods.append(info.get("total_disruption_periods", 0))

        # Service level = fulfilled / demand
        service_level = (
            ep_fulfilled_total / ep_demand_total if ep_demand_total > 0 else 1.0
        )
        episode_service_levels.append(service_level)

        # Stockout rate = stockout / demand
        stockout_rate = (
            ep_stockout_total / ep_demand_total if ep_demand_total > 0 else 0.0
        )
        episode_stockout_rates.append(stockout_rate)
        all_inventories.append(inventory_trajectory)

        episode_details.append({
            "episode": ep,
            "profit": ep_profit,
            "service_level": service_level,
            "stockout_rate": stockout_rate,
            "total_demand": ep_demand_total,
            "total_fulfilled": ep_fulfilled_total,
            "total_stockout": ep_stockout_total,
        })

    profits_arr = np.array(episode_profits)
    service_arr = np.array(episode_service_levels)
    stockout_arr = np.array(episode_stockout_rates)
    actions_arr = np.array(all_actions) if all_actions else np.zeros((1, 2))

    return {
        "mean_profit": float(np.mean(profits_arr)),
        "std_profit": float(np.std(profits_arr)),
        "mean_service_level": float(np.mean(service_arr)),
        "std_service_level": float(np.std(service_arr)),
        "mean_stockout_rate": float(np.mean(stockout_arr)),
        "std_stockout_rate": float(np.std(stockout_arr)),
        "episode_profits": episode_profits,
        "episode_service_levels": episode_service_levels,
        "episode_stockout_rates": episode_stockout_rates,
        "episode_details": episode_details,
        "inventory_trajectories": all_inventories,
        "mean_action": np.mean(actions_arr, axis=0),
        "std_action": np.std(actions_arr, axis=0),
        "mean_disruption_cost": float(np.mean(episode_disruption_costs)) if episode_disruption_costs else 0.0,
        "std_disruption_cost": float(np.std(episode_disruption_costs)) if episode_disruption_costs else 0.0,
        "mean_logistics_loss": float(np.mean(episode_logistics_losses)) if episode_logistics_losses else 0.0,
        "std_logistics_loss": float(np.std(episode_logistics_losses)) if episode_logistics_losses else 0.0,
        "mean_disruption_periods": float(np.mean(episode_disruption_periods)) if episode_disruption_periods else 0.0,
    }
