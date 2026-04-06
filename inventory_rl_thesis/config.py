# config.py
# STATUS: Central configuration — all hyperparams, env params, paths
# DEPENDS ON: nothing
# TEST: python -c "from config import SCIMAI_CONFIG; print(SCIMAI_CONFIG)"

"""
Central configuration for the LLM-Augmented RL Inventory Control thesis.
All parameters consolidated here — nothing hardcoded in other files.
"""

from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED: int = 42

# ── SCIMAI Environment (Stranieri & Stella 2023, 1P1W scenario) ──────────────
SCIMAI_CONFIG: dict = {
    "product_types_num": 1,
    "distr_warehouses_num": 1,
    "T": 25,
    "d_max": 10,
    "d_var": 2,
    "sale_price": 15.0,
    "production_cost": 5.0,
    "storage_capacity_producer": 5,
    "storage_capacity_warehouse": 10,
    "storage_cost_producer": 2.0,
    "storage_cost_warehouse": 1.0,
    "transportation_cost": 0.25,
    "penalty_cost": 22.5,
    "lead_time": 1,  # periods for shipment to arrive
    "max_production": 20,  # max units producible per period (2 * d_max)
}

# ── (s,Q) Heuristic ───────────────────────────────────────────────────────────
SQ_CONFIG: dict = {
    "s": 3,  # reorder point
    "Q": 8,  # order quantity
}

# ── PPO Hyperparameters (SB3) ─────────────────────────────────────────────────
PPO_CONFIG: dict = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 512,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "policy_kwargs": {"net_arch": [64, 64]},
    "verbose": 0,
    "seed": SEED,
}

# ── A2C Hyperparameters (SB3) — synchronous substitute for A3C ───────────────
# A2C used as synchronous substitute for A3C. Produces comparable results
# in single-environment settings per Mnih et al. 2016.
A2C_CONFIG: dict = {
    "policy": "MlpPolicy",
    "learning_rate": 7e-4,
    "n_steps": 5,
    "gamma": 0.99,
    "gae_lambda": 1.0,
    "ent_coef": 0.0,
    "policy_kwargs": {"net_arch": [64, 64]},
    "verbose": 0,
    "seed": SEED,
}

# ── Training ──────────────────────────────────────────────────────────────────
TRAIN_CONFIG: dict = {
    "total_timesteps": 300_000,
    "quick_test_timesteps": 5_000,  # for --quick-test flag
}

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_CONFIG: dict = {
    "n_eval_episodes": 100,
    "deterministic": True,
}

# ── Disruption Regimes (Lu et al. 2025) ──────────────────────────────────────
DISRUPTION_CONFIG: dict = {
    "frequent_short": {
        "p_start": 0.05,
        "mean_duration": 3,
        "mu1": 0.6,
        "mu2": 0.7,
    },
    "infrequent_long": {
        "p_start": 0.01,
        "mean_duration": 20,
        "mu1": 0.4,
        "mu2": 0.5,
    },
    "stress_test": {
        "p_start": 0.15,
        "mean_duration": 10,
        "mu1": 0.3,
        "mu2": 0.4,
    },
}

# ── LLM Risk Signal (simulation only — no API calls) ─────────────────────────
RISK_CONFIG: dict = {
    "noise_std": 0.25,
    # Accuracy profile at noise_std=0.25:
    # True positive rate  ≈ 75%
    # False positive rate ≈ 20%
    # This simulates an imperfect LLM-based news classifier
}

# ── Paper Reference Values (Stranieri & Stella 2023, 1P1W Exp 1) ─────────────
PAPER_RESULTS: dict = {
    "sq_mean": 1226.0,
    "sq_std": 71.0,
    "ppo_mean": 1213.0,
    "ppo_std": 68.0,
    "a3c_mean": 870.0,
    "a3c_std": 67.0,
    "oracle_mean": 1474.0,
    "oracle_std": 45.0,
}

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent
MODELS_DIR: Path = BASE_DIR / "saved_models"
RESULTS_DIR: Path = BASE_DIR / "results" / "output"
PLOTS_DIR: Path = BASE_DIR / "results" / "plots"

for _dir in [MODELS_DIR, RESULTS_DIR, PLOTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
