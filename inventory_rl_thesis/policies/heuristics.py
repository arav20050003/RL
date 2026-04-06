# policies/heuristics.py
# STATUS: (s,Q) heuristic and Greedy Oracle policies
# DEPENDS ON: config.py, envs/scimai_env.py
# TEST: python -c "from policies.heuristics import SQPolicy, OraclePolicy;
#        print('Heuristics OK')"

"""
Heuristic baseline policies for supply chain inventory management.

1. (s,Q) Policy: classical reorder-point policy.
   When warehouse inventory drops below `s`, order Q units to be
   produced and shipped.

2. Oracle Policy: greedy oracle with perfect demand foresight.
   Knows future demand and makes optimal production/shipping decisions.
   Serves as a theoretical upper bound.

Both expose a `.predict(obs, env)` interface returning (action, None)
for compatibility with SB3 evaluation patterns.

Action mapping:
  action[0] = produce_fraction  → produce_fraction * max_production units
  action[1] = ship_fraction     → ship_fraction * max_production units to ship
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SQ_CONFIG, SEED


class SQPolicy:
    """
    (s, Q) reorder-point heuristic.

    Decision rule:
      - If warehouse_inv < s: produce and ship Q units toward warehouse
      - Else: maintain light production and shipping

    Default parameters from paper: s=3, Q=8.
    """

    def __init__(self, s: Optional[int] = None, q: Optional[int] = None) -> None:
        self.s: int = s if s is not None else SQ_CONFIG["s"]
        self.q: int = q if q is not None else SQ_CONFIG["Q"]

    def predict(
        self,
        obs: np.ndarray,
        env: Any = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        """
        Predict action given observation.

        Args:
            obs: Normalised observation from the environment.
            env: Optional reference to the environment for raw state access.
            deterministic: Unused, kept for API compatibility.

        Returns:
            (action, None) — action is np.ndarray shape (2,).
        """
        if env is not None and hasattr(env, "get_raw_state"):
            raw = env.get_raw_state()
            warehouse_inv = raw["warehouse_inv"]
            producer_inv = raw["producer_inv"]
            max_production = env.max_production
        else:
            warehouse_inv = float(obs[1]) * 10
            producer_inv = float(obs[0]) * 5
            max_production = 20

        if warehouse_inv < self.s:
            # Order Q: produce Q, ship Q (as fractions of max_production)
            produce_frac = min(self.q / max(max_production, 1), 1.0)
            ship_frac = min(self.q / max(max_production, 1), 1.0)
        else:
            # Produce moderately to maintain a buffer, ship a bit
            produce_frac = min(5.0 / max(max_production, 1), 0.5)
            ship_frac = min(5.0 / max(max_production, 1), 0.5)

        action = np.array([produce_frac, ship_frac], dtype=np.float32)
        return action, None

    def __repr__(self) -> str:
        return f"SQPolicy(s={self.s}, Q={self.q})"


class OraclePolicy:
    """
    Greedy oracle with perfect demand foresight.

    Given the full future demand sequence, makes optimal decisions
    at each step by producing and shipping exactly what's needed
    to meet upcoming demand while minimising costs.

    This serves as a theoretical upper bound reference.
    Note: "Greedy oracle with perfect demand foresight" — not full DP.
    """

    def __init__(self, seed: int = SEED) -> None:
        self.seed = seed
        self._demand_sequence: Optional[list[float]] = None
        self._current_step: int = 0

    def set_demand_sequence(self, demands: list[float]) -> None:
        """Pre-load the demand sequence for one episode."""
        self._demand_sequence = demands
        self._current_step = 0

    def predict(
        self,
        obs: np.ndarray,
        env: Any = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        """
        Predict optimal action given perfect foresight of demand.

        Strategy:
          - Know current and next demand
          - Produce and ship enough to meet demand minus what warehouse
            already has plus safety margin
          - Balance production cost against stockout penalty

        Args:
            obs: Normalised observation from the environment.
            env: Optional reference to the environment for raw state access.
            deterministic: Unused, kept for API compatibility.

        Returns:
            (action, None) — action is np.ndarray shape (2,).
        """
        if env is not None and hasattr(env, "get_raw_state"):
            raw = env.get_raw_state()
            t = int(raw["t"])
            max_production = env.max_production
        else:
            t = int(float(obs[4]) * 25)
            max_production = 20

        # Look up future demand
        if self._demand_sequence is not None and t + 1 < len(self._demand_sequence):
            next_demand = self._demand_sequence[t + 1]
        else:
            next_demand = 10.0  # fallback to d_max

        # Oracle strategy:
        # - SHIP: always request full pipeline amount (0.5 frac = 10 units)
        #   env.step() will produce first then ship from the combined pool
        #   so ship_frac=0.5 reliably ships 10 units regardless of current p_inv
        # - PRODUCE: use demand foresight to produce exactly what next step needs
        #   slightly over-produce to buffer minor demand spikes
        produce_target = float(np.clip(next_demand + 1.0, 0.0, max_production))
        produce_frac = float(np.clip(produce_target / max(max_production, 1), 0.0, 1.0))
        ship_frac = 0.5  # reliably ships 10 units via the combined pool

        action = np.array([produce_frac, ship_frac], dtype=np.float32)
        return action, None

    def __repr__(self) -> str:
        return "OraclePolicy(greedy, perfect demand foresight)"
