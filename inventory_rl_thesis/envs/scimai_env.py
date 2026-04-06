# envs/scimai_env.py
# STATUS: Phase 1 base environment replicating SCIMAI-Gym 1P1W scenario
# DEPENDS ON: config.py
# TEST: python -c "from envs.scimai_env import ScimaiEnv;
#        from gymnasium.utils.env_checker import check_env;
#        check_env(ScimaiEnv()); print('ENV OK')"

"""
Replicates the SCIMAI-Gym environment from:
  Stranieri & Stella (2023), "Comparing Deep Reinforcement Learning
  Algorithms in Two-Echelon Supply Chains", ECML-PKDD AI4M Workshop.
  arXiv: 2204.09603

Two-echelon supply chain: 1 producer → 1 warehouse → customers.

In the SCIMAI model the agent decides two quantities each period:
  action[0] = production order (how many units to produce)
  action[1] = shipment order  (how many units to send to warehouse)

Both are normalised to [0, 1] and mapped to [0, max_production].
Production adds to producer inventory (capped at producer_capacity).
Shipping removes from producer inventory, arrives at warehouse after lead_time.
Demand is served from warehouse inventory.

State vector (normalised to [0, 1]):
  [producer_inv / cap_p, warehouse_inv / cap_w,
   in_transit / max_production, demand / (2*d_max), t / T]

Reward:
  profit = sales_revenue − production_cost − storage_cost
           − transport_cost − stockout_penalty

Episode length: T steps (truncated, not terminated).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SCIMAI_CONFIG, SEED

logger = logging.getLogger(__name__)


class ScimaiEnv(gym.Env):
    """
    Gymnasium environment replicating the 1-product, 1-warehouse (1P1W)
    two-echelon supply chain from Stranieri & Stella (2023).
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        config: Optional[dict] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        cfg = config or SCIMAI_CONFIG

        # ── Environment parameters ────────────────────────────────────────
        self.T: int = cfg["T"]
        self.d_max: int = cfg["d_max"]
        self.d_var: int = cfg["d_var"]
        self.sale_price: float = cfg["sale_price"]
        self.production_cost: float = cfg["production_cost"]
        self.cap_producer: int = cfg["storage_capacity_producer"]
        self.cap_warehouse: int = cfg["storage_capacity_warehouse"]
        self.storage_cost_producer: float = cfg["storage_cost_producer"]
        self.storage_cost_warehouse: float = cfg["storage_cost_warehouse"]
        self.transport_cost: float = cfg["transportation_cost"]
        self.penalty_cost: float = cfg["penalty_cost"]
        self.lead_time: int = cfg["lead_time"]
        self.max_production: int = cfg.get("max_production", 2 * cfg["d_max"])
        self.render_mode = render_mode

        # ── Spaces ────────────────────────────────────────────────────────
        # Observation: [prod_inv, wh_inv, in_transit, demand, time]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )
        # Action: [produce_fraction, ship_fraction]
        # produce_fraction * max_production → units to produce
        # ship_fraction * max_production → units to ship
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # ── Internal state (set in reset) ─────────────────────────────────
        self._rng: Optional[np.random.Generator] = None
        self.producer_inv: float = 0.0
        self.warehouse_inv: float = 0.0
        self.in_transit: float = 0.0
        self.current_demand: float = 0.0
        self.t: int = 0

        # ── Episode metrics ───────────────────────────────────────────────
        self.cumulative_profit: float = 0.0
        self.total_sales_revenue: float = 0.0
        self.total_stockout_units: float = 0.0
        self.total_units_sold: float = 0.0
        self.demand_history: list[float] = []

    # ──────────────────────────────────────────────────────────────────────
    # Core API
    # ──────────────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed if seed is not None else SEED)

        # Start with half-full inventories (common in SCIMAI)
        self.producer_inv = self.cap_producer / 2.0
        self.warehouse_inv = self.cap_warehouse / 2.0
        self.in_transit = 0.0
        self.t = 0

        # Generate initial demand
        self.current_demand = self._compute_demand(self.t)

        # Reset metrics
        self.cumulative_profit = 0.0
        self.total_sales_revenue = 0.0
        self.total_stockout_units = 0.0
        self.total_units_sold = 0.0
        self.demand_history = [self.current_demand]

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one time step.

        Dynamics order:
          1. Receive in-transit goods at warehouse
          2. Generate demand
          3. Fulfill demand from warehouse (record stockout)
          4. Execute production decision → add to producer inventory
          5. Execute shipping decision → move from producer to in-transit
          6. Cap inventories at storage_capacity
          7. Compute and return reward
        """
        action = np.clip(action, 0.0, 1.0)
        produce_frac = float(action[0])
        ship_frac = float(action[1])

        # ── Step 1: Receive in-transit goods at warehouse ──────────────────
        received = self._receive_goods()
        self.warehouse_inv = min(
            self.warehouse_inv + received, self.cap_warehouse
        )

        # ── Step 2: Generate demand ────────────────────────────────────────
        self.current_demand = self._compute_demand(self.t)
        demand = self.current_demand

        # ── Step 3: Fulfill demand from warehouse ──────────────────────────
        fulfilled, stockout_qty = self._fulfill_demand(demand)

        # ── Step 4: Production ─────────────────────────────────────────────────
        desired_produce = produce_frac * self.max_production
        space_at_producer = self.cap_producer - self.producer_inv
        desired_ship = ship_frac * self.max_production
        # Only produce what can be stored OR immediately shipped
        produce_qty = float(np.clip(
            min(desired_produce, space_at_producer + desired_ship), 
            0.0, self.max_production
        ))
        available_at_producer = self.producer_inv + produce_qty

        # ── Step 5: Shipping ───────────────────────────────────────────────────
        ship_qty = float(np.clip(
            min(desired_ship, available_at_producer), 
            0.0, self.max_production
        ))
        self.producer_inv = float(np.clip(
            available_at_producer - ship_qty, 0.0, self.cap_producer
        ))
        self.in_transit = ship_qty

        # ── Step 6: Cap inventories ────────────────────────────────────────
        self.producer_inv = max(self.producer_inv, 0.0)
        self.warehouse_inv = max(self.warehouse_inv, 0.0)

        # ── Step 7: Compute reward ─────────────────────────────────────────
        reward = self._compute_reward(
            fulfilled=fulfilled,
            produce_qty=produce_qty,
            ship_qty=ship_qty,
            stockout_qty=stockout_qty,
        )

        # Update metrics
        self.cumulative_profit += reward
        self.total_units_sold += fulfilled
        self.total_stockout_units += stockout_qty
        self.total_sales_revenue += fulfilled * self.sale_price
        self.demand_history.append(demand)

        # Advance time
        self.t += 1
        terminated = False
        truncated = self.t >= self.T

        obs = self._get_obs()
        info = self._get_info()
        info.update({
            "step_reward": reward,
            "fulfilled": fulfilled,
            "stockout": stockout_qty,
            "produced": produce_qty,
            "shipped": ship_qty,
            "received": received,
            "demand": demand,
        })

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Return normalised observation vector."""
        obs = np.array(
            [
                self.producer_inv / max(self.cap_producer, 1),
                self.warehouse_inv / max(self.cap_warehouse, 1),
                self.in_transit / max(self.max_production, 1),
                self.current_demand / max(2 * self.d_max, 1),
                self.t / max(self.T, 1),
            ],
            dtype=np.float32,
        )
        return np.clip(obs, 0.0, 1.0)

    def _get_info(self) -> dict[str, Any]:
        """Return info dict with current state."""
        return {
            "t": self.t,
            "producer_inv": self.producer_inv,
            "warehouse_inv": self.warehouse_inv,
            "in_transit": self.in_transit,
            "current_demand": self.current_demand,
            "cumulative_profit": self.cumulative_profit,
        }

    def _compute_demand(self, t: int) -> float:
        """
        Stochastic demand with seasonal sinusoidal component.

        d(t) = d_max + d_var * sin(2π·t / T) + ε
        where ε ~ N(0, 0.5), clipped to [0, 2·d_max].
        """
        assert self._rng is not None, "Call reset() before step()"
        seasonal = self.d_var * np.sin(2 * np.pi * t / self.T)
        noise = self._rng.normal(0.0, 0.5)
        demand = self.d_max + seasonal + noise
        return float(np.clip(demand, 0.0, 2 * self.d_max))

    def _receive_goods(self) -> float:
        """Receive in-transit goods. Override in DisruptionEnv."""
        received = self.in_transit
        self.in_transit = 0.0
        return received

    def _fulfill_demand(self, demand: float) -> tuple[float, float]:
        """
        Fulfill demand from warehouse inventory.
        Returns (fulfilled_qty, stockout_qty). Override in DisruptionEnv.
        """
        fulfilled = min(demand, self.warehouse_inv)
        stockout = max(0.0, demand - self.warehouse_inv)
        self.warehouse_inv -= fulfilled
        return fulfilled, stockout

    def _compute_reward(
        self,
        fulfilled: float,
        produce_qty: float,
        ship_qty: float,
        stockout_qty: float,
    ) -> float:
        """
        Compute single-step profit / reward.

        reward = sales_revenue − production_cost − storage_cost
                 − transport_cost − stockout_penalty
        """
        sales_revenue = fulfilled * self.sale_price
        prod_cost = produce_qty * self.production_cost
        storage_cost = (
            self.producer_inv * self.storage_cost_producer
            + self.warehouse_inv * self.storage_cost_warehouse
        )
        trans_cost = ship_qty * self.transport_cost
        penalty = stockout_qty * self.penalty_cost

        reward = sales_revenue - prod_cost - storage_cost - trans_cost - penalty
        return float(reward)

    def render(self) -> None:
        """Human-readable state printout."""
        logger.debug(
            f"t={self.t:3d} | P_inv={self.producer_inv:.1f} "
            f"W_inv={self.warehouse_inv:.1f} transit={self.in_transit:.1f} "
            f"demand={self.current_demand:.1f} profit={self.cumulative_profit:.1f}"
        )

    def close(self) -> None:
        """Clean up resources."""
        pass

    # ──────────────────────────────────────────────────────────────────────
    # Utilities for heuristic / oracle policies
    # ──────────────────────────────────────────────────────────────────────

    def get_raw_state(self) -> dict[str, float]:
        """Return un-normalised state for heuristic policies."""
        return {
            "producer_inv": self.producer_inv,
            "warehouse_inv": self.warehouse_inv,
            "in_transit": self.in_transit,
            "current_demand": self.current_demand,
            "t": self.t,
        }

    def generate_demand_sequence(self, seed: Optional[int] = None) -> list[float]:
        """Pre-generate full demand sequence for oracle policy."""
        rng = np.random.default_rng(seed if seed is not None else SEED)
        demands = []
        for t in range(self.T):
            seasonal = self.d_var * np.sin(2 * np.pi * t / self.T)
            noise = rng.normal(0.0, 0.5)
            d = float(np.clip(self.d_max + seasonal + noise, 0.0, 2 * self.d_max))
            demands.append(d)
        return demands
