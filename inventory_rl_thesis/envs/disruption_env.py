# envs/disruption_env.py
# STATUS: Phase 2 extended environment with disruption events + risk signals
# DEPENDS ON: config.py, envs/scimai_env.py, risk/llm_risk_signal.py
# TEST: python -c "from envs.disruption_env import DisruptionEnv;
#        from gymnasium.utils.env_checker import check_env;
#        env = DisruptionEnv(disruption_regime='frequent_short',
#                            include_risk_score=True);
#        check_env(env); print('DISRUPTION ENV OK')"

"""
Phase 2 extension of ScimaiEnv with supply chain disruption events
and an optional noisy LLM risk signal observation.

Disruption Model (inspired by Lu et al. 2025):
  - Markov ON/OFF process with geometric duration
  - When active: upstream delivery reduced by mu1, downstream by mu2
  - Three regimes: frequent_short, infrequent_long, stress_test

Observation extensions (mutually exclusive):
  - include_true_flag=True  → appends true Et(t) to state (Disruption-Aware PPO)
  - include_risk_score=True → appends noisy risk score to state (LLM-Aug PPO)
  - both False              → Blind PPO (base state only)
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

from config import DISRUPTION_CONFIG, SEED
from envs.scimai_env import ScimaiEnv
from risk.llm_risk_signal import get_risk_score

logger = logging.getLogger(__name__)


class DisruptionEnv(ScimaiEnv):
    """
    Extended supply chain environment with disruption events.

    Inherits all base dynamics from ScimaiEnv. Adds:
      - Stochastic disruption ON/OFF process
      - Reduced goods receipt during disruption (mu1)
      - Reduced demand fulfillment during disruption (mu2)
      - Optional extra observation dimensions for disruption info
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        disruption_regime: str = "frequent_short",
        include_true_flag: bool = False,
        include_risk_score: bool = False,
        risk_noise_std: Optional[float] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """
        Args:
            config: Base environment config (defaults to SCIMAI_CONFIG).
            disruption_regime: Key into DISRUPTION_CONFIG.
            include_true_flag: If True, append true Et to observation.
            include_risk_score: If True, append noisy risk score to observation.
            risk_noise_std: Override for risk signal noise std.
            render_mode: Gymnasium render mode.
        """
        super().__init__(config=config, render_mode=render_mode)

        # Validate inputs
        if include_true_flag and include_risk_score:
            raise ValueError(
                "Cannot include both true_flag and risk_score in observation. "
                "Choose one for fair comparison."
            )

        if disruption_regime not in DISRUPTION_CONFIG:
            raise ValueError(
                f"Unknown disruption regime '{disruption_regime}'. "
                f"Available: {list(DISRUPTION_CONFIG.keys())}"
            )

        # ── Disruption parameters ─────────────────────────────────────────
        d_cfg = DISRUPTION_CONFIG[disruption_regime]
        self.p_start: float = d_cfg["p_start"]
        self.mean_duration: int = d_cfg["mean_duration"]
        self.mu1: float = d_cfg["mu1"]
        self.mu2: float = d_cfg["mu2"]
        self.disruption_regime: str = disruption_regime

        # ── Observation configuration ─────────────────────────────────────
        self.include_true_flag: bool = include_true_flag
        self.include_risk_score: bool = include_risk_score
        self.risk_noise_std: Optional[float] = risk_noise_std

        # Extend observation space
        obs_dim = 5  # base ScimaiEnv dimensions
        if include_true_flag:
            obs_dim += 1
        if include_risk_score:
            obs_dim += 1

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # ── Disruption state (set in reset) ───────────────────────────────
        self.disruption_active: bool = False
        self.disruption_remaining: int = 0
        self._current_risk_score: float = 0.0

        # ── Episode disruption metrics ────────────────────────────────────
        self.total_disruption_periods: int = 0
        self.total_logistics_loss: float = 0.0
        self.disruption_cost: float = 0.0
        self.disruption_history: list[bool] = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment including disruption state."""
        obs, info = super().reset(seed=seed, options=options)

        # Reset disruption state
        self.disruption_active = False
        self.disruption_remaining = 0
        self._current_risk_score = 0.0

        # Reset disruption metrics
        self.total_disruption_periods = 0
        self.total_logistics_loss = 0.0
        self.disruption_cost = 0.0
        self.disruption_history = [False]

        # Return extended observation
        obs = self._get_obs()
        info.update(self._get_disruption_info())
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one time step with disruption mechanics."""
        # ── Update disruption state BEFORE the step ────────────────────────
        self._update_disruption_state()

        # ── Generate risk score for this period ────────────────────────────
        disruption_flag = 1 if self.disruption_active else 0
        self._current_risk_score = get_risk_score(
            true_disruption=disruption_flag,
            noise_std=self.risk_noise_std,
            rng=self._rng,
        )

        # ── Execute base step (uses overridden _receive_goods, _fulfill_demand)
        obs, reward, terminated, truncated, info = super().step(action)

        # Track disruption metrics
        if self.disruption_active:
            self.total_disruption_periods += 1
        self.disruption_history.append(self.disruption_active)

        # Re-compute obs with disruption extensions
        obs = self._get_obs()

        # Add disruption info
        info.update(self._get_disruption_info())
        info["logistics_loss"] = self._step_logistics_loss

        return obs, reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────
    # Override base methods to inject disruption mechanics
    # ──────────────────────────────────────────────────────────────────────

    def _receive_goods(self) -> float:
        """
        Modified goods receipt during disruptions.

        During disruption: received = mu1 * in_transit
        Normal:           received = in_transit
        """
        if self.disruption_active:
            received = self.mu1 * self.in_transit
            loss = (1.0 - self.mu1) * self.in_transit
            self.total_logistics_loss += loss
            self.disruption_cost += loss * self.production_cost
        else:
            received = self.in_transit

        self.in_transit = 0.0
        return received

    def _fulfill_demand(self, demand: float) -> tuple[float, float]:
        """
        Modified demand fulfillment during disruptions.

        During disruption:
          fulfilled = mu2 * min(demand, inventory)
          logistics_loss = (1 - mu2) * min(demand, inventory)
        Normal:
          fulfilled = min(demand, inventory)
        """
        available = min(demand, self.warehouse_inv)

        if self.disruption_active:
            fulfilled = self.mu2 * available
            logistics_loss = (1.0 - self.mu2) * available
            stockout = max(0.0, demand - fulfilled)
            self.total_logistics_loss += logistics_loss
            self.disruption_cost += logistics_loss * self.sale_price
            self._step_logistics_loss = logistics_loss
        else:
            fulfilled = available
            stockout = max(0.0, demand - self.warehouse_inv)
            self._step_logistics_loss = 0.0

        self.warehouse_inv -= available
        self.warehouse_inv = max(0.0, self.warehouse_inv)
        return fulfilled, stockout

    # ──────────────────────────────────────────────────────────────────────
    # Extended observation
    # ──────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Return observation with optional disruption dimensions."""
        base_obs = super()._get_obs()  # shape (5,)

        extra = []
        if self.include_true_flag:
            extra.append(1.0 if self.disruption_active else 0.0)
        if self.include_risk_score:
            extra.append(self._current_risk_score)

        if extra:
            obs = np.concatenate([base_obs, np.array(extra, dtype=np.float32)])
        else:
            obs = base_obs

        return np.clip(obs, 0.0, 1.0).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Disruption state machine
    # ──────────────────────────────────────────────────────────────────────

    def _update_disruption_state(self) -> None:
        """
        Markov ON/OFF disruption process.

        If not disrupted: start with probability p_start, duration ~ Geometric.
        If disrupted: decrement remaining, end when remaining <= 0.
        """
        assert self._rng is not None, "Call reset() before step()"

        if not self.disruption_active:
            if self._rng.random() < self.p_start:
                self.disruption_active = True
                self.disruption_remaining = max(
                    1, int(self._rng.geometric(1.0 / self.mean_duration))
                )
                logger.debug(
                    f"t={self.t}: Disruption STARTED, "
                    f"duration={self.disruption_remaining}"
                )
        else:
            self.disruption_remaining -= 1
            if self.disruption_remaining <= 0:
                self.disruption_active = False
                logger.debug(f"t={self.t}: Disruption ENDED")

    def _get_disruption_info(self) -> dict[str, Any]:
        """Return disruption-specific info."""
        return {
            "disruption_active": self.disruption_active,
            "disruption_remaining": self.disruption_remaining,
            "risk_score": self._current_risk_score,
            "true_disruption_flag": 1 if self.disruption_active else 0,
            "total_disruption_periods": self.total_disruption_periods,
            "total_logistics_loss": self.total_logistics_loss,
            "disruption_cost": self.disruption_cost,
            "disruption_regime": self.disruption_regime,
        }

    def render(self) -> None:
        """Human-readable state printout with disruption info."""
        disruption_marker = "⚠️ DISRUPTED" if self.disruption_active else "  normal   "
        logger.debug(
            f"t={self.t:3d} | {disruption_marker} | "
            f"P={self.producer_inv:.1f} W={self.warehouse_inv:.1f} "
            f"risk={self._current_risk_score:.2f} "
            f"profit={self.cumulative_profit:.1f}"
        )
