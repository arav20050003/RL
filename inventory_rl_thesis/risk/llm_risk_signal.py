# risk/llm_risk_signal.py
# STATUS: Simulated LLM disruption risk signal — zero external dependencies
# DEPENDS ON: config.py
# TEST: python -c "from risk.llm_risk_signal import get_risk_score;
#        print(get_risk_score(1)); print(get_risk_score(0))"

"""
Simulated LLM-based supply chain risk classifier.

This module provides a *pure mathematical simulation* of what an
LLM-based news/event classifier would output as a disruption
probability score. There are ZERO external API calls.

Noise model:
  score = clip(true_disruption + N(0, noise_std), 0, 1)

At the default noise_std=0.25:
  - When disruption=1: score ~ N(1, 0.25) → ~98% TPR (P(score>0.5))
  - When disruption=0: score ~ N(0, 0.25) → ~2% FPR (P(score>0.5))
  - This simulates an imperfect but well-calibrated classifier

In deployment, replace `get_risk_score()` body with an actual LLM API call:
  # def get_risk_score_from_news(news_text: str) -> float:
  #     response = llm_client.chat(
  #         system="You are a supply chain risk analyst. Given news text,
  #                 return only a float 0-1 for disruption probability.",
  #         user=news_text
  #     )
  #     return float(response)
"""

from __future__ import annotations

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RISK_CONFIG


def get_risk_score(
    true_disruption: int,
    noise_std: float | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Simulate the output of an LLM-based supply chain risk classifier.

    Args:
        true_disruption: Ground truth disruption flag (0 or 1).
        noise_std: Gaussian noise std dev (default from RISK_CONFIG).
        rng: numpy random generator for reproducibility.

    Returns:
        float in [0.0, 1.0] representing estimated disruption probability.
    """
    if noise_std is None:
        noise_std = RISK_CONFIG["noise_std"]
    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(0.0, noise_std)
    score = float(np.clip(true_disruption + noise, 0.0, 1.0))
    return score


def get_risk_score_batch(
    disruption_sequence: list[int],
    noise_std: float | None = None,
    seed: int = 42,
) -> list[float]:
    """
    Generate risk scores for a full episode's disruption sequence.

    Useful for pre-generating a consistent signal for one episode.

    Args:
        disruption_sequence: List of ground truth disruption flags.
        noise_std: Gaussian noise std dev.
        seed: Random seed for reproducibility.

    Returns:
        List of risk scores, one per element in disruption_sequence.
    """
    rng = np.random.default_rng(seed)
    return [get_risk_score(d, noise_std, rng) for d in disruption_sequence]
