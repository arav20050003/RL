# risk/llm_risk_signal.py
# STATUS: Simulated LLM disruption risk signal via NewsOracle + RiskParser pipeline
# DEPENDS ON: config.py
# TEST: python -c "from risk.llm_risk_signal import get_risk_score; print(get_risk_score(1)); print(get_risk_score(0))"

"""
Oracle News Module pipeline for supply chain risk classification.

Simulates an LLM pipeline by first generating synthetic news headlines
based on true disruption state, then parsing them to output risk scores.
ZERO external API calls.
"""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RISK_CONFIG

class NewsOracle:
    """Generates synthetic supply chain news simulating what an LLM would read."""
    @staticmethod
    def generate_news(disruption: int, rng: np.random.Generator) -> str:
        if disruption == 1:
            return rng.choice([
                "SEVERE HURRICANE WARNING: Port operations halted indefinitely.",
                "URGENT: Major factory explosion stops all production lines.",
                "CRITICAL ALERT: Key component supplier declares bankruptcy.",
            ])
        else:
            return rng.choice([
                "Supply chain operations running smoothly as usual.",
                "Routine maritime shipping continues without issues.",
                "Clear weather and stable markets reported across regions.",
            ])


class RiskParser:
    """Parses text headlines and outputs a calibrated probability [0,1]."""
    @staticmethod
    def parse_risk(news_text: str, rng: np.random.Generator, noise_std: float) -> float:
        # Simulate LLM extracting sentiment / severity
        if "SEVERE" in news_text or "URGENT" in news_text or "CRITICAL" in news_text:
            base_score = 0.95
        else:
            base_score = 0.05
            
        noise = rng.normal(0.0, noise_std)
        return float(np.clip(base_score + noise, 0.0, 1.0))


def get_risk_score(
    true_disruption: int,
    noise_std: float | None = None,
    rng: np.random.Generator | None = None,
    seed: int | None = None
) -> float:
    """
    Simulate the pipeline of an LLM-based supply chain risk classifier.

    Args:
        true_disruption: Ground truth disruption flag (0 or 1).
        noise_std: Gaussian noise std dev (default from RISK_CONFIG).
        rng: numpy random generator for reproducibility.
        seed: random seed if rng is not provided.

    Returns:
        float in [0.0, 1.0] representing estimated disruption probability.
    """
    if noise_std is None:
        noise_std = RISK_CONFIG["noise_std"]
    if rng is None:
        rng = np.random.default_rng(seed)

    news_headline = NewsOracle.generate_news(true_disruption, rng)
    score = RiskParser.parse_risk(news_headline, rng, noise_std)
    return score


def get_risk_score_batch(
    disruption_sequence: list[int],
    noise_std: float | None = None,
    seed: int = 42,
) -> list[float]:
    """
    Generate risk scores for a full episode's disruption sequence.

    Args:
        disruption_sequence: List of ground truth disruption flags.
        noise_std: Gaussian noise std dev.
        seed: Random seed for reproducibility.

    Returns:
        List of risk scores, one per element in disruption_sequence.
    """
    rng = np.random.default_rng(seed)
    return [get_risk_score(d, noise_std, rng=rng) for d in disruption_sequence]
