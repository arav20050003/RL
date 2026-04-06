# results/tables.py
# STATUS: Publication-quality comparison tables for Phase 1 and Phase 2
# DEPENDS ON: config.py
# TEST: python -c "from results.tables import print_phase1_table"

"""
Formatted comparison tables for thesis results presentation.

Table 1: Phase 1 — Paper vs. Our replication
Table 2: Phase 2 — Disruption extension results per regime
Table 3: Stress test results
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RESULTS_DIR


def print_phase1_table(comparison: dict[str, Any]) -> None:
    """
    Print Phase 1 replication comparison table.

    Expected format of `comparison`:
    {
        "(s,Q)":       {"paper_mean": ..., "paper_std": ..., "our_mean": ..., "our_std": ..., "within_range": ...},
        "PPO (SB3)":   {...},
        "A2C (≈A3C)":  {...},
        "Oracle":      {...},
    }
    """
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          PHASE 1: REPLICATION vs STRANIERI & STELLA (2023)          ║")
    print("╠══════════════╦════════════════╦════════════════╦════════════════════╣")
    print("║ Policy       ║ Paper Result   ║ Our Result     ║ Within Range?      ║")
    print("╠══════════════╬════════════════╬════════════════╬════════════════════╣")

    for name, data in comparison.items():
        paper_str = f"{data['paper_mean']:7.0f} ± {data['paper_std']:<4.0f}"
        our_str = f"{data['our_mean']:7.0f} ± {data['our_std']:<4.0f}"
        delta_str = f"{'✓' if data['within_range'] else '✗'} ({data['delta_pct']:.0f}%)"
        print(f"║ {name:12s} ║ {paper_str:14s} ║ {our_str:14s} ║ {delta_str:18s} ║")

    print("╚══════════════╩════════════════╩════════════════╩════════════════════╝")
    print("Note: PPO implemented with SB3 (original used Ray/RLlib).")
    print("      A2C used as synchronous substitute for A3C.")
    print('      "Within Range" = our mean within ±20% of paper mean.')
    print()


def print_phase2_table(
    regime_results: dict[str, Any], regime_name: str
) -> None:
    """
    Print Phase 2 disruption results table for one regime.

    Expected format of `regime_results`:
    {
        "sq":        {"mean_profit": ..., "mean_service_level": ..., "mean_stockout_rate": ..., "mean_disruption_cost": ...},
        "ppo_blind": {...},
        "ppo_aware": {...},
        "ppo_llm":   {...},
    }
    """
    display_name = regime_name.upper().replace("_", " ")

    # Header map
    policy_names = {
        "sq": "(s,Q)",
        "ppo_blind": "PPO Blind",
        "ppo_aware": "PPO Disrpt-Aware",
        "ppo_llm": "PPO LLM-Augmented",
    }

    print()
    print(f"╔══════════════════════════════════════════════════════════════════════════════╗")
    print(f"║  PHASE 2: DISRUPTION EXTENSION — {display_name:42s} ║")
    print(f"╠════════════════════╦═══════════╦═══════════════╦═════════════╦═════════════╣")
    print(f"║ Policy             ║ Avg Prof  ║ Service Level ║ Stockout %  ║ Disrpt Cost ║")
    print(f"╠════════════════════╬═══════════╬═══════════════╬═════════════╬═════════════╣")

    for key in ["sq", "ppo_blind", "ppo_aware", "ppo_llm"]:
        if key not in regime_results:
            continue
        m = regime_results[key]
        name = policy_names.get(key, key)
        profit = f"{m['mean_profit']:9.1f}"
        service = f"{m['mean_service_level']:13.3f}"
        stockout = f"{m['mean_stockout_rate'] * 100:10.1f}%"
        disrpt = f"{m.get('mean_disruption_cost', 0.0):11.1f}"
        print(f"║ {name:18s} ║ {profit} ║ {service} ║ {stockout} ║ {disrpt} ║")

    print(f"╚════════════════════╩═══════════╩═══════════════╩═════════════╩═════════════╝")
    print()


def print_stress_table(stress_results: dict[str, Any]) -> None:
    """Print stress test results (same format as Phase 2)."""
    if "stress_test" in stress_results:
        print_phase2_table(stress_results["stress_test"], "stress_test")
    else:
        for regime, data in stress_results.items():
            print_phase2_table(data, regime)


def save_results_csv(
    phase1_results: dict[str, Any] | None = None,
    phase2_results: dict[str, Any] | None = None,
) -> None:
    """Save all results to CSV files for further analysis."""
    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1
    if phase1_results and "comparison" in phase1_results:
        csv_path = output_dir / "phase1_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Policy", "Paper_Mean", "Paper_Std",
                "Our_Mean", "Our_Std", "Delta_Pct", "Within_Range",
            ])
            for name, data in phase1_results["comparison"].items():
                writer.writerow([
                    name, data["paper_mean"], data["paper_std"],
                    f"{data['our_mean']:.1f}", f"{data['our_std']:.1f}",
                    f"{data['delta_pct']:.1f}", data["within_range"],
                ])
        print(f"  Phase 1 results saved to {csv_path}")

    # Phase 2
    if phase2_results:
        csv_path = output_dir / "phase2_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Regime", "Policy", "Mean_Profit", "Std_Profit",
                "Service_Level", "Stockout_Rate", "Disruption_Cost",
            ])

            policy_names = {
                "sq": "(s,Q)",
                "ppo_blind": "PPO Blind",
                "ppo_aware": "PPO Disrupt-Aware",
                "ppo_llm": "PPO LLM-Augmented",
            }

            for regime, regime_data in phase2_results.items():
                for key in ["sq", "ppo_blind", "ppo_aware", "ppo_llm"]:
                    if key not in regime_data:
                        continue
                    m = regime_data[key]
                    writer.writerow([
                        regime,
                        policy_names.get(key, key),
                        f"{m['mean_profit']:.1f}",
                        f"{m['std_profit']:.1f}",
                        f"{m['mean_service_level']:.4f}",
                        f"{m['mean_stockout_rate']:.4f}",
                        f"{m.get('mean_disruption_cost', 0.0):.1f}",
                    ])
        print(f"  Phase 2 results saved to {csv_path}")
