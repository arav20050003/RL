# main.py
# STATUS: Single entry point — orchestrates Phase 1 + Phase 2 experiments
# DEPENDS ON: everything
# TEST: python main.py --phase 1 --quick-test

"""
LLM-Augmented RL for Disruption-Resilient Inventory Control
============================================================

Main entry point for all experiments.

Usage:
    python main.py --phase 1                # Phase 1 replication only
    python main.py --phase 2                # Phase 2 disruption extension
    python main.py --phase both             # Full pipeline
    python main.py --phase 1 --quick-test   # Smoke test (5k steps)
    python main.py --eval-only              # Load saved models, skip training
    python main.py --seed 123               # Override random seed
    python main.py --regime stress_test     # Phase 2 specific regime only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MODELS_DIR, PLOTS_DIR, RESULTS_DIR, SEED


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LLM-Augmented RL for Disruption-Resilient Inventory Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --phase 1                 # Replicate Stranieri et al.
    python main.py --phase 2                 # Run novel LLM-aug experiments
    python main.py --phase both              # Run everything end to end
    python main.py --phase 1 --quick-test    # 5k timesteps smoke test
    python main.py --eval-only               # Skip training, load saved models
    python main.py --seed 123                # Override seed
    python main.py --regime stress_test      # Phase 2 specific regime only
        """,
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="both",
        choices=["1", "2", "both"],
        help="Which phase to run: 1 (replication), 2 (extension), or both",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Use reduced timesteps (5k) for smoke testing",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training, load saved models and evaluate only",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override global random seed",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total training timesteps",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default=None,
        help="Run Phase 2 for a specific regime only (e.g., stress_test)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Override seed if specified
    if args.seed is not None:
        import config
        config.SEED = args.seed

    start_time = time.time()

    print()
    print("══════════════════════════════════════════════════════════════")
    print("  LLM-Augmented RL for Disruption-Resilient Inventory Control")
    print("══════════════════════════════════════════════════════════════")
    print(f"  Phase:      {args.phase}")
    print(f"  Seed:       {args.seed or SEED}")
    print(f"  Quick test: {args.quick_test}")
    print(f"  Eval only:  {args.eval_only}")
    if args.regime:
        print(f"  Regime:     {args.regime}")
    print("══════════════════════════════════════════════════════════════")

    phase1_results = None
    phase2_results = None
    stress_results = None

    # ── Phase 1 ────────────────────────────────────────────────────────────
    if args.phase in ("1", "both"):
        from experiments.phase1_replicate import run_phase1
        from results.tables import print_phase1_table

        phase1_results = run_phase1(
            total_timesteps=args.timesteps,
            eval_only=args.eval_only,
            quick_test=args.quick_test,
        )
        print_phase1_table(phase1_results["comparison"])

    # ── Phase 2 ────────────────────────────────────────────────────────────
    if args.phase in ("2", "both"):
        from experiments.phase2_extend import run_phase2
        from experiments.stress_test import run_stress_test
        from results.tables import print_phase2_table, print_stress_table

        if args.regime:
            # Run specific regime only
            if args.regime == "stress_test":
                stress_results = run_stress_test(
                    total_timesteps=args.timesteps,
                    eval_only=args.eval_only,
                    quick_test=args.quick_test,
                )
                print_stress_table(stress_results)
            else:
                phase2_results = run_phase2(
                    regimes=[args.regime],
                    total_timesteps=args.timesteps,
                    eval_only=args.eval_only,
                    quick_test=args.quick_test,
                )
                for regime, data in phase2_results.items():
                    print_phase2_table(data, regime)
        else:
            # Run all Phase 2 regimes
            phase2_results = run_phase2(
                total_timesteps=args.timesteps,
                eval_only=args.eval_only,
                quick_test=args.quick_test,
            )
            for regime, data in phase2_results.items():
                print_phase2_table(data, regime)

            # Also run stress test
            stress_results = run_stress_test(
                total_timesteps=args.timesteps,
                eval_only=args.eval_only,
                quick_test=args.quick_test,
            )
            print_stress_table(stress_results)

    # ── Generate plots ─────────────────────────────────────────────────────
    from results.plots import generate_all_plots

    # Merge Phase 2 + stress results for plotting
    all_phase2 = {}
    if phase2_results:
        all_phase2.update(phase2_results)
    if stress_results:
        all_phase2.update(stress_results)

    generate_all_plots(
        phase1_results=phase1_results,
        phase2_results=all_phase2 if all_phase2 else None,
    )

    # ── Save results ───────────────────────────────────────────────────────
    from results.tables import save_results_csv

    save_results_csv(
        phase1_results=phase1_results,
        phase2_results=all_phase2 if all_phase2 else None,
    )

    # ── Final summary ──────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print()
    print("══════════════════════════════════════════════")
    print("  THESIS PROJECT — EXECUTION COMPLETE")
    print(f"  Elapsed time: {elapsed / 60:.1f} minutes")
    print(f"  Models saved to: {MODELS_DIR}")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"  Plots saved to:  {PLOTS_DIR}")
    print("══════════════════════════════════════════════")


if __name__ == "__main__":
    main()
