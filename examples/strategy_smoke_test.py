#!/usr/bin/env python3
"""
Strategy Smoke Test

Runs a short, seeded simulation for a handful of attack strategies and
prints a compact summary with QBER, detection verdict, channel stats,
and Eve information gain. Useful for quick sanity checks against
expected ranges.

Usage:
    python examples/strategy_smoke_test.py --n 2000 --preset laboratory

Options:
    --n / -n           Number of qubits (default: 2000)
    --preset           One of {laboratory, urban, satellite, adversarial}
    --seed             RNG seed (default: 1337)
"""
from __future__ import annotations
import argparse
import json
import random
from typing import Dict, List

import numpy as np

# Local imports
from src.main.bb84_main import (
    AdaptiveJammingSimulator,
    ClassicalBackend,
    QBERAdaptiveStrategy,
    GradientDescentQBERAdaptive,
    BasisLearningStrategy,
    ChannelAdaptiveStrategy,
    PhotonNumberSplittingAttack,
)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_once(n_qubits: int, preset: str | None) -> List[Dict]:
    backend = ClassicalBackend()

    strategies = [
        ("QBERAdaptiveStrategy", QBERAdaptiveStrategy(backend=backend, target_qber=0.10, threshold=0.11)),
        ("GradientDescentQBERAdaptive", GradientDescentQBERAdaptive(backend=backend, target_qber=0.10, threshold=0.11, learning_rate=0.01)),
        ("BasisLearningStrategy", BasisLearningStrategy(backend=backend, base_intercept_prob=0.3, confidence_threshold=0.8)),
        ("ChannelAdaptiveStrategy", ChannelAdaptiveStrategy(backend=backend, distance_km=10.0, wavelength_nm=1550.0)),
        ("PhotonNumberSplittingAttack", PhotonNumberSplittingAttack(backend=backend, mean_photon_number=0.1)),
    ]

    summaries: List[Dict] = []

    for name, strategy in strategies:
        sim = AdaptiveJammingSimulator(
            n_qubits=n_qubits,
            attack_strategy=strategy,
            backend=backend,
            preset=preset,
        )
        results = sim.run_simulation()

        eve_info = 0.0
        if results.get("eve") and results["eve"].get("attack_strategy_stats"):
            eve_info = float(results["eve"]["attack_strategy_stats"].get("information_gained", 0.0))

        summaries.append({
            "strategy": name,
            "qber": float(results.get("qber", 0.0)),
            "detected": bool(results.get("detection", {}).get("attack_detected", False)),
            "lost": int(results.get("channel", {}).get("lost", 0)),
            "errors": int(results.get("channel", {}).get("error_events", 0)),
            "eve_info_bits": eve_info,
        })

    return summaries


def print_table(rows: List[Dict]) -> None:
    # Column headers
    headers = ["Strategy", "QBER", "Detected", "Lost", "Errors", "Eve Info (bits)"]
    # Compute column widths
    def fmt_row(r: Dict) -> List[str]:
        return [
            r["strategy"],
            f"{100*r['qber']:.2f}%",
            "Yes" if r["detected"] else "No",
            str(r["lost"]),
            str(r["errors"]),
            f"{r['eve_info_bits']:.3f}",
        ]

    rows_fmt = [fmt_row(r) for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in rows_fmt)) for i, h in enumerate(headers)]

    def print_line(parts: List[str]) -> None:
        print(" | ".join(p.ljust(widths[i]) for i, p in enumerate(parts)))

    print_line(headers)
    print("-+-".join("-" * w for w in widths))
    for r in rows_fmt:
        print_line(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a quick seeded strategy smoke test")
    parser.add_argument("--n", "-n", type=int, default=2000, help="Number of qubits")
    parser.add_argument("--preset", type=str, default=None, choices=[None, "laboratory", "urban", "satellite", "adversarial"], help="Channel preset")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    args = parser.parse_args()

    set_global_seed(args.seed)
    summaries = run_once(args.n, args.preset)

    print_table(summaries)
    # Also print JSON for machine parsing
    print("\nJSON:")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
