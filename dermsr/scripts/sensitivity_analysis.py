#!/usr/bin/env python3
"""Reproduce the manuscript's sensitivity-analysis table (Table 5).

Usage
-----
    python scripts/sensitivity_analysis.py
"""
from __future__ import annotations

import pandas as pd

from dermsr.datasets import load_cifar10_benchmark
from dermsr.sensitivity import sensitivity_analysis
from dermsr.tables import table_derms_results


def main() -> None:
    pd.set_option("display.width", 120)

    benchmark = load_cifar10_benchmark()
    derms_table = table_derms_results(benchmark)

    # Hold each architecture's accuracy/std/params fixed at its baseline
    # optimal ratio, then vary only (alpha, beta, gamma, delta).
    operating_points = derms_table.merge(
        benchmark,
        left_on=["architecture", "optimal_ratio"],
        right_on=["architecture", "dataset_ratio"],
    )[["architecture", "mean_val_accuracy", "std_val_accuracy", "n_parameters", "dataset_ratio"]]

    ranks = sensitivity_analysis(operating_points)

    print("=" * 78)
    print("Table 5: Architecture ranks under alternative DERMS-R parameterizations")
    print("=" * 78)
    print(ranks.to_string())

    print()
    for scenario in ranks.columns:
        if scenario == "Baseline":
            continue
        shift = (ranks[scenario] - ranks["Baseline"]).abs()
        print(f"{scenario:<20} max |rank shift| = {shift.max():>2}  "
              f"(architecture: {shift.idxmax()})  mean |rank shift| = {shift.mean():.2f}")


if __name__ == "__main__":
    main()
