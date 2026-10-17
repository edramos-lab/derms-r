#!/usr/bin/env python3
"""Reproduce the manuscript's Friedman / Iman-Davenport / Nemenyi analysis.

Usage
-----
    python scripts/statistical_analysis.py
"""
from __future__ import annotations

from dermsr.datasets import load_cifar10_benchmark
from dermsr.statistics import critical_pairs, friedman_test, nemenyi_critical_difference
from dermsr.tables import table_rank_matrix


def main() -> None:
    benchmark = load_cifar10_benchmark()

    # Blocks = the 5 dataset-utilization ratios, treatments = the 12
    # architectures -- see the paper's "Important Statistical Unit"
    # discussion for why cross-validation folds are not used as blocks.
    ranks = table_rank_matrix(benchmark).drop(columns="average_rank")

    result = friedman_test(ranks)
    print(f"Friedman chi2_F = {result.chi2_f:.4f}  (df={result.df}, k={result.k}, N={result.n})")
    print(f"Iman-Davenport F_F = {result.f_f:.4f}  (df1={result.f_df1}, df2={result.f_df2})")
    if result.p_value is not None:
        print(f"chi2 p-value = {result.p_value:.6g}  (scipy available)")
    else:
        print("scipy unavailable (not installed, or failed to import); compare chi2_F "
              f"against a critical-value table instead, e.g. chi2(0.05, df={result.df}) ~ 19.68 for df=11.")

    print()
    print("Average ranks (ascending = better):")
    print(result.average_ranks.to_string())

    cd = nemenyi_critical_difference(k=result.k, n=result.n)
    print()
    print(f"Nemenyi critical difference (alpha=0.05) = {cd:.4f}")

    pairs = critical_pairs(result.average_ranks, cd)
    print(f"\n{len(pairs)} of {result.k * (result.k - 1) // 2} pairs exceed the critical difference:")
    print(pairs.to_string(index=False))


if __name__ == "__main__":
    main()
