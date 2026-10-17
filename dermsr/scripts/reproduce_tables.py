#!/usr/bin/env python3
"""Reproduce the manuscript's DERMS-R tables (2-6) from the bundled benchmark.

Usage
-----
    python scripts/reproduce_tables.py
"""
from __future__ import annotations

import pandas as pd

from dermsr.datasets import load_cifar10_benchmark
from dermsr.tables import (
    table_ablation,
    table_derms_results,
    table_predictive_performance,
    table_rank_matrix,
)


def main() -> None:
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", None)

    benchmark = load_cifar10_benchmark()

    print("=" * 78)
    print("Table 2: Predictive performance at r = 1.00")
    print("=" * 78)
    print(table_predictive_performance(benchmark, ratio=1.00).to_string(index=False))

    print()
    print("=" * 78)
    print("Table 3: DERMS-R component decomposition and normalized ranking")
    print("=" * 78)
    print(table_derms_results(benchmark).to_string(index=False))

    print()
    print("=" * 78)
    print("Table 4: Rank matrix (architecture x dataset ratio)")
    print("=" * 78)
    print(table_rank_matrix(benchmark).to_string())

    print()
    print("=" * 78)
    print("Table 6: Ablation (nested DERMS-R formulations)")
    print("=" * 78)
    print(table_ablation(benchmark).to_string(index=False))


if __name__ == "__main__":
    main()
