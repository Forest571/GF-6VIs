from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from gf6vis.indices import calculate_indices, safe_divide
from gf6vis.utils import find_columns, read_excel_clean, require_columns, write_table


def clean_pair(x: pd.Series, y: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()


def correlation_and_r2(indices: pd.DataFrame, target: pd.Series, target_name: str) -> pd.DataFrame:
    rows = []
    for index_name in indices.columns:
        pair = clean_pair(indices[index_name], target)
        if len(pair) < 2:
            rows.append(
                {
                    "Target": target_name,
                    "Index": index_name,
                    "Correlation": np.nan,
                    "R2": np.nan,
                    "N": len(pair),
                }
            )
            continue

        x = pair["x"].astype(float).to_numpy()
        y = pair["y"].astype(float).to_numpy()
        model = LinearRegression().fit(x.reshape(-1, 1), y)
        y_pred = model.predict(x.reshape(-1, 1))

        rows.append(
            {
                "Target": target_name,
                "Index": index_name,
                "Correlation": pd.Series(x).corr(pd.Series(y)),
                "R2": r2_score(y, y_pred),
                "N": len(pair),
            }
        )

    return pd.DataFrame(rows)


def run_correlation_analysis(
    input_file: str | Path,
    output_file: str | Path,
    sheet_name: str | int = 0,
    targets: list[str] | None = None,
    indices: list[str] | None = None,
    fuzzy_targets: bool = True,
) -> pd.DataFrame:
    data = read_excel_clean(input_file, sheet_name)
    vi = calculate_indices(data, indices=indices)

    if not targets:
        raise ValueError("At least one target column or keyword is required")

    target_columns = find_columns(data, targets, fuzzy=fuzzy_targets)
    results = [
        correlation_and_r2(vi, data[target_column], target_column)
        for target_column in target_columns
    ]
    result = pd.concat(results, ignore_index=True)
    write_table(result, output_file, index=False)
    return result


def coefficient_of_variation(indices: pd.DataFrame) -> pd.Series:
    return indices.std(skipna=True) / indices.mean(skipna=True)


def index_stats(indices: pd.DataFrame) -> pd.DataFrame:
    stats = indices.agg(["min", "max", "mean", "std"]).T
    stats["cv"] = coefficient_of_variation(indices)
    stats["min/max"] = safe_divide(stats["min"], stats["max"])
    return stats.reset_index(names="Index")


def combine_bands(data: pd.DataFrame, bands: list[str], mode: str = "ratio") -> pd.DataFrame:
    require_columns(data, bands)
    mode = mode.lower()
    results: dict[str, pd.Series] = {}

    if mode == "ratio":
        for band1, band2 in itertools.permutations(bands, 2):
            results[f"({band1}/{band2})"] = safe_divide(data[band1], data[band2])
    elif mode == "normalized-difference":
        for band1, band2 in itertools.permutations(bands, 2):
            results[f"({band1}-{band2})/({band1}+{band2})"] = safe_divide(
                data[band1] - data[band2],
                data[band1] + data[band2],
            )
    elif mode == "triple":
        for band1, band2, band3 in itertools.permutations(bands, 3):
            results[f"({band1}-{band2})/({band1}-{band3})"] = safe_divide(
                data[band1] - data[band2],
                data[band1] - data[band3],
            )
    else:
        raise ValueError("mode must be ratio, normalized-difference, or triple")

    return pd.DataFrame(results, index=data.index).replace([np.inf, -np.inf], np.nan)


def correlate_columns(data: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target_name in targets.columns:
        for column in data.columns:
            pair = clean_pair(data[column], targets[target_name])
            rows.append(
                {
                    "Target": target_name,
                    "Column": column,
                    "Correlation": pair["x"].corr(pair["y"]) if len(pair) >= 2 else np.nan,
                    "N": len(pair),
                }
            )
    return pd.DataFrame(rows).sort_values(["Target", "Correlation"], ascending=[True, False])
