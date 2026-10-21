from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from gf6vis.models import FitResult, fit_best_model, predict_curve


def density_color(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return np.ones_like(x, dtype=float)
    try:
        xy = np.vstack([x, y])
        return gaussian_kde(xy)(xy)
    except Exception:
        return np.ones_like(x, dtype=float)


def plot_density_fit(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path: str | Path,
    models: list[str] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    dpi: int = 300,
) -> FitResult:
    pair = data[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    x = pair[x_col].astype(float).to_numpy()
    y = pair[y_col].astype(float).to_numpy()
    result = fit_best_model(x, y, models=models)

    z = density_color(x, y)
    order = z.argsort()
    x = x[order]
    y = y[order]
    z = z[order]

    x_min = float(np.nanmin(result.x_fit))
    x_max = float(np.nanmax(result.x_fit))
    if result.model_type in {"logarithmic", "power_law"}:
        x_min = max(x_min, 1e-10)
    x_seq = np.linspace(x_min, x_max, 200)
    y_seq = predict_curve(result, x_seq)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    ax.scatter(x, y, c=z, s=30, cmap="viridis_r", edgecolor="none", alpha=0.9)
    ax.plot(x_seq, y_seq, color="red", linestyle="--", linewidth=2)
    ax.text(
        0.05,
        0.95,
        (
            f"{result.equation}\n"
            f"R2 = {result.r2:.3f}\n"
            f"RMSE = {result.rmse:.3f}\n"
            f"MAE = {result.mae:.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
    )
    ax.set_xlabel(x_label or x_col)
    ax.set_ylabel(y_label or y_col)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return result


def plot_scatter(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path: str | Path,
    dpi: int = 300,
) -> None:
    pair = data[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.scatter(pair[x_col], pair[y_col], color="red", alpha=0.5, s=30)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_box(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    bins: list[float],
    labels: list[str],
    output_path: str | Path,
    dpi: int = 300,
) -> None:
    frame = data.copy()
    frame["bin"] = pd.cut(frame[x_col], bins=bins, labels=labels, include_lowest=True)
    grouped = [frame.loc[frame["bin"] == label, y_col].dropna() for label in labels]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    ax.boxplot(
        grouped,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "#90ba48", "alpha": 1},
        medianprops={"color": "black", "linewidth": 2},
    )
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
