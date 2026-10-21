from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


@dataclass
class FitResult:
    model_type: str
    model: object
    x_fit: np.ndarray
    y_fit: np.ndarray
    y_pred: np.ndarray
    equation: str
    r2: float
    rmse: float
    mae: float

    @property
    def n(self) -> int:
        return int(len(self.x_fit))


def clean_xy(x, y) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    return x_arr[mask], y_arr[mask]


def _finish(model_type: str, model: object, x, y, y_pred, equation: str) -> FitResult:
    return FitResult(
        model_type=model_type,
        model=model,
        x_fit=x,
        y_fit=y,
        y_pred=y_pred,
        equation=equation,
        r2=r2_score(y, y_pred),
        rmse=float(np.sqrt(mean_squared_error(y, y_pred))),
        mae=float(mean_absolute_error(y, y_pred)),
    )


def fit_linear(x: np.ndarray, y: np.ndarray) -> FitResult:
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    equation = f"y = {model.coef_[0]:.6g}x {model.intercept_:+.6g}"
    return _finish("linear", model, x, y, y_pred, equation)


def fit_quadratic(x: np.ndarray, y: np.ndarray) -> FitResult:
    poly = PolynomialFeatures(degree=2)
    x_poly = poly.fit_transform(x.reshape(-1, 1))
    model = LinearRegression().fit(x_poly, y)
    y_pred = model.predict(x_poly)
    equation = (
        f"y = {model.coef_[2]:.6g}x^2 "
        f"{model.coef_[1]:+.6g}x {model.intercept_:+.6g}"
    )
    return _finish("quadratic", (model, poly), x, y, y_pred, equation)


def fit_logarithmic(x: np.ndarray, y: np.ndarray) -> FitResult:
    mask = x > 0
    x_fit = x[mask]
    y_fit = y[mask]
    if len(x_fit) < 2:
        raise ValueError("Not enough positive x values for logarithmic fit")
    model = LinearRegression().fit(np.log(x_fit).reshape(-1, 1), y_fit)
    y_pred = model.predict(np.log(x_fit).reshape(-1, 1))
    equation = f"y = {model.coef_[0]:.6g}ln(x) {model.intercept_:+.6g}"
    return _finish("logarithmic", model, x_fit, y_fit, y_pred, equation)


def fit_exponential(x: np.ndarray, y: np.ndarray) -> FitResult:
    def func(x_value, a, b):
        return a * np.exp(b * x_value)

    popt, _ = curve_fit(func, x, y, maxfev=10000)
    y_pred = func(x, *popt)
    equation = f"y = {popt[0]:.6g}e^({popt[1]:.6g}x)"
    return _finish("exponential", tuple(popt), x, y, y_pred, equation)


def fit_power_law(x: np.ndarray, y: np.ndarray) -> FitResult:
    mask = x > 0
    x_fit = x[mask]
    y_fit = y[mask]
    if len(x_fit) < 2:
        raise ValueError("Not enough positive x values for power-law fit")

    def func(x_value, a, b):
        return a * x_value**b

    popt, _ = curve_fit(func, x_fit, y_fit, maxfev=10000)
    y_pred = func(x_fit, *popt)
    equation = f"y = {popt[0]:.6g}x^{popt[1]:.6g}"
    return _finish("power_law", tuple(popt), x_fit, y_fit, y_pred, equation)


MODEL_FITTERS: dict[str, Callable[[np.ndarray, np.ndarray], FitResult]] = {
    "linear": fit_linear,
    "quadratic": fit_quadratic,
    "logarithmic": fit_logarithmic,
    "exponential": fit_exponential,
    "power_law": fit_power_law,
}


def fit_best_model(
    x,
    y,
    models: list[str] | None = None,
    min_samples: int = 3,
) -> FitResult:
    x_clean, y_clean = clean_xy(x, y)
    if len(x_clean) < min_samples:
        raise ValueError(f"Need at least {min_samples} valid samples")

    selected = models or list(MODEL_FITTERS)
    candidates: list[FitResult] = []
    for model_type in selected:
        fitter = MODEL_FITTERS.get(model_type)
        if fitter is None:
            raise ValueError(f"Unknown model type: {model_type}")
        try:
            candidates.append(fitter(x_clean, y_clean))
        except Exception:
            continue

    if not candidates:
        raise ValueError("All model fits failed")
    return max(candidates, key=lambda result: result.r2)


def predict_curve(result: FitResult, x_values) -> np.ndarray:
    x_seq = np.asarray(x_values, dtype=float)

    if result.model_type == "linear":
        return result.model.predict(x_seq.reshape(-1, 1))
    if result.model_type == "quadratic":
        model, poly = result.model
        return model.predict(poly.transform(x_seq.reshape(-1, 1)))
    if result.model_type == "logarithmic":
        return result.model.predict(np.log(x_seq).reshape(-1, 1))
    if result.model_type == "exponential":
        a, b = result.model
        return a * np.exp(b * x_seq)
    if result.model_type == "power_law":
        a, b = result.model
        return a * x_seq**b

    raise ValueError(f"Unknown model type: {result.model_type}")
