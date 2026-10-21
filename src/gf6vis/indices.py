from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from gf6vis.utils import require_columns

EPSILON = 1e-10

DEFAULT_BANDS = ("Blue", "Green", "Red", "NIR", "RE1", "RE2", "High Blue", "Yellow")
REQUIRED_INDEX_BANDS = ("Blue", "Red", "NIR", "RE1", "RE2")

BASE_INDEX_NAMES = (
    "NDVI",
    "MTCI",
    "NDRE1",
    "NDVIRE",
    "RERNDVI",
    "CSI",
    "BNDRE1",
    "NCSI",
    "CIRE2",
    "IRECI",
)
CALIBRATED_INDEX_NAMES = ("real_BNDRE1", "real_CSI", "real_MTCI", "real_NCSI")


def available_indices(include_calibrated: bool = True) -> list[str]:
    names = list(BASE_INDEX_NAMES)
    if include_calibrated:
        names.extend(CALIBRATED_INDEX_NAMES)
    return names


def safe_divide(numerator: Any, denominator: Any, eps: float = EPSILON) -> Any:
    denominator = np.where(np.abs(denominator) > eps, denominator, np.nan)
    return numerator / denominator


def _sqrt_ratio(numerator: Any, denominator: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = safe_divide(numerator, denominator)
        return np.sqrt(ratio)


def _compute_all(bands: Mapping[str, Any]) -> dict[str, Any]:
    blue = bands["Blue"]
    red = bands["Red"]
    nir = bands["NIR"]
    re1 = bands["RE1"]
    re2 = bands["RE2"]

    ndvi = safe_divide(nir - red, nir + red)
    mtci = safe_divide(re2 - re1, re1 - red)
    ndre1 = safe_divide(re2 - re1, re2 + re1)
    ndvire = safe_divide(nir - re1, nir + re1)
    rerndvi = ndvi * _sqrt_ratio(re2, re1)
    csi = ndvire * safe_divide(blue, re1)
    bndre1 = ndre1 * safe_divide(blue, re1)
    ncsi = csi * safe_divide(nir, re2)
    cire2 = safe_divide(re2, red) - 1
    ireci = safe_divide(nir - red, safe_divide(re1, re2))

    return {
        "NDVI": ndvi,
        "MTCI": mtci,
        "NDRE1": ndre1,
        "NDVIRE": ndvire,
        "RERNDVI": rerndvi,
        "CSI": csi,
        "BNDRE1": bndre1,
        "NCSI": ncsi,
        "CIRE2": cire2,
        "IRECI": ireci,
        "real_BNDRE1": bndre1 * 309.047 - 0.91,
        "real_CSI": csi * 280.257 + 0.879,
        "real_MTCI": mtci * 6.244 + 17.284,
        "real_NCSI": ncsi * 230.155 + 3.024,
    }


def calculate_indices_from_bands(
    bands: Mapping[str, Any],
    indices: list[str] | tuple[str, ...] | None = None,
    include_calibrated: bool = False,
) -> dict[str, Any]:
    missing = [name for name in REQUIRED_INDEX_BANDS if name not in bands]
    if missing:
        raise ValueError(f"Missing required bands: {missing}")

    computed = _compute_all(bands)
    selected = indices if indices is not None else available_indices(include_calibrated)

    unknown = [name for name in selected if name not in computed]
    if unknown:
        raise ValueError(f"Unknown indices: {unknown}")

    return {name: computed[name] for name in selected}


def calculate_indices(
    data: pd.DataFrame,
    indices: list[str] | tuple[str, ...] | None = None,
    include_calibrated: bool = False,
    append: bool = False,
) -> pd.DataFrame:
    require_columns(data, REQUIRED_INDEX_BANDS)

    bands = {name: pd.to_numeric(data[name], errors="coerce") for name in REQUIRED_INDEX_BANDS}
    computed = calculate_indices_from_bands(
        bands,
        indices=indices,
        include_calibrated=include_calibrated,
    )
    result = pd.DataFrame(computed, index=data.index)
    result = result.replace([np.inf, -np.inf], np.nan)

    if append:
        return pd.concat([data.copy(), result], axis=1)
    return result
