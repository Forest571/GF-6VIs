from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from gf6vis.indices import available_indices, calculate_indices_from_bands
from gf6vis.utils import ensure_dir, safe_filename

DEFAULT_RASTER_BAND_MAP = {
    "Blue": 1,
    "Red": 3,
    "NIR": 4,
    "RE1": 5,
    "RE2": 6,
}


def parse_band_map(value: str | None) -> dict[str, int]:
    if not value:
        return DEFAULT_RASTER_BAND_MAP.copy()

    result: dict[str, int] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        key, sep, band = item.partition("=")
        if not sep:
            raise ValueError(f"Invalid band map item: {item}")
        result[key.strip()] = int(band.strip())
    return {**DEFAULT_RASTER_BAND_MAP, **result}


def _read_required_bands(src, band_map: Mapping[str, int]) -> dict[str, np.ndarray]:
    band_count = src.count
    bands: dict[str, np.ndarray] = {}
    for name, number in band_map.items():
        if number < 1 or number > band_count:
            raise ValueError(
                f"Band {name}={number} is outside raster band range 1-{band_count}"
            )
        bands[name] = src.read(number).astype("float32")
    return bands


def calculate_raster_indices(
    input_path: str | Path,
    output_dir: str | Path,
    indices: list[str] | None = None,
    band_map: Mapping[str, int] | None = None,
    ndvi_threshold: float | None = None,
    nodata: float = 0,
) -> list[Path]:
    import rasterio

    input_path = Path(input_path)
    output_dir = ensure_dir(output_dir)
    band_map = band_map or DEFAULT_RASTER_BAND_MAP
    selected_indices = indices or available_indices(include_calibrated=True)

    if ndvi_threshold is not None and "NDVI" not in selected_indices:
        selected_indices = ["NDVI", *selected_indices]

    written: list[Path] = []
    with rasterio.open(input_path) as src:
        bands = _read_required_bands(src, band_map)
        meta = src.meta.copy()
        meta.update(dtype=rasterio.float32, count=1, nodata=nodata)

        with np.errstate(divide="ignore", invalid="ignore"):
            results = calculate_indices_from_bands(
                bands,
                indices=selected_indices,
                include_calibrated=True,
            )

        ndvi = results.get("NDVI")
        for name, values in results.items():
            values = np.nan_to_num(values, nan=nodata, posinf=nodata, neginf=nodata)
            if ndvi_threshold is not None and name != "NDVI":
                values = np.where(ndvi >= ndvi_threshold, values, nodata)

            suffix = f"{safe_filename(name)}"
            if ndvi_threshold is not None and name != "NDVI":
                suffix = f"{suffix}_ndvi{ndvi_threshold:g}"

            output_path = output_dir / f"{input_path.stem}_{suffix}.tif"
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(values.astype("float32"), 1)
            written.append(output_path)

    return written


def sample_indices_with_truth(
    input_path: str | Path,
    truth_path: str | Path,
    output_path: str | Path,
    indices: list[str] | None = None,
    band_map: Mapping[str, int] | None = None,
    ndvi_threshold: float = 0.3,
    sample_size: int = 5000,
    random_seed: int | None = None,
) -> pd.DataFrame:
    import rasterio

    input_path = Path(input_path)
    output_path = Path(output_path)
    band_map = band_map or DEFAULT_RASTER_BAND_MAP
    selected_indices = indices or available_indices(include_calibrated=True)
    if "NDVI" not in selected_indices:
        selected_indices = ["NDVI", *selected_indices]

    with rasterio.open(truth_path) as truth_src:
        truth = truth_src.read(1)

    with rasterio.open(input_path) as src:
        bands = _read_required_bands(src, band_map)
        with np.errstate(divide="ignore", invalid="ignore"):
            results = calculate_indices_from_bands(
                bands,
                indices=selected_indices,
                include_calibrated=True,
            )

    height = min(truth.shape[0], next(iter(results.values())).shape[0])
    width = min(truth.shape[1], next(iter(results.values())).shape[1])
    ndvi = results["NDVI"][:height, :width]
    rows, cols = np.where(ndvi >= ndvi_threshold)

    if len(rows) == 0:
        raise ValueError("No pixels passed the NDVI threshold")

    rng = np.random.default_rng(random_seed)
    take = min(sample_size, len(rows))
    chosen = rng.choice(len(rows), size=take, replace=False)

    records: list[dict[str, float | int]] = []
    for row, col in zip(rows[chosen], cols[chosen], strict=True):
        record: dict[str, float | int] = {"Row": int(row), "Col": int(col)}
        for name, values in results.items():
            record[name] = float(values[row, col])
        record["True Value"] = float(truth[row, col])
        records.append(record)

    data = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_excel(output_path, index=False)
    return data
