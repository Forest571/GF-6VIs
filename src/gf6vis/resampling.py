from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from gf6vis.indices import DEFAULT_BANDS


def normalize_wavelength_unit(wavelengths) -> pd.Series:
    values = pd.to_numeric(wavelengths, errors="coerce").astype(float)
    if np.nanmax(values) > 10:
        values = values / 1000
    return values


def read_response_function(
    file_path: str | Path,
    sheet_name: str | int = 0,
    band_names: Sequence[str] = DEFAULT_BANDS,
) -> pd.DataFrame:
    response_raw = pd.read_excel(file_path, sheet_name=sheet_name)
    response_raw.columns = response_raw.columns.astype(str).str.strip()

    wavelength_col = response_raw.columns[0]
    wavelengths = normalize_wavelength_unit(response_raw[wavelength_col])
    values = response_raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")

    if values.shape[1] != len(band_names):
        raise ValueError(
            f"Response has {values.shape[1]} band columns, "
            f"but {len(band_names)} band names were provided"
        )

    values.columns = list(band_names)
    values.index = wavelengths
    return values.sort_index().dropna(how="all")


def read_excel_spectrum(file_path: str | Path, sheet_name: str | int = 0) -> pd.DataFrame:
    spectrum = pd.read_excel(file_path, sheet_name=sheet_name)
    spectrum.columns = spectrum.columns.astype(str).str.strip()

    wavelength_col = spectrum.columns[0]
    wavelengths = normalize_wavelength_unit(spectrum[wavelength_col])
    values = spectrum.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    values.index = wavelengths
    return values.sort_index()


def read_envi_spectrum(hdr_file: str | Path) -> pd.DataFrame:
    import spectral as spl

    spec_lib = spl.envi.open(str(hdr_file))
    wavelengths = normalize_wavelength_unit(spec_lib.bands.centers)
    spectrum = pd.DataFrame(
        data=np.asarray(spec_lib.spectra).T,
        index=wavelengths,
        columns=spec_lib.names,
    )
    return spectrum.sort_index()


def select_wavelength_range(
    spectrum: pd.DataFrame,
    min_wave: float | None = None,
    max_wave: float | None = None,
) -> pd.DataFrame:
    selected = spectrum
    if min_wave is not None:
        selected = selected.loc[selected.index >= min_wave]
    if max_wave is not None:
        selected = selected.loc[selected.index <= max_wave]
    if selected.empty:
        raise ValueError("No spectrum rows remain after wavelength filtering")
    return selected


def align_response_to_spectrum(response: pd.DataFrame, spectrum: pd.DataFrame) -> pd.DataFrame:
    aligned = pd.DataFrame(index=spectrum.index)
    for band in response.columns:
        aligned[band] = np.interp(
            spectrum.index,
            response.index,
            response[band],
            left=0,
            right=0,
        )
    return aligned


def calculate_band_response(spectrum: pd.DataFrame, response: pd.DataFrame) -> pd.DataFrame:
    weight_sum = response.sum(axis=0)
    if (weight_sum == 0).any():
        zero_bands = weight_sum[weight_sum == 0].index.tolist()
        raise ValueError(f"Response weights sum to zero for bands: {zero_bands}")

    result = spectrum.T.dot(response).div(weight_sum, axis=1)
    result = result[response.columns]
    result.index.name = "Sample"
    return result


def resample_spectrum(
    response_file: str | Path,
    spectrum_file: str | Path,
    output_file: str | Path,
    source: str = "excel",
    response_sheet: str | int = 0,
    spectrum_sheet: str | int = 0,
    wavelength_min: float | None = 0.4,
    wavelength_max: float | None = 1.0,
    band_names: Sequence[str] = DEFAULT_BANDS,
) -> pd.DataFrame:
    response = read_response_function(response_file, response_sheet, band_names)

    if source.lower() == "excel":
        spectrum = read_excel_spectrum(spectrum_file, spectrum_sheet)
    elif source.lower() == "envi":
        spectrum = read_envi_spectrum(spectrum_file)
    else:
        raise ValueError("source must be either 'excel' or 'envi'")

    spectrum = select_wavelength_range(spectrum, wavelength_min, wavelength_max)
    aligned = align_response_to_spectrum(response, spectrum)
    result = calculate_band_response(spectrum, aligned)

    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, encoding="utf-8-sig")
    return result
