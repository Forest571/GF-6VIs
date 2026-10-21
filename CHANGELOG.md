# Changelog

## 0.1.0

Initial alpha release of the GF-6VIs Python package.

- Centralized vegetation-index formulas and configurable file paths.
- Added CLI commands for table indices, correlations, band combinations, model fitting, raster indices, truth sampling, and spectral resampling.
- Added optional raster and ENVI dependencies, installation documentation, legacy-script mapping, unit tests, and GitHub Actions CI.

### Validation and limitations

- Five unit tests and Ruff checks pass locally on Python 3.12.
- Real GeoTIFF and ENVI workflows have not been validated end to end.
- Raster truth sampling assumes pre-aligned grids; input NoData handling needs manual verification.
- Raster operations read full images into memory.
- Model ranking uses in-sample R2; logarithmic and power fits can use different positive-value subsets.
- Legacy calibration coefficients require validation for each study area.
- Some legacy plotting and data-preparation scripts are not yet migrated.
- No open-source license has been selected.
