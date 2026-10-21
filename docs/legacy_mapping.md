# Legacy Script Mapping

This project replaces the hard-coded one-off scripts under `高分指数` with reusable modules and CLI commands.

| Legacy scripts | New location | CLI command |
| --- | --- | --- |
| `遥感影像多VI计算.py`, `NDVI筛选范围指数.py` | `gf6vis.raster` | `gf6vis raster ...` |
| `NDVI筛选真值与指数值.py` | `gf6vis.raster.sample_indices_with_truth` | `gf6vis sample-truth ...` |
| `光谱重采样.py`, `文件下所有光谱重采样.py`, `文件夹下所有数据读取.py` | `gf6vis.resampling` | `gf6vis resample ...` |
| `皮尔逊相关性分析.py`, `简单误差计算.py`, `变异系数计算.py` | `gf6vis.analysis` | `gf6vis correlate ...` and API calls |
| `拟合选优密度图.py`, `线性回归密度图.py`, `任意两列最优拟合.py` | `gf6vis.models`, `gf6vis.plots` | `gf6vis fit ...` |
| `散点图.py`, `箱线图.py` | `gf6vis.plots` | API calls; CLI can be extended as needed |
| `直方图.py`, `雷达图.py` | Not migrated yet | No equivalent API yet |
| `波段组合.py`, `三波段组合.py` | `gf6vis.analysis.combine_bands` | `gf6vis combine ...` |
| `基底函数.py` | Not fully migrated | Band combinations do not replace all basis functions |
| `多个TXT合并CSV.py`, `光谱数据转换格式.py`, `光谱数据处理.py` | Not first-class CLI yet | Keep as small data-prep helpers or add under `gf6vis.io` later |

Design notes:

- Paths are command-line arguments instead of global variables.
- Vegetation-index formulas are centralized in `gf6vis.indices`.
- Optional heavy dependencies are lazy imports: `rasterio` is only needed for raster commands, and `spectral` is only needed for ENVI spectrum input.
- Generated rasters, images, CSV files, and Excel files are ignored by Git by default.
