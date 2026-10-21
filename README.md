# GF-6VIs

GF-6VIs 是把原来的高分六号指数计算、光谱重采样、表格相关性分析和拟合绘图脚本整理后的 Python 项目。目标是把硬编码路径和重复公式移除，改成统一包函数和命令行工具。

## 功能

- GF-6 多波段 GeoTIFF 植被指数计算，支持 NDVI 阈值掩膜。
- 影像指数与真值栅格随机采样并导出 Excel。
- GF-6 响应函数光谱重采样，支持 Excel 光谱表和 ENVI 光谱库。
- Excel 表格植被指数计算、相关系数、线性 R2、变异系数和波段组合。
- 指数与目标变量的密度散点拟合图，自动优选线性、二次、对数、指数和幂函数模型。

## 安装

```powershell
git clone https://github.com/Forest571/GF-6VIs.git
cd GF-6VIs
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e '.[dev]'
```

如果需要处理 GeoTIFF：

```powershell
python -m pip install -e '.[raster]'
```

如果需要读取 ENVI 光谱库：

```powershell
python -m pip install -e '.[spectral]'
```

## 快速使用

计算 Excel 表格中的指数：

```powershell
gf6vis indices .\data\sample.xlsx .\outputs\indices.xlsx --sheet 总3 --append
```

计算目标列与指数的相关系数和线性 R2：

```powershell
gf6vis correlate .\data\sample.xlsx .\outputs\corr.xlsx --sheet 总3 --targets CAB,LAI
```

生成指数与目标变量的拟合密度图：

```powershell
gf6vis fit .\data\sample.xlsx .\outputs\fit --sheet 总3 --target CAB
```

计算 GeoTIFF 指数：

```powershell
gf6vis raster .\data\gf6.tif .\outputs\raster --ndvi-threshold 0.3
```

默认影像波段映射沿用原脚本：`Blue=1,Red=3,NIR=4,RE1=5,RE2=6`。如影像波段顺序不同：

```powershell
gf6vis raster .\data\gf6.tif .\outputs\raster --band-map Blue=1,Red=4,NIR=5,RE1=6,RE2=7
```

采样指数值和真值影像：

```powershell
gf6vis sample-truth .\data\gf6.tif .\data\truth.tif .\outputs\samples.xlsx --sample-size 5000 --seed 42
```

使用响应函数重采样光谱：

```powershell
gf6vis resample .\data\GF-6响应函数.xlsx .\data\spectra.xlsx .\outputs\resampled.csv --spectrum-sheet 2
```

生成波段组合并计算相关性：

```powershell
gf6vis combine .\data\sample.xlsx .\outputs\band_ratio.csv --sheet Sheet1 --bands Blue,Green,Red,NIR,RE1,RE2 --targets CAB,LAI
```

## API 示例

```python
import pandas as pd
from gf6vis import calculate_indices

df = pd.read_excel("sample.xlsx", sheet_name="总3")
indices = calculate_indices(df)
indices.to_excel("indices.xlsx", index=False)
```

## 公式列名

表格指数计算要求至少包含这些列：

- `Blue`
- `Red`
- `NIR`
- `RE1`
- `RE2`

内置指数包括：

`NDVI`, `MTCI`, `NDRE1`, `NDVIRE`, `RERNDVI`, `CSI`, `BNDRE1`, `NCSI`, `CIRE2`, `IRECI`。

影像计算默认包含原脚本里的经验换算指数：`real_BNDRE1`, `real_CSI`, `real_MTCI`, `real_NCSI`。

## 开发

```powershell
python -m pip install -e '.[dev]'
pytest
ruff check .
```

## 说明

当前为 `0.1.0` Alpha 版本。已运行 5 项单元测试和代码检查；尚未完成真实 GeoTIFF、ENVI 光谱库的端到端验证。

- 栅格采样前需要自行保证影像与真值的坐标系、分辨率及像元网格一致；当前程序不会自动配准。
- 栅格处理一次性读取整幅影像；大影像可能需要较多内存，输入 NoData 也需预先核查。
- 模型优选使用拟合样本的 R2，不代表独立预测精度；对数和幂函数只使用正值样本。
- `real_*` 系数沿用旧脚本，仅适用于经验证的原标定条件。

仓库尚未指定开源许可证，公开可见不等于授予开源使用许可。版本说明见 [CHANGELOG.md](CHANGELOG.md)。
