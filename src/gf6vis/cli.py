from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gf6vis.analysis import (
    combine_bands,
    correlate_columns,
    index_stats,
    run_correlation_analysis,
)
from gf6vis.indices import calculate_indices
from gf6vis.plots import plot_density_fit
from gf6vis.raster import calculate_raster_indices, parse_band_map, sample_indices_with_truth
from gf6vis.resampling import resample_spectrum
from gf6vis.utils import read_excel_clean, safe_filename, split_csv, write_table


def parse_sheet(value: str | int) -> str | int:
    if isinstance(value, int):
        return value
    value = value.strip()
    if value.isdigit():
        return int(value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gf6vis",
        description="GF-6 vegetation-index, spectrum resampling, and analysis tools.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("indices", help="Calculate vegetation indices for an Excel table.")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--sheet", type=parse_sheet, default=0)
    p.add_argument("--indices", default="all")
    p.add_argument("--append", action="store_true")
    p.add_argument("--include-calibrated", action="store_true")
    p.set_defaults(func=cmd_indices)

    p = sub.add_parser("correlate", help="Calculate correlation and linear R2.")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--sheet", type=parse_sheet, default=0)
    p.add_argument("--targets", required=True, help="Comma-separated target columns or keywords.")
    p.add_argument("--indices", default="all")
    p.add_argument("--exact-targets", action="store_true")
    p.set_defaults(func=cmd_correlate)

    p = sub.add_parser("combine", help="Generate band combinations and optional correlations.")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--sheet", type=parse_sheet, default=0)
    p.add_argument("--bands", required=True, help="Comma-separated band columns.")
    p.add_argument(
        "--mode",
        default="ratio",
        choices=["ratio", "normalized-difference", "triple"],
    )
    p.add_argument("--targets", help="Optional comma-separated target columns.")
    p.add_argument("--correlations-output")
    p.set_defaults(func=cmd_combine)

    p = sub.add_parser("fit", help="Fit target against indices and save density plots.")
    p.add_argument("input")
    p.add_argument("output_dir")
    p.add_argument("--sheet", type=parse_sheet, default=0)
    p.add_argument("--target", required=True)
    p.add_argument("--indices", default="all")
    p.add_argument("--models", default="linear,quadratic,logarithmic,exponential,power_law")
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("raster", help="Calculate vegetation-index GeoTIFF files.")
    p.add_argument("input")
    p.add_argument("output_dir")
    p.add_argument("--indices", default="all")
    p.add_argument("--band-map", help="Example: Blue=1,Red=3,NIR=4,RE1=5,RE2=6")
    p.add_argument("--ndvi-threshold", type=float)
    p.set_defaults(func=cmd_raster)

    p = sub.add_parser("sample-truth", help="Sample index values and a truth raster to Excel.")
    p.add_argument("input")
    p.add_argument("truth")
    p.add_argument("output")
    p.add_argument("--indices", default="all")
    p.add_argument("--band-map")
    p.add_argument("--ndvi-threshold", type=float, default=0.3)
    p.add_argument("--sample-size", type=int, default=5000)
    p.add_argument("--seed", type=int)
    p.set_defaults(func=cmd_sample_truth)

    p = sub.add_parser("resample", help="Resample spectra with GF-6 response functions.")
    p.add_argument("response")
    p.add_argument("spectrum")
    p.add_argument("output")
    p.add_argument("--source", default="excel", choices=["excel", "envi"])
    p.add_argument("--response-sheet", type=parse_sheet, default=0)
    p.add_argument("--spectrum-sheet", type=parse_sheet, default=0)
    p.add_argument("--wavelength-min", type=float, default=0.4)
    p.add_argument("--wavelength-max", type=float, default=1.0)
    p.add_argument("--band-names", default=None)
    p.set_defaults(func=cmd_resample)

    return parser


def cmd_indices(args: argparse.Namespace) -> None:
    data = read_excel_clean(args.input, args.sheet)
    result = calculate_indices(
        data,
        indices=split_csv(args.indices),
        include_calibrated=args.include_calibrated,
        append=args.append,
    )
    write_table(result, args.output, index=False)
    print(f"Saved {result.shape[1]} columns to {args.output}")


def cmd_correlate(args: argparse.Namespace) -> None:
    result = run_correlation_analysis(
        args.input,
        args.output,
        sheet_name=args.sheet,
        targets=split_csv(args.targets),
        indices=split_csv(args.indices),
        fuzzy_targets=not args.exact_targets,
    )
    print(f"Saved {len(result)} rows to {args.output}")


def cmd_combine(args: argparse.Namespace) -> None:
    data = read_excel_clean(args.input, args.sheet)
    bands = split_csv(args.bands)
    if not bands:
        raise ValueError("--bands is required")
    combined = combine_bands(data, bands, mode=args.mode)
    write_table(combined, args.output, index=False)

    if args.targets:
        target_names = split_csv(args.targets) or []
        correlations = correlate_columns(combined, data[target_names])
        output = args.correlations_output or str(Path(args.output).with_name("band_correlations.xlsx"))
        write_table(correlations, output, index=False)
        print(f"Saved combinations to {args.output} and correlations to {output}")
    else:
        print(f"Saved combinations to {args.output}")


def cmd_fit(args: argparse.Namespace) -> None:
    data = read_excel_clean(args.input, args.sheet)
    indices = calculate_indices(data, indices=split_csv(args.indices))
    target = data[args.target]
    plot_data = pd.concat([indices, target.rename(args.target)], axis=1)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index_name in indices.columns:
        output = output_dir / f"{safe_filename(index_name)}_vs_{safe_filename(args.target)}.png"
        result = plot_density_fit(
            plot_data,
            x_col=index_name,
            y_col=args.target,
            output_path=output,
            models=split_csv(args.models),
        )
        rows.append(
            {
                "Index": index_name,
                "Best_Model": result.model_type,
                "Equation": result.equation,
                "R2": result.r2,
                "RMSE": result.rmse,
                "MAE": result.mae,
                "N": result.n,
                "Figure": str(output),
            }
        )

    metrics = pd.DataFrame(rows)
    metrics_path = output_dir / f"{safe_filename(args.target)}_fit_results.xlsx"
    metrics.to_excel(metrics_path, index=False)
    stats_path = output_dir / "index_stats.xlsx"
    index_stats(indices).to_excel(stats_path, index=False)
    print(f"Saved plots and metrics to {output_dir}")


def cmd_raster(args: argparse.Namespace) -> None:
    outputs = calculate_raster_indices(
        args.input,
        args.output_dir,
        indices=split_csv(args.indices),
        band_map=parse_band_map(args.band_map),
        ndvi_threshold=args.ndvi_threshold,
    )
    print(f"Saved {len(outputs)} raster files to {args.output_dir}")


def cmd_sample_truth(args: argparse.Namespace) -> None:
    data = sample_indices_with_truth(
        args.input,
        args.truth,
        args.output,
        indices=split_csv(args.indices),
        band_map=parse_band_map(args.band_map),
        ndvi_threshold=args.ndvi_threshold,
        sample_size=args.sample_size,
        random_seed=args.seed,
    )
    print(f"Saved {len(data)} samples to {args.output}")


def cmd_resample(args: argparse.Namespace) -> None:
    band_names = split_csv(args.band_names)
    kwargs = {"band_names": band_names} if band_names else {}
    result = resample_spectrum(
        args.response,
        args.spectrum,
        args.output,
        source=args.source,
        response_sheet=args.response_sheet,
        spectrum_sheet=args.spectrum_sheet,
        wavelength_min=args.wavelength_min,
        wavelength_max=args.wavelength_max,
        **kwargs,
    )
    print(f"Saved {result.shape[0]} samples x {result.shape[1]} bands to {args.output}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
