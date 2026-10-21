from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_filename(name: object) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", str(name))


def split_csv(value: str | Sequence[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() == "all":
            return None
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def read_excel_clean(file_path: str | Path, sheet_name: str | int | None = 0) -> pd.DataFrame:
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    data.columns = data.columns.astype(str).str.strip()
    return data


def write_table(data: pd.DataFrame, output_path: str | Path, index: bool = False) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    suffix = output.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        data.to_excel(output, index=index)
    elif suffix == ".csv":
        data.to_csv(output, index=index, encoding="utf-8-sig")
    else:
        raise ValueError("Output path must end with .xlsx, .xls, or .csv")

    return output


def find_columns(df: pd.DataFrame, keywords: Iterable[str], fuzzy: bool = True) -> list[str]:
    matched: list[str] = []
    for keyword in keywords:
        if fuzzy:
            hits = [
                column
                for column in df.columns
                if str(keyword).lower() in str(column).lower()
            ]
        else:
            hits = [str(keyword)] if str(keyword) in df.columns else []

        if not hits:
            raise ValueError(f"Column not found: {keyword}")

        matched.extend(hits)

    return list(dict.fromkeys(matched))


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
