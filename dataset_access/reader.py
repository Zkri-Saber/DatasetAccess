"""Core dataset reader — format detection, single-file and directory reading."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Map file extensions to reader functions
FORMAT_READERS = {
    ".csv": lambda p, **kw: pd.read_csv(p, **kw),
    ".tsv": lambda p, **kw: pd.read_csv(p, sep="\t", **kw),
    ".json": lambda p, **kw: pd.read_json(p, **kw),
    ".jsonl": lambda p, **kw: pd.read_json(p, lines=True, **kw),
    ".xlsx": lambda p, **kw: pd.read_excel(p, engine="openpyxl", **kw),
    ".xls": lambda p, **kw: pd.read_excel(p, **kw),
    ".parquet": lambda p, **kw: pd.read_parquet(p, **kw),
    ".feather": lambda p, **kw: pd.read_feather(p, **kw),
    ".orc": lambda p, **kw: pd.read_orc(p, **kw),
    ".hdf": lambda p, **kw: pd.read_hdf(p, **kw),
    ".h5": lambda p, **kw: pd.read_hdf(p, **kw),
    ".html": lambda p, **kw: pd.read_html(p, **kw)[0],
    ".xml": lambda p, **kw: pd.read_xml(p, **kw),
    ".fwf": lambda p, **kw: pd.read_fwf(p, **kw),
    ".pkl": lambda p, **kw: pd.read_pickle(p, **kw),
    ".sas7bdat": lambda p, **kw: pd.read_sas(p, **kw),
    ".xpt": lambda p, **kw: pd.read_sas(p, **kw),
    ".dta": lambda p, **kw: pd.read_stata(p, **kw),
    ".sav": lambda p, **kw: pd.read_spss(p, **kw),
}

SUPPORTED_EXTENSIONS = set(FORMAT_READERS.keys())


def read_dataset(source, format=None, **kwargs):
    """Read a dataset from a file path or URL into a pandas DataFrame.

    Args:
        source: File path, URL, or SQLAlchemy connection string.
        format: Explicit format override (e.g. "csv", "sql"). If None,
                the format is detected from the file extension.
        **kwargs: Extra keyword arguments forwarded to the underlying
                  pandas reader (e.g. sheet_name, sep, query).

    Returns:
        A pandas DataFrame.
    """
    if format == "sql":
        from sqlalchemy import create_engine

        query = kwargs.pop("query", None)
        if query is None:
            raise ValueError("A 'query' argument is required for SQL sources.")
        engine = create_engine(source)
        return pd.read_sql(query, engine, **kwargs)

    if format is not None:
        ext = f".{format.lstrip('.')}"
    else:
        ext = Path(source).suffix.lower()

    reader = FORMAT_READERS.get(ext)
    if reader is None:
        raise ValueError(
            f"Unsupported format '{ext}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    return reader(source, **kwargs)


def read_directory(directory, extensions=None, recursive=False):
    """Read all supported dataset files from a directory.

    Args:
        directory: Path to the directory to scan.
        extensions: Optional set/list of extensions to include
                    (e.g. {".csv", ".json"}). Defaults to all supported.
        recursive: If True, scan subdirectories as well.

    Returns:
        A dict mapping file paths (str) to DataFrames.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"'{directory}' is not a directory.")

    allowed = (
        {f".{e.lstrip('.')}" for e in extensions}
        if extensions
        else SUPPORTED_EXTENSIONS
    )

    results = {}
    pattern = "**/*" if recursive else "*"

    for filepath in sorted(directory.glob(pattern)):
        if not filepath.is_file():
            continue
        ext = filepath.suffix.lower()
        if ext not in allowed:
            continue
        try:
            df = read_dataset(str(filepath))
            results[str(filepath)] = df
        except Exception as exc:
            print(f"Warning: could not read '{filepath}': {exc}")

    if not results:
        print(f"No supported dataset files found in '{directory}'.")

    return results


def search_missing(source, **kwargs):
    """Analyze missing values in a dataset.

    Args:
        source: A DataFrame, or a file path/URL to load first.

    Returns:
        A DataFrame with columns: column, missing_count, total_count,
        missing_percent, dtype.
    """
    if isinstance(source, (str, os.PathLike)):
        df = read_dataset(str(source), **kwargs)
    else:
        df = source

    total = len(df)
    records = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        records.append(
            {
                "column": col,
                "missing_count": missing,
                "total_count": total,
                "missing_percent": round(missing / total * 100, 1) if total else 0.0,
                "dtype": str(df[col].dtype),
            }
        )

    return pd.DataFrame(records)


def _detect_target_column(df):
    """Heuristic to find the most likely target/label column.

    Checks for common names first, then falls back to the last
    categorical or low-cardinality integer column.
    """
    common_names = [
        "target", "label", "class", "y", "outcome", "category",
        "is_fraud", "survived", "diagnosis", "species",
    ]
    lower_cols = {c.lower(): c for c in df.columns}
    for name in common_names:
        if name in lower_cols:
            return lower_cols[name]

    # Fallback: last column that is categorical or has few unique values
    for col in reversed(df.columns.tolist()):
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            return col
        if pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() <= 20:
            return col

    return None


def _imbalance_ratio(series):
    """Return the imbalance ratio (majority / minority) for a Series.

    Returns None if the column has fewer than 2 unique non-null values.
    """
    counts = series.dropna().value_counts()
    if len(counts) < 2:
        return None
    return round(counts.iloc[0] / counts.iloc[-1], 2)


def summarize_dataset(df, name="dataset"):
    """Build a single-row summary dict for one DataFrame."""
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols
    total_missing = int(df.isna().sum().sum())
    missing_pct = round(total_missing / total_cells * 100, 2) if total_cells else 0.0

    numeric_df = df.select_dtypes(include="number")
    cat_df = df.select_dtypes(include=["object", "category"])

    target_col = _detect_target_column(df)
    imbalance = None
    target_classes = None
    if target_col is not None:
        imbalance = _imbalance_ratio(df[target_col])
        target_classes = int(df[target_col].nunique())

    duplicate_rows = int(df.duplicated().sum())

    summary = {
        "dataset_name": name,
        "num_instances": n_rows,
        "num_features": n_cols,
        "numeric_features": len(numeric_df.columns),
        "categorical_features": len(cat_df.columns),
        "missing_values": total_missing,
        "missing_pct": missing_pct,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round(duplicate_rows / n_rows * 100, 2) if n_rows else 0.0,
        "target_column": target_col if target_col else "N/A",
        "target_classes": target_classes if target_classes else "N/A",
        "imbalance_ratio": imbalance if imbalance else "N/A",
        "mean": round(numeric_df.mean().mean(), 4) if not numeric_df.empty else "N/A",
        "std": round(numeric_df.std().mean(), 4) if not numeric_df.empty else "N/A",
        "min": round(numeric_df.min().min(), 4) if not numeric_df.empty else "N/A",
        "max": round(numeric_df.max().max(), 4) if not numeric_df.empty else "N/A",
        "median": round(numeric_df.median().median(), 4) if not numeric_df.empty else "N/A",
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1_048_576, 3),
    }
    return summary


def summarize_datasets(directory, extensions=None, recursive=False):
    """Read all datasets in a directory and return a summary table.

    Args:
        directory: Path to the directory to scan.
        extensions: Optional set/list of extensions to include.
        recursive: If True, scan subdirectories.

    Returns:
        A pandas DataFrame where each row summarises one dataset file,
        with columns such as dataset_name, num_instances, num_features,
        missing_pct, imbalance_ratio, mean, std, min, max, median, etc.
    """
    datasets = read_directory(directory, extensions=extensions, recursive=recursive)
    if not datasets:
        return pd.DataFrame()

    rows = []
    for path, df in datasets.items():
        name = Path(path).name
        rows.append(summarize_dataset(df, name=name))

    summary_df = pd.DataFrame(rows)
    return summary_df


def plot_missing(source, output=None, **kwargs):
    """Generate a bar chart of missing values per column.

    Args:
        source: A DataFrame or a file path to load.
        output: Optional file path to save the chart image.
    """
    import matplotlib.pyplot as plt

    report = search_missing(source, **kwargs)
    report = report[report["missing_count"] > 0]

    if report.empty:
        print("No missing values found.")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(report) * 0.4)))
    bars = ax.barh(report["column"], report["missing_count"])

    for bar, pct in zip(bars, report["missing_percent"]):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{pct}%", va="center",
        )

    ax.set_xlabel("Missing Count")
    ax.set_title("Missing Values per Column")
    plt.tight_layout()

    if output:
        fig.savefig(output)
        print(f"Chart saved to {output}")
    else:
        plt.show()
