"""Core dataset reader — format detection, single-file and directory reading."""

import os
from pathlib import Path

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
