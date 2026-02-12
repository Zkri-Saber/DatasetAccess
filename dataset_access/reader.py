"""Unified dataset reader supporting multiple file formats and data sources."""

import os
from pathlib import Path

import pandas as pd


SUPPORTED_FORMATS = {
    ".csv": "CSV",
    ".tsv": "TSV",
    ".json": "JSON",
    ".jsonl": "JSON Lines",
    ".xlsx": "Excel (.xlsx)",
    ".xls": "Excel (.xls)",
    ".parquet": "Parquet",
    ".feather": "Feather",
    ".orc": "ORC",
    ".hdf": "HDF5",
    ".h5": "HDF5",
    ".html": "HTML table",
    ".xml": "XML",
    ".fwf": "Fixed-width",
    ".pkl": "Pickle",
    ".sas7bdat": "SAS",
    ".xpt": "SAS XPORT",
    ".dta": "Stata",
    ".sav": "SPSS",
}


def read_dataset(source: str, format: str | None = None, **kwargs) -> pd.DataFrame:
    """Read a dataset from a file path, URL, or SQL connection string.

    Args:
        source: File path, URL, or SQLAlchemy connection string.
        format: Explicit format override (e.g. "csv", "json", "parquet").
                 Auto-detected from file extension if not provided.
        **kwargs: Extra keyword arguments forwarded to the underlying pandas reader.

    Returns:
        A pandas DataFrame with the loaded data.

    Supported formats:
        csv, tsv, json, jsonl, xlsx, xls, parquet, feather, orc,
        hdf5, html, xml, fixed-width, pickle, sas, stata, spss.

    Also supports:
        - SQL queries via SQLAlchemy connection strings (set format="sql"
          and pass `query` in kwargs).
        - URLs for any format that pandas can stream.

    Examples:
        >>> df = read_dataset("data.csv")
        >>> df = read_dataset("data.json", orient="records")
        >>> df = read_dataset("s3://bucket/data.parquet")
        >>> df = read_dataset("sqlite:///my.db", format="sql", query="SELECT * FROM t")
    """
    if format:
        ext = f".{format.lower().lstrip('.')}"
    else:
        ext = _detect_format(source)

    reader = _get_reader(ext)
    if reader is None:
        raise ValueError(
            f"Unsupported format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    return reader(source, **kwargs)


def list_supported_formats() -> dict[str, str]:
    """Return a mapping of file extension to human-readable format name."""
    return dict(SUPPORTED_FORMATS)


def search_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze missing values in a DataFrame.

    Returns a summary DataFrame with one row per column, containing:
        - column: Column name
        - missing_count: Number of missing (NaN/None) values
        - total_count: Total number of rows
        - missing_percent: Percentage of missing values
        - dtype: Data type of the column

    Args:
        df: The DataFrame to analyze.

    Returns:
        A DataFrame summarizing missing values, sorted by missing_count descending.

    Example:
        >>> df = read_dataset("data.csv")
        >>> report = search_missing(df)
        >>> print(report)
    """
    missing_count = df.isnull().sum()
    total_count = len(df)
    missing_percent = (missing_count / total_count * 100).round(2)

    report = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_count.values,
        "total_count": total_count,
        "missing_percent": missing_percent.values,
        "dtype": df.dtypes.values,
    })

    report = report.sort_values("missing_count", ascending=False).reset_index(drop=True)
    return report


def plot_missing(
    source: "pd.DataFrame | str",
    format: str | None = None,
    output: str | None = None,
    **kwargs,
) -> None:
    """Display a bar chart of missing values per column.

    Args:
        source: A pandas DataFrame, or a file path / URL to a dataset.
                When a string is given the dataset is loaded via read_dataset().
        format: Explicit format override when *source* is a path (e.g. "csv").
        output: Optional file path to save the chart (e.g. "missing.png").
                If *None* the chart is displayed interactively.
        **kwargs: Extra keyword arguments forwarded to read_dataset() when
                  *source* is a path.
    """
    import matplotlib.pyplot as plt

    if isinstance(source, pd.DataFrame):
        df = source
    else:
        df = read_dataset(source, format=format, **kwargs)

    report = search_missing(df)
    report = report[report["missing_count"] > 0]

    if report.empty:
        print("No missing values found.")
        return

    columns = report["column"]
    counts = report["missing_count"]
    percentages = report["missing_percent"]

    fig, ax = plt.subplots(figsize=(max(6, len(columns) * 0.8), 5))

    bars = ax.bar(columns, counts, color="#e74c3c", edgecolor="white")

    for bar, pct in zip(bars, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{pct}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Column")
    ax.set_ylabel("Missing Count")
    ax.set_title("Missing Values per Column")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
        print(f"Chart saved to {output}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_format(source: str) -> str:
    """Infer the file format from the source path/URL."""
    # Strip query params for URLs
    path = source.split("?")[0]
    ext = Path(path).suffix.lower()
    if not ext:
        raise ValueError(
            f"Cannot detect format from '{source}'. Pass format= explicitly."
        )
    return ext


def _get_reader(ext: str):
    """Return the appropriate reader function for a given extension."""
    readers = {
        ".csv": _read_csv,
        ".tsv": _read_tsv,
        ".json": _read_json,
        ".jsonl": _read_jsonl,
        ".xlsx": _read_excel,
        ".xls": _read_excel,
        ".parquet": _read_parquet,
        ".feather": _read_feather,
        ".orc": _read_orc,
        ".hdf": _read_hdf,
        ".h5": _read_hdf,
        ".html": _read_html,
        ".xml": _read_xml,
        ".fwf": _read_fwf,
        ".pkl": _read_pickle,
        ".sas7bdat": _read_sas,
        ".xpt": _read_sas,
        ".dta": _read_stata,
        ".sav": _read_spss,
        ".sql": _read_sql,
    }
    return readers.get(ext)


# ---------------------------------------------------------------------------
# Format-specific readers
# ---------------------------------------------------------------------------

def _read_csv(source, **kwargs):
    return pd.read_csv(source, **kwargs)


def _read_tsv(source, **kwargs):
    kwargs.setdefault("sep", "\t")
    return pd.read_csv(source, **kwargs)


def _read_json(source, **kwargs):
    return pd.read_json(source, **kwargs)


def _read_jsonl(source, **kwargs):
    kwargs.setdefault("lines", True)
    return pd.read_json(source, **kwargs)


def _read_excel(source, **kwargs):
    return pd.read_excel(source, **kwargs)


def _read_parquet(source, **kwargs):
    return pd.read_parquet(source, **kwargs)


def _read_feather(source, **kwargs):
    return pd.read_feather(source, **kwargs)


def _read_orc(source, **kwargs):
    return pd.read_orc(source, **kwargs)


def _read_hdf(source, **kwargs):
    return pd.read_hdf(source, **kwargs)


def _read_html(source, **kwargs):
    tables = pd.read_html(source, **kwargs)
    if not tables:
        raise ValueError(f"No HTML tables found in '{source}'.")
    return tables[0]


def _read_xml(source, **kwargs):
    return pd.read_xml(source, **kwargs)


def _read_fwf(source, **kwargs):
    return pd.read_fwf(source, **kwargs)


def _read_pickle(source, **kwargs):
    return pd.read_pickle(source, **kwargs)


def _read_sas(source, **kwargs):
    return pd.read_sas(source, **kwargs)


def _read_stata(source, **kwargs):
    return pd.read_stata(source, **kwargs)


def _read_spss(source, **kwargs):
    return pd.read_spss(source, **kwargs)


def _read_sql(source, query=None, **kwargs):
    if not query:
        raise ValueError("format='sql' requires a `query` keyword argument.")
    from sqlalchemy import create_engine
    engine = create_engine(source)
    return pd.read_sql(query, engine, **kwargs)
