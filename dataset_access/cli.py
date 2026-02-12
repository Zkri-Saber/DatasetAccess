"""Command-line interface for dataset-access."""

import argparse
import sys

import pandas as pd

from dataset_access.reader import (
    read_dataset,
    read_directory,
    list_supported_formats,
    search_missing,
    plot_missing,
)


def main():
    parser = argparse.ArgumentParser(
        description="Read any dataset and display or convert it."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help="File path, directory path, URL, or SQL connection string. "
             "When a directory is given, all supported dataset files inside it are read.",
    )
    parser.add_argument(
        "-f", "--format",
        help="Explicit format (e.g. csv, json, parquet). Auto-detected if omitted.",
    )
    parser.add_argument(
        "-q", "--query",
        help="SQL query (required when format=sql).",
    )
    parser.add_argument(
        "-n", "--head",
        type=int,
        default=None,
        help="Show only the first N rows.",
    )
    parser.add_argument(
        "-o", "--output",
        help="Write the result to a file (format inferred from extension).",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print DataFrame info (shape, dtypes, memory) instead of data.",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print summary statistics instead of data.",
    )
    parser.add_argument(
        "--missing",
        action="store_true",
        help="Show missing value report (count, percentage per column).",
    )
    parser.add_argument(
        "--missing-chart",
        nargs="?",
        const="",
        default=None,
        metavar="FILE",
        help="Show a bar chart of missing values. Optionally save to FILE (e.g. missing.png).",
    )
    parser.add_argument(
        "--formats",
        action="store_true",
        help="List all supported formats and exit.",
    )

    args = parser.parse_args()

    if args.formats:
        for ext, name in sorted(list_supported_formats().items()):
            print(f"  {ext:12s}  {name}")
        return

    if args.source is None:
        parser.error("the following arguments are required: source")

    kwargs = {}
    if args.query:
        kwargs["query"] = args.query

    # --- load dataset(s) ----------------------------------------------------
    import os

    source = args.source.rstrip(os.sep) if args.source != os.sep else args.source
    is_dir = os.path.isdir(source)

    if is_dir:
        try:
            datasets = read_directory(args.source, **kwargs)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(datasets)} dataset(s) in '{args.source}':\n")
        for name, df in datasets.items():
            print(f"{'=' * 60}")
            print(f"  File: {name}  |  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
            print(f"{'=' * 60}")
            _display_single(df, name, args)
            print()
    else:
        try:
            df = read_dataset(args.source, format=args.format, **kwargs)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        if args.head is not None:
            df = df.head(args.head)

        if args.output:
            _write_output(df, args.output)
            print(f"Written {len(df)} rows to {args.output}")
            return

        _display_single(df, args.source, args)


def _display_single(df: pd.DataFrame, name: str, args) -> None:
    """Display a single DataFrame according to the CLI flags."""
    if args.head is not None:
        df = df.head(args.head)

    if args.missing_chart is not None:
        output_path = args.missing_chart if args.missing_chart else None
        plot_missing(df, output=output_path)
    elif args.missing:
        report = search_missing(df)
        total_missing = report["missing_count"].sum()
        total_cells = report["total_count"].iloc[0] * len(report)
        print(f"Dataset: {name}")
        print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"Total missing: {total_missing} / {total_cells} cells "
              f"({total_missing / total_cells * 100:.2f}%)\n")
        print(report.to_string(index=False))
    elif args.info:
        df.info()
    elif args.describe:
        print(df.describe().to_string())
    else:
        print(df.to_string())


def _write_output(df: pd.DataFrame, path: str):
    """Write a DataFrame to a file, format inferred from extension."""
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    writers = {
        "csv": lambda: df.to_csv(path, index=False),
        "tsv": lambda: df.to_csv(path, index=False, sep="\t"),
        "json": lambda: df.to_json(path, orient="records", indent=2),
        "jsonl": lambda: df.to_json(path, orient="records", lines=True),
        "xlsx": lambda: df.to_excel(path, index=False),
        "parquet": lambda: df.to_parquet(path, index=False),
        "feather": lambda: df.to_feather(path),
        "pkl": lambda: df.to_pickle(path),
    }
    writer = writers.get(ext)
    if writer is None:
        raise ValueError(f"Cannot write to '.{ext}' format. Supported: {', '.join(sorted(writers))}")
    writer()


if __name__ == "__main__":
    main()
