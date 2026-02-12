"""Command-line interface for DatasetAccess."""

import argparse
import sys

from .reader import (
    SUPPORTED_EXTENSIONS,
    read_dataset,
    read_directory,
    search_missing,
    plot_missing,
)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="dataset-access",
        description="Read and inspect any dataset from the command line.",
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="File path, directory, URL, or SQLAlchemy connection string.",
    )
    parser.add_argument(
        "-n", "--rows", type=int, default=None,
        help="Number of rows to display (default: all).",
    )
    parser.add_argument(
        "-f", "--format", default=None,
        help="Explicit format override (e.g. csv, sql).",
    )
    parser.add_argument(
        "-q", "--query", default=None,
        help="SQL query (required when format is sql).",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Convert and save dataset to this file path.",
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Show shape, dtypes, and memory usage.",
    )
    parser.add_argument(
        "--describe", action="store_true",
        help="Show summary statistics.",
    )
    parser.add_argument(
        "--missing", action="store_true",
        help="Show missing value report.",
    )
    parser.add_argument(
        "--missing-chart", nargs="?", const=True, default=None,
        help="Show missing value bar chart (optionally save to file).",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Scan subdirectories when reading a directory.",
    )
    parser.add_argument(
        "--formats", action="store_true",
        help="List all supported file formats and exit.",
    )
    return parser


def _print_df(df, rows):
    if rows is not None:
        print(df.head(rows).to_string())
    else:
        print(df.to_string())


def _save_output(df, output_path):
    from pathlib import Path

    ext = Path(output_path).suffix.lower()
    writers = {
        ".csv": lambda: df.to_csv(output_path, index=False),
        ".tsv": lambda: df.to_csv(output_path, sep="\t", index=False),
        ".json": lambda: df.to_json(output_path, orient="records", indent=2),
        ".jsonl": lambda: df.to_json(output_path, orient="records", lines=True),
        ".xlsx": lambda: df.to_excel(output_path, index=False),
        ".parquet": lambda: df.to_parquet(output_path, index=False),
        ".feather": lambda: df.to_feather(output_path),
    }
    writer = writers.get(ext)
    if writer is None:
        print(f"Error: cannot write to format '{ext}'.", file=sys.stderr)
        sys.exit(1)
    writer()
    print(f"Saved to {output_path}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.formats:
        print("Supported extensions:")
        for ext in sorted(SUPPORTED_EXTENSIONS):
            print(f"  {ext}")
        return

    if not args.source:
        parser.print_help()
        sys.exit(1)

    # --- Directory mode ---
    import os

    if os.path.isdir(args.source):
        datasets = read_directory(
            args.source, recursive=args.recursive,
        )
        if not datasets:
            sys.exit(1)

        for path, df in datasets.items():
            print(f"\n{'='*60}")
            print(f"File: {path}  ({df.shape[0]} rows x {df.shape[1]} cols)")
            print(f"{'='*60}")

            if args.info:
                df.info()
            elif args.describe:
                print(df.describe().to_string())
            elif args.missing:
                print(search_missing(df).to_string(index=False))
            elif args.output:
                # skip batch output in directory mode
                _print_df(df, args.rows)
            else:
                _print_df(df, args.rows)
        return

    # --- Single file mode ---
    extra = {}
    if args.query:
        extra["query"] = args.query

    try:
        df = read_dataset(args.source, format=args.format, **extra)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.missing_chart is not None:
        output = args.missing_chart if isinstance(args.missing_chart, str) else None
        plot_missing(df, output=output)
        return

    if args.missing:
        print(search_missing(df).to_string(index=False))
        return

    if args.info:
        print(f"Shape: {df.shape}")
        print()
        df.info()
        return

    if args.describe:
        print(df.describe().to_string())
        return

    if args.output:
        _save_output(df, args.output)
        return

    _print_df(df, args.rows)


if __name__ == "__main__":
    main()
