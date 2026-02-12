"""Command-line interface for dataset-access."""

import argparse
import sys

import pandas as pd

from dataset_access.reader import read_dataset, list_supported_formats


def main():
    parser = argparse.ArgumentParser(
        description="Read any dataset and display or convert it."
    )
    parser.add_argument("source", help="File path, URL, or SQL connection string.")
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
        "--formats",
        action="store_true",
        help="List all supported formats and exit.",
    )

    args = parser.parse_args()

    if args.formats:
        for ext, name in sorted(list_supported_formats().items()):
            print(f"  {ext:12s}  {name}")
        return

    kwargs = {}
    if args.query:
        kwargs["query"] = args.query

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

    if args.info:
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
