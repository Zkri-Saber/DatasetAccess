"""DatasetAccess — read any dataset format with a single function call."""

from .reader import (
    read_dataset,
    read_directory,
    search_missing,
    plot_missing,
    summarize_dataset,
    summarize_datasets,
)

__all__ = [
    "read_dataset",
    "read_directory",
    "search_missing",
    "plot_missing",
    "summarize_dataset",
    "summarize_datasets",
]
