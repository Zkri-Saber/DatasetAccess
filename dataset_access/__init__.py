"""DatasetAccess — read any dataset format with a single function call."""

from .reader import read_dataset, read_directory, search_missing, plot_missing

__all__ = ["read_dataset", "read_directory", "search_missing", "plot_missing"]
