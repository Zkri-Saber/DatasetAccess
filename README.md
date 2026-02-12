# DatasetAccess

A Python library and CLI tool to read **any** dataset format into a pandas DataFrame with a single function call.

## Supported Formats

| Extension | Format |
|-----------|--------|
| `.csv` | CSV |
| `.tsv` | TSV |
| `.json` | JSON |
| `.jsonl` | JSON Lines |
| `.xlsx` / `.xls` | Excel |
| `.parquet` | Apache Parquet |
| `.feather` | Feather |
| `.orc` | ORC |
| `.hdf` / `.h5` | HDF5 |
| `.html` | HTML tables |
| `.xml` | XML |
| `.fwf` | Fixed-width |
| `.pkl` | Pickle |
| `.sas7bdat` / `.xpt` | SAS |
| `.dta` | Stata |
| `.sav` | SPSS |
| SQL | Via SQLAlchemy connection string |

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Python API

```python
from dataset_access import read_dataset, read_directory, search_missing, summarize_datasets

# Auto-detect format from extension
df = read_dataset("data.csv")
df = read_dataset("data.parquet")
df = read_dataset("data.xlsx", sheet_name="Sheet2")

# Explicit format
df = read_dataset("myfile.dat", format="csv", sep="|")

# Read from URL
df = read_dataset("https://example.com/data.json")

# Read from SQL database
df = read_dataset("sqlite:///my.db", format="sql", query="SELECT * FROM users")

# Read all files from a directory
datasets = read_directory("data/", recursive=True)
for path, df in datasets.items():
    print(f"{path}: {df.shape}")

# Summarize all datasets in a directory into one table
summary = summarize_datasets("data/")
print(summary)
#   dataset_name  num_instances  num_features  missing_pct  imbalance_ratio  mean  ...

# Search for missing values
report = search_missing(df)
print(report)
#   column  missing_count  total_count  missing_percent   dtype
#   age              15          100            15.0     float64
#   email             3          100             3.0      object
#   name              0          100             0.0      object
```

### CLI

```bash
# Display a dataset
dataset-access data.csv

# Show first 10 rows
dataset-access data.parquet -n 10

# Get shape, dtypes, and memory info
dataset-access data.xlsx --info

# Summary statistics
dataset-access data.csv --describe

# Convert between formats
dataset-access data.csv -o data.parquet

# Search for missing values in a dataset
dataset-access data.csv --missing

# SQL query
dataset-access "sqlite:///my.db" -f sql -q "SELECT * FROM users"

# Summarize all datasets in a directory
dataset-access data/ --summary

# Summarize recursively and save to CSV
dataset-access data/ --summary --recursive --summary-output summary.csv

# List all supported formats
dataset-access --formats
```

## Summary Table Columns

The `--summary` flag produces a table with these columns for each dataset file:

| Column | Description |
|--------|-------------|
| `dataset_name` | File name |
| `num_instances` | Number of rows |
| `num_features` | Number of columns |
| `numeric_features` | Count of numeric columns |
| `categorical_features` | Count of categorical columns |
| `missing_values` | Total missing cells |
| `missing_pct` | Percentage of missing cells |
| `duplicate_rows` | Number of duplicate rows |
| `duplicate_pct` | Percentage of duplicate rows |
| `target_column` | Auto-detected target/label column |
| `target_classes` | Number of unique classes in target |
| `imbalance_ratio` | Majority / minority class ratio |
| `mean` | Grand mean of all numeric features |
| `std` | Average standard deviation |
| `min` | Global minimum |
| `max` | Global maximum |
| `median` | Grand median |
| `memory_mb` | Memory usage in MB |

## Project Structure

```
DatasetAccess/
  dataset_access/
    __init__.py        # Public API (re-exports read_dataset)
    reader.py          # Core reader — format detection & dispatch
    cli.py             # Command-line interface
  setup.py             # Package setup
  requirements.txt     # Dependencies
```
