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
from dataset_access import read_dataset

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

# SQL query
dataset-access "sqlite:///my.db" -f sql -q "SELECT * FROM users"

# List all supported formats
dataset-access --formats
```

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
