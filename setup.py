from setuptools import setup, find_packages

setup(
    name="dataset-access",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0",
        "openpyxl>=3.1",
        "pyarrow>=14.0",
        "sqlalchemy>=2.0",
    ],
    entry_points={
        "console_scripts": [
            "dataset-access=dataset_access.cli:main",
        ],
    },
    python_requires=">=3.9",
)
