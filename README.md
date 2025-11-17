# PyOD AutoEncoder Demo (uv project)

This is a minimal **uv + Python** project that demonstrates how to use
**PyOD's AutoEncoder** for anomaly detection on a mock CSV dataset.

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) installed

## Setup

```bash
cd pyod_ae_uv_project
uv sync
```

This will create a `.venv` and install dependencies listed in `pyproject.toml`.

## Run demo

```bash
uv run pyod-ae-demo
```

The script will:

1. Load `data/mock_data.csv`
2. Train a PyOD AutoEncoder on the data
3. Evaluate on a train/test split
4. Print a classification report (precision/recall/F1 for normal vs anomaly)

## Data format

`data/mock_data.csv` has the following columns:

- `f1` ... `f20` : numeric features
- `label` : 0 for normal, 1 for anomaly
```
