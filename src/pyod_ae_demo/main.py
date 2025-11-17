import os
from pathlib import Path

import numpy as np
import pandas as pd
from pyod.models.auto_encoder import AutoEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def load_data(csv_path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    """Load mock CSV data.

    Expected columns:
      - f1 ... f20 : numeric features
      - label     : 0 (normal) or 1 (anomaly)
    """
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    return X, y


def train_and_evaluate(X: np.ndarray, y: np.ndarray) -> None:
    # Train/test split with stratification so anomalies are preserved in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    contamination = float(np.mean(y)) if np.any(y == 1) else 0.01
    print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"[INFO] Estimated contamination (anomaly ratio): {contamination:.4f}")

    clf = AutoEncoder(
        hidden_neuron_list=[64, 32, 16, 32, 64],
        hidden_activation_name="relu",
        epoch_num=40,
        batch_size=128,
        dropout_rate=0.1,
        contamination=contamination,
        verbose=1,
    )

    print("[INFO] Fitting AutoEncoder...")
    clf.fit(X_train)

    print("[INFO] Evaluating on test set...")
    y_pred = clf.predict(X_test)          # 0 = normal, 1 = anomaly
    scores = clf.decision_function(X_test)

    print("\n[INFO] Classification report (0=normal, 1=anomaly):")
    print(classification_report(y_test, y_pred, digits=4))

    # Show a few example scores
    for i in range(5):
        print(f"Sample {i:02d} - label={y_test[i]}, pred={y_pred[i]}, score={scores[i]:.4f}")


def main() -> None:
    # Resolve path to data/mock_data.csv relative to this file
    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "data" / "mock_data.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV at {csv_path}")

    print(f"[INFO] Loading data from {csv_path} ...")
    X, y = load_data(csv_path)
    train_and_evaluate(X, y)


if __name__ == "__main__":
    main()
