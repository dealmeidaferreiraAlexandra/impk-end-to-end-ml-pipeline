from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH
from src.models import train_and_save_models

def main():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_DATA_PATH}. Put Titanic train.csv there first."
        )

    df = pd.read_csv(RAW_DATA_PATH)
    _, metrics_df, best_model = train_and_save_models(df)

    print("\nTraining finished.\n")
    print(metrics_df.to_string(index=False))
    print(f"\nBest model: {best_model}")

if __name__ == "__main__":
    main()