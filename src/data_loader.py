import pandas as pd
from pathlib import Path


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the customer dataset from a CSV file.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"No file found at {path.resolve()}")

    df = pd.read_csv(path)
    return df