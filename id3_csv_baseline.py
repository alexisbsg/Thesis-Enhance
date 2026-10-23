
import pandas as pd
from pathlib import Path


def load_dataset(file_path):
    path = Path(file_path)

    if not path.exists():
        print(f"Dataset not found: {path}")
        return None

    try:
        data = pd.read_csv(path)
    except Exception as error:
        print(f"Could not read dataset: {error}")
        return None

    if data.empty:
        print("The dataset is empty.")
        return None

    # Clean column names and remove fully empty rows.
    data.columns = data.columns.astype(str).str.strip()
    data = data.dropna(how="all").reset_index(drop=True)

    print("\nDATASET LOADED SUCCESSFULLY")
    print("-" * 40)
    print("Rows:", len(data))
    print("Columns:", len(data.columns))

    print("\nColumn names and data types:")
    print(data.dtypes.to_string())

    print("\nFirst five records:")
    print(data.head().to_string(index=False))

    print("\nMissing values per column:")
    print(data.isnull().sum().to_string())

    print("\nDuplicate rows:", data.duplicated().sum())

    print("\nUnique values per column:")
    for column in data.columns:
        print(f"{column}: {data[column].nunique(dropna=True)}")

    return data


if __name__ == "__main__":
    file_name = input("Enter the CSV file path: ").strip().strip('"')
    data = load_dataset(file_name)

    if data is not None:
        print("\nDataset inspection completed successfully.")
        print("No classification or clinical conclusions have been made.")
