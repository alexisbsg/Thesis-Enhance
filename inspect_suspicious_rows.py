import pandas as pd

df = pd.read_csv("diagnosed_cbc_cleaned.csv")

checks = {
    "NEUTp": [5317, 671],
    "RBC": [41.1, 90.8, 29.7],
    "HGB": [85.1, -10, 35.2, 87.1, 41.0],
    "HCT": [316, 741, 3715],
    "MCV": [-79.3, 990],
    "MCH": [241, 275, 3117, 2511],
    "PCT": [11.3, 12.8, 13.6, 12.0]
}

for col, values in checks.items():
    for value in values:
        rows = df[df[col] == value]
        if not rows.empty:
            print(f"\n{col} = {value}")
            print(rows.to_string(index=False))