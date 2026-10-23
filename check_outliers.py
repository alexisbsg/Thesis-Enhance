import pandas as pd

df = pd.read_csv("diagnosed_cbc_cleaned.csv")

limits = {
    "WBC": (0, 200),
    "LYMp": (0, 100),
    "NEUTp": (0, 100),
    "LYMn": (0, 200),
    "NEUTn": (0, 200),
    "RBC": (0, 20),
    "HGB": (0, 30),
    "HCT": (0, 100),
    "MCV": (0, 200),
    "MCH": (0, 100),
    "MCHC": (0, 100),
    "PLT": (0, 2000),
    "PDW": (0, 100),
    "PCT": (0, 10)
}

for col, (low, high) in limits.items():
    bad = df[(df[col] < low) | (df[col] > high)]
    if not bad.empty:
        print(f"\n{col}: {len(bad)} suspicious records")
        print(bad[["Diagnosis", col]].head(10).to_string(index=False))

print("\nCheck complete.")