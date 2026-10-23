import pandas as pd

df = pd.read_csv("diagnosed_cbc_cleaned.csv")

original_count = len(df)

# Broad screening limits for clearly invalid values.
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

valid = pd.Series(True, index=df.index)

for col, (low, high) in limits.items():
    valid &= df[col].between(low, high)

clean = df.loc[valid].copy()

print("Starting records:", original_count)
print("Flagged records:", original_count - len(clean))
print("Remaining records:", len(clean))
print("\nDiagnosis counts after screening:")
print(clean["Diagnosis"].value_counts())

clean.to_csv("diagnosed_cbc_screened.csv", index=False)
print("\nSaved as diagnosed_cbc_screened.csv")