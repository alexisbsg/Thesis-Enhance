import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("diagnosed_cbc_data_v4.csv")
df.columns = df.columns.astype(str).str.strip()
df = df.dropna().reset_index(drop=True)

train, test = train_test_split(
    df, test_size=0.25, random_state=42, stratify=df["Diagnosis"]
)

features = [c for c in df.columns if c != "Diagnosis"]

train_records = set(
    map(tuple, train[features].to_numpy())
)
test_records = set(
    map(tuple, test[features].to_numpy())
)

print("Exact duplicate rows:", df.duplicated().sum())
print("Train/test shared feature records:", len(train_records & test_records))

print("\nSuspicious HGB/HCT values:")
print(df.loc[(df["HGB"] < 2) | (df["HCT"] > 100),
             ["HGB", "HCT", "Diagnosis"]].head(20))

print("\nFeature ranges:")
print(df[features].describe().T[["min", "max", "mean"]])