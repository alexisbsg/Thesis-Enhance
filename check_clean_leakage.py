import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("diagnosed_cbc_screened.csv")
features = [c for c in df.columns if c != "Diagnosis"]

train, test = train_test_split(
    df, test_size=0.25, random_state=42,
    stratify=df["Diagnosis"]
)

train_records = set(map(tuple, train[features].to_numpy()))
test_records = set(map(tuple, test[features].to_numpy()))

print("Screened dataset:", len(df))
print("Shared feature combinations:", len(train_records & test_records))