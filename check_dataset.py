import pandas as pd

df = pd.read_csv("diagnosed_cbc_data_v4.csv")

print("Dataset shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())

print("\nDiagnosis distribution:")
print(df["Diagnosis"].value_counts())

print("\nUnique values per column:")
print(df.nunique())

print("\nFirst 5 rows:")
print(df.head())