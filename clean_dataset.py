import pandas as pd

df = pd.read_csv("diagnosed_cbc_data_v4.csv")
df.columns = df.columns.astype(str).str.strip()

print("Original records:", len(df))

# Remove exact duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

print("Records after removing duplicates:", len(df))

# Save a separate cleaned copy
df.to_csv("diagnosed_cbc_cleaned.csv", index=False)

print("Saved as diagnosed_cbc_cleaned.csv")