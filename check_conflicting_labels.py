import pandas as pd

df = pd.read_csv("diagnosed_cbc_data_v4.csv")
df.columns = df.columns.astype(str).str.strip()

features = [c for c in df.columns if c != "Diagnosis"]

# Find identical CBC measurements assigned different diagnoses
label_counts = df.groupby(features)["Diagnosis"].nunique()
conflicts = label_counts[label_counts > 1]

print("Total records:", len(df))
print("Identical CBC feature combinations:", df.duplicated(subset=features).sum())
print("Feature combinations with conflicting diagnoses:", len(conflicts))

if len(conflicts) > 0:
    conflict_data = df.set_index(features).loc[conflicts.index]
    print("\nExamples of conflicting diagnoses:")
    print(conflict_data.head(20))
else:
    print("\nNo conflicting diagnoses found.")