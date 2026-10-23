"""Print the non-destructive dataset audit."""

import json

from hematology.constants import DEFAULT_DATASET_PATH
from hematology.data import audit_dataset, load_dataset

data = load_dataset(DEFAULT_DATASET_PATH)
print(json.dumps(audit_dataset(data, DEFAULT_DATASET_PATH).to_dict(), indent=2))
