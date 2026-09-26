from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.features import (
    prepare_record,
    compute_pair_features,
    MODEL_FEATURE_COLUMNS,
)


a = prepare_record(
    "Star Technologies Private Limited",
    "27 Amratolsa Street, Kolkata",
)

b = prepare_record(
    "STAR TECHNOLOGIES PRIVATE LIMITED",
    "47 Amratolsa Street, Kolkata",
)

features = compute_pair_features(
    a,
    b,
)

print(
    "Feature count:",
    len(features),
)

print(
    "Expected:",
    len(MODEL_FEATURE_COLUMNS),
)

for key in [
    "name_ratio",
    "legal_suffix_stripped_ratio",
    "translit_name_ratio",
    "address_token_jaccard",
    "number_conflict",
]:
    print(
        key,
        features[key],
    )

assert (
    set(features)
    ==
    set(MODEL_FEATURE_COLUMNS)
)

print(
    "\nFeature module OK."
)