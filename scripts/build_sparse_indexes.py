from pathlib import Path
import sys
import json
import gc
import time

import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.normalize import (
    normalize_name,
    normalize_address,
)


# ============================================================
# PATHS
# ============================================================

TRAIN_DIR = ROOT / "data" / "train"
INDEX_DIR = ROOT / "outputs" / "indexes"

INDEX_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

READ_CHUNK_SIZE = 100_000

HASH_FEATURES = 2 ** 18


# ============================================================
# VECTORIZERS
# ============================================================

def make_name_vectorizer():

    return HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        n_features=HASH_FEATURES,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )


def make_address_vectorizer():

    return HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        n_features=HASH_FEATURES,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )


# ============================================================
# SOURCE ARGUMENT
# ============================================================

if len(sys.argv) != 2:

    print(
        "Usage:\n"
        "  python scripts/build_sparse_indexes.py S2\n"
        "  python scripts/build_sparse_indexes.py S3"
    )

    sys.exit(1)


SOURCE = sys.argv[1].upper()

if SOURCE == "S2":

    TARGET_PATH = (
        TRAIN_DIR
        / "train_source2.tsv"
    )

elif SOURCE == "S3":

    TARGET_PATH = (
        TRAIN_DIR
        / "train_source3.tsv"
    )

else:

    raise ValueError(
        "SOURCE must be S2 or S3"
    )


print("=" * 70)
print(f"BUILDING PERSISTENT INDEXES: {SOURCE}")
print("=" * 70)


# ============================================================
# DISCOVER COUNTRIES
# ============================================================

print("\nDiscovering countries...")


countries = set()


reader = pd.read_csv(
    TARGET_PATH,
    sep="\t",
    usecols=["country"],
    dtype=str,
    keep_default_na=False,
    chunksize=READ_CHUNK_SIZE,
)


for chunk in reader:

    countries.update(
        x
        for x in chunk["country"].unique()
        if x
    )


countries = sorted(countries)


print(
    "Countries:",
    countries,
)


# ============================================================
# BUILD EACH COUNTRY INDEPENDENTLY
# ============================================================

for country in countries:

    safe_country = (
        country
        .replace("/", "_")
        .replace(" ", "_")
    )


    prefix = (
        INDEX_DIR
        / f"{SOURCE.lower()}_{safe_country}"
    )


    print()
    print("=" * 70)
    print(
        f"{SOURCE} / {country}"
    )
    print("=" * 70)


    name_vectorizer = (
        make_name_vectorizer()
    )

    address_vectorizer = (
        make_address_vectorizer()
    )


    name_parts = []
    address_parts = []

    entity_ids = []

    row_count = 0

    started = time.perf_counter()


    # ========================================================
    # STREAM SOURCE ONCE FOR THIS COUNTRY
    # ========================================================

    reader = pd.read_csv(
        TARGET_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=READ_CHUNK_SIZE,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )


    for chunk_no, chunk in enumerate(
        reader,
        start=1,
    ):

        chunk = chunk[
            chunk["country"]
            == country
        ]


        if len(chunk) == 0:
            continue


        names = (
            chunk["business_name"]
            .map(normalize_name)
            .tolist()
        )


        addresses = (
            chunk["business_address"]
            .map(normalize_address)
            .tolist()
        )


        name_matrix = (
            name_vectorizer
            .transform(names)
            .tocsr()
        )


        address_matrix = (
            address_vectorizer
            .transform(addresses)
            .tocsr()
        )


        name_parts.append(
            name_matrix
        )

        address_parts.append(
            address_matrix
        )


        entity_ids.extend(
            chunk[
                "entity_id"
            ].tolist()
        )


        row_count += len(chunk)


        print(
            f"\r"
            f"chunk={chunk_no:<4} "
            f"country_rows={row_count:,}",
            end="",
            flush=True,
        )


    print()


    # ========================================================
    # COMBINE
    # ========================================================

    print(
        "Combining name matrix..."
    )


    name_full = sparse.vstack(
        name_parts,
        format="csr",
        dtype=np.float32,
    )


    del name_parts
    gc.collect()


    print(
        "Combining address matrix..."
    )


    address_full = sparse.vstack(
        address_parts,
        format="csr",
        dtype=np.float32,
    )


    del address_parts
    gc.collect()


    # ========================================================
    # SAVE
    # ========================================================

    name_path = Path(
        f"{prefix}_name.npz"
    )

    address_path = Path(
        f"{prefix}_address.npz"
    )

    ids_path = Path(
        f"{prefix}_ids.npy"
    )

    metadata_path = Path(
        f"{prefix}_meta.json"
    )


    print(
        "Saving name index..."
    )

    sparse.save_npz(
        name_path,
        name_full,
        compressed=True,
    )


    print(
        "Saving address index..."
    )

    sparse.save_npz(
        address_path,
        address_full,
        compressed=True,
    )


    print(
        "Saving entity IDs..."
    )

    np.save(
        ids_path,
        np.asarray(
            entity_ids,
            dtype=str,
        ),
    )


    elapsed = (
        time.perf_counter()
        - started
    )


    metadata = {

        "source": SOURCE,

        "country": country,

        "rows": row_count,

        "hash_features":
            HASH_FEATURES,

        "name_ngram_min":
            3,

        "name_ngram_max":
            4,

        "address_ngram_min":
            3,

        "address_ngram_max":
            5,

        "name_nnz":
            int(
                name_full.nnz
            ),

        "address_nnz":
            int(
                address_full.nnz
            ),

        "runtime_seconds":
            elapsed,
    }


    with open(
        metadata_path,
        "w",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )


    name_mb = (
        name_full.data.nbytes
        + name_full.indices.nbytes
        + name_full.indptr.nbytes
    ) / (1024 ** 2)


    address_mb = (
        address_full.data.nbytes
        + address_full.indices.nbytes
        + address_full.indptr.nbytes
    ) / (1024 ** 2)


    print()
    print(
        "Rows:",
        f"{row_count:,}",
    )

    print(
        "Name sparse size:",
        f"{name_mb:.2f} MB",
    )

    print(
        "Address sparse size:",
        f"{address_mb:.2f} MB",
    )

    print(
        "Combined:",
        f"{name_mb + address_mb:.2f} MB",
    )

    print(
        "Runtime:",
        f"{elapsed / 60:.2f} min",
    )


    del name_full
    del address_full
    del entity_ids

    gc.collect()


print()
print("=" * 70)
print(
    f"{SOURCE} indexing complete."
)
print("=" * 70)