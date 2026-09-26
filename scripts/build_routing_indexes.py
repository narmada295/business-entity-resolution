from pathlib import Path
from collections import defaultdict
import pickle
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.normalize import normalize_name, normalize_address


TRAIN_DIR = ROOT / "data" / "train"
INDEX_DIR = ROOT / "outputs" / "indexes"

INDEX_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


READ_CHUNK_SIZE = 100_000


NAME_STOP_TOKENS = {
    "private",
    "limited",
    "ltd",
    "llp",
    "pvt",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
}


ADDRESS_STOP_TOKENS = {
    "road",
    "rd",
    "street",
    "st",
    "lane",
    "ln",
    "floor",
    "fl",
    "building",
    "bldg",
    "house",
    "no",
    "number",
    "plot",
    "shop",
    "door",
    "near",
    "opp",
    "opposite",
    "district",
    "city",
    "state",
}


def safe_country_name(country):
    return (
        country
        .replace("/", "_")
        .replace(" ", "_")
    )


def token_prefix(token):
    token = "".join(
        ch
        for ch in token
        if ch.isalnum()
    )

    if not token:
        return None

    if len(token) == 1:
        return f"p:{token}"

    return f"p:{token[:2]}"


def name_route_keys(text):
    text = normalize_name(text)

    keys = set()

    for token in text.split():

        if token in NAME_STOP_TOKENS:
            continue

        key = token_prefix(token)

        if key:
            keys.add(key)

    return keys


def address_route_keys(text):
    text = normalize_address(text)

    keys = set()

    for token in text.split():

        cleaned = "".join(
            ch
            for ch in token
            if ch.isalnum()
        )

        if not cleaned:
            continue

        if cleaned.isdigit():

            # Exact address numbers are highly selective.
            try:
                cleaned = str(int(cleaned))
            except ValueError:
                pass

            keys.add(
                f"n:{cleaned}"
            )

            continue


        if token in ADDRESS_STOP_TOKENS:
            continue

        key = token_prefix(token)

        if key:
            keys.add(key)

    return keys


# ============================================================
# ARGUMENT
# ============================================================

if len(sys.argv) != 2:

    print(
        "Usage:\n"
        "  python scripts/build_routing_indexes.py S2\n"
        "  python scripts/build_routing_indexes.py S3"
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


# ============================================================
# DISCOVER COUNTRIES
# ============================================================

countries = set()


for chunk in pd.read_csv(
    TARGET_PATH,
    sep="\t",
    usecols=["country"],
    dtype=str,
    keep_default_na=False,
    chunksize=READ_CHUNK_SIZE,
):

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
# BUILD EACH COUNTRY
# ============================================================

for country in countries:

    print()
    print("=" * 70)
    print(
        f"ROUTING INDEX: {SOURCE} / {country}"
    )
    print("=" * 70)


    name_postings = defaultdict(
        list
    )

    address_postings = defaultdict(
        list
    )


    row_idx = 0

    start = time.perf_counter()


    reader = pd.read_csv(
        TARGET_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "business_name",
            "business_address",
            "country",
        ],
        chunksize=READ_CHUNK_SIZE,
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


        names = chunk[
            "business_name"
        ].tolist()

        addresses = chunk[
            "business_address"
        ].tolist()


        for name, address in zip(
            names,
            addresses,
        ):

            for key in name_route_keys(
                name
            ):

                name_postings[
                    key
                ].append(
                    row_idx
                )


            for key in address_route_keys(
                address
            ):

                address_postings[
                    key
                ].append(
                    row_idx
                )


            row_idx += 1


        print(
            f"\r"
            f"chunk={chunk_no:<4} "
            f"rows={row_idx:,}",
            end="",
            flush=True,
        )


    print()


    # Convert lists to compact arrays.
    import numpy as np


    name_postings = {
        key: np.asarray(
            values,
            dtype=np.int32,
        )
        for key, values
        in name_postings.items()
    }


    address_postings = {
        key: np.asarray(
            values,
            dtype=np.int32,
        )
        for key, values
        in address_postings.items()
    }


    name_df = {
        key: len(values)
        for key, values
        in name_postings.items()
    }


    address_df = {
        key: len(values)
        for key, values
        in address_postings.items()
    }


    safe_country = safe_country_name(
        country
    )


    path = (
        INDEX_DIR
        / (
            f"{SOURCE.lower()}_"
            f"{safe_country}_routing.pkl"
        )
    )


    with open(
        path,
        "wb",
    ) as f:

        pickle.dump(
            {
                "source": SOURCE,
                "country": country,
                "rows": row_idx,

                "name_postings":
                    name_postings,

                "address_postings":
                    address_postings,

                "name_df":
                    name_df,

                "address_df":
                    address_df,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


    elapsed = (
        time.perf_counter()
        - start
    )


    print(
        "Rows:",
        f"{row_idx:,}",
    )

    print(
        "Name routing keys:",
        f"{len(name_postings):,}",
    )

    print(
        "Address routing keys:",
        f"{len(address_postings):,}",
    )

    print(
        "Saved:",
        path,
    )

    print(
        "Runtime:",
        f"{elapsed / 60:.2f} min",
    )