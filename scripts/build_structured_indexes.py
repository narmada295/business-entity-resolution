from pathlib import Path
from collections import Counter, defaultdict
import pickle
import re
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.normalize import (
    normalize_name,
    normalize_address,
)


TRAIN_DIR = ROOT / "data" / "train"
INDEX_DIR = ROOT / "outputs" / "indexes"

INDEX_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


READ_CHUNK_SIZE = 100_000

MAX_TOKEN_DF = 500
MAX_NUMBER_DF = 1000

NUMBER_RE = re.compile(r"\d+")


STRUCTURED_STOP_TOKENS = {
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


def canonical_number(value):
    try:
        return str(int(value))
    except ValueError:
        return value


def extract_numbers(text):
    if not text:
        return set()

    return {
        canonical_number(x)
        for x in NUMBER_RE.findall(text)
    }


def address_tokens(text):
    if not text:
        return set()

    output = set()

    for token in text.split():

        if token in STRUCTURED_STOP_TOKENS:
            continue

        if token.isdigit():
            continue

        if len(token) <= 1:
            continue

        output.add(token)

    return output


# ============================================================
# SOURCE
# ============================================================

if len(sys.argv) != 2:

    print(
        "Usage:\n"
        "  python scripts/build_structured_indexes.py S2\n"
        "  python scripts/build_structured_indexes.py S3"
    )

    sys.exit(1)


SOURCE = sys.argv[1].upper()


if SOURCE == "S2":
    TARGET_PATH = TRAIN_DIR / "train_source2.tsv"

elif SOURCE == "S3":
    TARGET_PATH = TRAIN_DIR / "train_source3.tsv"

else:
    raise ValueError(
        "SOURCE must be S2 or S3"
    )


# ============================================================
# COUNTRIES
# ============================================================

countries = set()


for chunk in pd.read_csv(
    TARGET_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    usecols=["country"],
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
# BUILD COUNTRY
# ============================================================

for country in countries:

    print()
    print("=" * 70)
    print(
        f"STRUCTURED INDEX: {SOURCE} / {country}"
    )
    print("=" * 70)


    safe_country = safe_country_name(
        country
    )


    prefix = (
        INDEX_DIR
        / f"{SOURCE.lower()}_{safe_country}"
    )


    structured_path = Path(
        f"{prefix}_structured.pkl"
    )

    records_path = Path(
        f"{prefix}_records.parquet"
    )


    # ========================================================
    # PASS 1: DOCUMENT FREQUENCIES
    # ========================================================

    print(
        "Pass 1: counting token/number frequencies..."
    )


    token_df = Counter()
    number_df = Counter()

    total_rows = 0

    started = time.perf_counter()


    reader = pd.read_csv(
        TARGET_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
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


        for address in chunk[
            "business_address"
        ].tolist():

            normalized = normalize_address(
                address
            )

            token_df.update(
                address_tokens(
                    normalized
                )
            )

            number_df.update(
                extract_numbers(
                    normalized
                )
            )


        total_rows += len(chunk)


        print(
            f"\r"
            f"chunk={chunk_no:<4} "
            f"rows={total_rows:,}",
            end="",
            flush=True,
        )


    print()


    # Keep only selective values.
    useful_tokens = {
        token
        for token, count in token_df.items()
        if count <= MAX_TOKEN_DF
    }


    useful_numbers = {
        number
        for number, count in number_df.items()
        if count <= MAX_NUMBER_DF
    }


    print(
        "Useful tokens:",
        f"{len(useful_tokens):,}",
    )

    print(
        "Useful numbers:",
        f"{len(useful_numbers):,}",
    )


    # ========================================================
    # PASS 2
    # ========================================================

    print(
        "\nPass 2: building postings + record cache..."
    )


    token_postings = defaultdict(
        list
    )

    number_postings = defaultdict(
        list
    )


    writer = None

    row_idx = 0


    reader = pd.read_csv(
        TARGET_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
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

        chunk = (
            chunk[
                chunk["country"]
                == country
            ]
            .copy()
        )


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


        entity_ids = (
            chunk["entity_id"]
            .tolist()
        )


        row_indices = np.arange(
            row_idx,
            row_idx + len(chunk),
            dtype=np.int32,
        )


        # ----------------------------------------------------
        # POSTINGS
        # ----------------------------------------------------

        for local_idx, address in enumerate(
            addresses
        ):

            global_idx = (
                row_idx
                + local_idx
            )


            for token in address_tokens(
                address
            ):

                if token in useful_tokens:

                    token_postings[
                        token
                    ].append(
                        global_idx
                    )


            for number in extract_numbers(
                address
            ):

                if number in useful_numbers:

                    number_postings[
                        number
                    ].append(
                        global_idx
                    )


        # ----------------------------------------------------
        # RECORD CACHE
        # ----------------------------------------------------

        out = pd.DataFrame(
            {
                "row_idx":
                    row_indices,

                "entity_id":
                    entity_ids,

                "name_norm":
                    names,

                "address_norm":
                    addresses,
            }
        )


        table = pa.Table.from_pandas(
            out,
            preserve_index=False,
        )


        if writer is None:

            writer = pq.ParquetWriter(
                records_path,
                table.schema,
                compression="zstd",
            )


        writer.write_table(
            table
        )


        row_idx += len(chunk)


        print(
            f"\r"
            f"chunk={chunk_no:<4} "
            f"rows={row_idx:,}",
            end="",
            flush=True,
        )


    print()


    if writer is not None:
        writer.close()


    # ========================================================
    # COMPACT POSTINGS
    # ========================================================

    print(
        "Compacting postings..."
    )


    token_postings = {
        token: np.asarray(
            rows,
            dtype=np.int32,
        )
        for token, rows
        in token_postings.items()
    }


    number_postings = {
        number: np.asarray(
            rows,
            dtype=np.int32,
        )
        for number, rows
        in number_postings.items()
    }


    structured_data = {
        "source":
            SOURCE,

        "country":
            country,

        "rows":
            row_idx,

        "max_token_df":
            MAX_TOKEN_DF,

        "max_number_df":
            MAX_NUMBER_DF,

        "token_df":
            {
                token: token_df[token]
                for token
                in token_postings
            },

        "number_df":
            {
                number: number_df[number]
                for number
                in number_postings
            },

        "token_postings":
            token_postings,

        "number_postings":
            number_postings,
    }


    print(
        "Saving structured index..."
    )


    with open(
        structured_path,
        "wb",
    ) as f:

        pickle.dump(
            structured_data,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


    elapsed = (
        time.perf_counter()
        - started
    )


    print()
    print(
        "Rows:",
        f"{row_idx:,}",
    )

    print(
        "Token posting keys:",
        f"{len(token_postings):,}",
    )

    print(
        "Number posting keys:",
        f"{len(number_postings):,}",
    )

    print(
        "Records:",
        records_path,
    )

    print(
        "Structured:",
        structured_path,
    )

    print(
        "Runtime:",
        f"{elapsed / 60:.2f} min",
    )