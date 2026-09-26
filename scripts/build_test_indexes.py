from pathlib import Path
from collections import Counter, defaultdict
import argparse
import gc
import json
import pickle
import re
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.normalize import normalize_name, normalize_address


TEST_DIR = ROOT / "data" / "test"
INDEX_DIR = ROOT / "outputs" / "test_indexes"

INDEX_DIR.mkdir(parents=True, exist_ok=True)


READ_CHUNK_SIZE = 100_000
HASH_FEATURES = 2 ** 18

MAX_TOKEN_DF = 500
MAX_NUMBER_DF = 1000

NUMBER_RE = re.compile(r"\d+")


NAME_STOP_TOKENS = {
    "private", "limited", "ltd", "llp",
    "pvt", "inc", "incorporated",
    "corp", "corporation", "company", "co",
}


ADDRESS_STOP_TOKENS = {
    "road", "rd",
    "street", "st",
    "lane", "ln",
    "floor", "fl",
    "building", "bldg",
    "house",
    "no", "number",
    "plot",
    "shop",
    "door",
    "near",
    "opp", "opposite",
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


def structured_tokens(text):
    if not text:
        return set()

    return {
        token
        for token in text.split()
        if (
            token not in ADDRESS_STOP_TOKENS
            and not token.isdigit()
            and len(token) > 1
        )
    }


def prefix_key(token):

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


def name_route_keys(normalized_name):

    keys = set()

    for token in normalized_name.split():

        if token in NAME_STOP_TOKENS:
            continue

        key = prefix_key(token)

        if key:
            keys.add(key)

    return keys


def address_route_keys(normalized_address):

    keys = set()

    for token in normalized_address.split():

        cleaned = "".join(
            ch
            for ch in token
            if ch.isalnum()
        )

        if not cleaned:
            continue

        if cleaned.isdigit():

            keys.add(
                f"n:{canonical_number(cleaned)}"
            )

            continue

        if token in ADDRESS_STOP_TOKENS:
            continue

        key = prefix_key(token)

        if key:
            keys.add(key)

    return keys


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


def csr_mb(matrix):

    return (
        matrix.data.nbytes
        + matrix.indices.nbytes
        + matrix.indptr.nbytes
    ) / (1024 ** 2)


def discover_countries(path):

    countries = set()

    for chunk in pd.read_csv(
        path,
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

    return sorted(countries)


def build_country(
    source,
    path,
    country,
):

    safe_country = safe_country_name(
        country
    )

    prefix = (
        INDEX_DIR
        / f"{source.lower()}_{safe_country}"
    )


    print()
    print("=" * 70)
    print(
        f"{source} / {country}"
    )
    print("=" * 70)


    started = time.perf_counter()


    # ========================================================
    # PASS 1 — structured document frequencies
    # ========================================================

    print(
        "Pass 1: structured DF counts..."
    )


    token_df = Counter()
    number_df = Counter()

    total_rows = 0


    for chunk_no, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=[
                "business_address",
                "country",
            ],
            chunksize=READ_CHUNK_SIZE,
        ),
        start=1,
    ):

        chunk = chunk[
            chunk["country"] == country
        ]


        for address in chunk[
            "business_address"
        ].tolist():

            address = normalize_address(
                address
            )

            token_df.update(
                structured_tokens(
                    address
                )
            )

            number_df.update(
                extract_numbers(
                    address
                )
            )


        total_rows += len(chunk)


        print(
            f"\r"
            f"pass1 chunk={chunk_no:<4} "
            f"rows={total_rows:,}",
            end="",
            flush=True,
        )


    print()


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
        "Useful structured tokens:",
        f"{len(useful_tokens):,}",
    )

    print(
        "Useful numbers:",
        f"{len(useful_numbers):,}",
    )


    # ========================================================
    # PASS 2 — build everything in identical row order
    # ========================================================

    print(
        "Pass 2: matrices + routing + structured..."
    )


    name_vectorizer = (
        make_name_vectorizer()
    )

    address_vectorizer = (
        make_address_vectorizer()
    )


    name_parts = []
    address_parts = []

    entity_ids = []


    name_routing = defaultdict(list)
    address_routing = defaultdict(list)

    token_postings = defaultdict(list)
    number_postings = defaultdict(list)


    records_path = Path(
        f"{prefix}_records.parquet"
    )

    writer = None

    row_idx = 0


    reader = pd.read_csv(
        path,
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


        ids = (
            chunk["entity_id"]
            .tolist()
        )


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


        # Sparse matrices.
        name_parts.append(
            name_vectorizer
            .transform(names)
            .tocsr()
        )


        address_parts.append(
            address_vectorizer
            .transform(addresses)
            .tocsr()
        )


        entity_ids.extend(
            ids
        )


        # Routing and structured postings.
        for local_idx, (
            name,
            address,
        ) in enumerate(
            zip(
                names,
                addresses,
            )
        ):

            global_idx = (
                row_idx
                + local_idx
            )


            for key in name_route_keys(
                name
            ):

                name_routing[
                    key
                ].append(
                    global_idx
                )


            for key in address_route_keys(
                address
            ):

                address_routing[
                    key
                ].append(
                    global_idx
                )


            for token in structured_tokens(
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


        # Record cache required by HybridBlocker.
        out = pd.DataFrame(
            {
                "row_idx":
                    np.arange(
                        row_idx,
                        row_idx + len(chunk),
                        dtype=np.int32,
                    ),

                "entity_id":
                    ids,

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


        row_idx += len(
            chunk
        )


        print(
            f"\r"
            f"pass2 chunk={chunk_no:<4} "
            f"rows={row_idx:,}",
            end="",
            flush=True,
        )


    print()


    if writer is not None:
        writer.close()


    # ========================================================
    # COMBINE SPARSE MATRICES
    # ========================================================

    print(
        "Combining sparse matrices..."
    )


    name_full = sparse.vstack(
        name_parts,
        format="csr",
        dtype=np.float32,
    )


    address_full = sparse.vstack(
        address_parts,
        format="csr",
        dtype=np.float32,
    )


    del name_parts
    del address_parts

    gc.collect()


    sparse.save_npz(
        Path(
            f"{prefix}_name.npz"
        ),
        name_full,
        compressed=True,
    )


    sparse.save_npz(
        Path(
            f"{prefix}_address.npz"
        ),
        address_full,
        compressed=True,
    )


    np.save(
        Path(
            f"{prefix}_ids.npy"
        ),
        np.asarray(
            entity_ids,
            dtype=str,
        ),
    )


    metadata = {
        "source":
            source,

        "country":
            country,

        "rows":
            row_idx,

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
    }


    with open(
        Path(
            f"{prefix}_meta.json"
        ),
        "w",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )


    # ========================================================
    # ROUTING
    # ========================================================

    print(
        "Saving routing index..."
    )


    name_routing = {
        key:
            np.asarray(
                rows,
                dtype=np.int32,
            )
        for key, rows
        in name_routing.items()
    }


    address_routing = {
        key:
            np.asarray(
                rows,
                dtype=np.int32,
            )
        for key, rows
        in address_routing.items()
    }


    with open(
        Path(
            f"{prefix}_routing.pkl"
        ),
        "wb",
    ) as f:

        pickle.dump(
            {
                "source":
                    source,

                "country":
                    country,

                "rows":
                    row_idx,

                "name_postings":
                    name_routing,

                "address_postings":
                    address_routing,

                "name_df":
                    {
                        key: len(rows)
                        for key, rows
                        in name_routing.items()
                    },

                "address_df":
                    {
                        key: len(rows)
                        for key, rows
                        in address_routing.items()
                    },
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


    # ========================================================
    # STRUCTURED
    # ========================================================

    print(
        "Saving structured index..."
    )


    token_postings = {
        token:
            np.asarray(
                rows,
                dtype=np.int32,
            )
        for token, rows
        in token_postings.items()
    }


    number_postings = {
        number:
            np.asarray(
                rows,
                dtype=np.int32,
            )
        for number, rows
        in number_postings.items()
    }


    with open(
        Path(
            f"{prefix}_structured.pkl"
        ),
        "wb",
    ) as f:

        pickle.dump(
            {
                "source":
                    source,

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
                        token:
                            token_df[token]
                        for token
                        in token_postings
                    },

                "number_df":
                    {
                        number:
                            number_df[number]
                        for number
                        in number_postings
                    },

                "token_postings":
                    token_postings,

                "number_postings":
                    number_postings,
            },
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
        "Name sparse:",
        f"{csr_mb(name_full):.2f} MB",
    )

    print(
        "Address sparse:",
        f"{csr_mb(address_full):.2f} MB",
    )

    print(
        "Runtime:",
        f"{elapsed / 60:.2f} min",
    )


    del name_full
    del address_full

    gc.collect()


def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--source",
        required=True,
        choices=[
            "S2",
            "S3",
        ],
    )


    args = parser.parse_args()


    source = (
        args.source.upper()
    )


    target_path = (
        TEST_DIR
        / (
            "test_source2.tsv"
            if source == "S2"
            else "test_source3.tsv"
        )
    )


    if not target_path.exists():

        raise FileNotFoundError(
            target_path
        )


    countries = discover_countries(
        target_path
    )


    print(
        "Countries discovered:",
        countries,
    )


    for country in countries:

        build_country(
            source,
            target_path,
            country,
        )


if __name__ == "__main__":
    main()