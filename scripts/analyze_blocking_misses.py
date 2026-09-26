import gc
import heapq
import pickle
import re
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import HashingVectorizer
from sparse_dot_topn import awesome_cossim_topn


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.normalize import normalize_name, normalize_address


TRAIN_DIR = ROOT / "data" / "train"
OUTPUT_DIR = ROOT / "outputs"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
S2_PATH = TRAIN_DIR / "train_source2.tsv"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"
SPLIT_PATH = OUTPUT_DIR / "splits.tsv"

MISS_OUTPUT = (
    OUTPUT_DIR
    / "s2_india_blocking_misses.csv"
)

CACHE_OUTPUT = (
    OUTPUT_DIR
    / "s2_india_sample_candidates.pkl"
)


COUNTRY = "India"

SAMPLE_SIZE = 20_000
RANDOM_SEED = 42

K = 20

READ_CHUNK_SIZE = 100_000
N_FEATURES = 2 ** 18

MISS_SAMPLE_SIZE = 50


warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
)


NUMBER_RE = re.compile(r"\d+")


def now():
    return time.perf_counter()


def parse_matches(value):
    if not value:
        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


def tokenize(text):
    if not text:
        return set()

    return set(text.split())


def jaccard(a, b):
    if not a and not b:
        return 0.0

    union = a | b

    if not union:
        return 0.0

    return len(a & b) / len(union)


def extract_numbers(text):
    if not text:
        return set()

    return set(
        NUMBER_RE.findall(text)
    )


# ============================================================
# LOAD VALIDATION SAMPLE
# ============================================================

def load_validation_ids():

    splits = pd.read_csv(
        SPLIT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "source1_entity_id",
            "split",
        ],
    )

    return set(
        splits.loc[
            splits["split"] == "val",
            "source1_entity_id",
        ]
    )


def load_query_sample(val_ids):

    print("\nLoading India validation S1...")

    pieces = []

    reader = pd.read_csv(
        S1_PATH,
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

    for chunk in reader:

        filtered = chunk[
            (chunk["country"] == COUNTRY)
            &
            (chunk["entity_id"].isin(val_ids))
        ].copy()

        if not filtered.empty:
            pieces.append(
                filtered[
                    [
                        "entity_id",
                        "business_name",
                        "business_address",
                    ]
                ]
            )

    df = pd.concat(
        pieces,
        ignore_index=True,
    )

    print(
        f"India validation entities: "
        f"{len(df):,}"
    )

    df = df.sample(
        n=min(SAMPLE_SIZE, len(df)),
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    print(
        f"Sampled: {len(df):,}"
    )

    df["name_norm"] = (
        df["business_name"]
        .map(normalize_name)
    )

    df["address_norm"] = (
        df["business_address"]
        .map(normalize_address)
    )

    return df


# ============================================================
# GROUND TRUTH
# ============================================================

def load_truth(sample_ids):

    print("\nLoading sample ground truth...")

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    truth = defaultdict(set)

    for row in gt.itertuples(index=False):

        s1_id = row.source1_entity_id

        if s1_id not in sample_ids:
            continue

        for target_id in parse_matches(
            row.matched_entity_ids
        ):

            if target_id.startswith("S2-"):
                truth[s1_id].add(
                    target_id
                )

    total = sum(
        len(x)
        for x in truth.values()
    )

    print(
        f"True S2 links: {total:,}"
    )

    return truth


# ============================================================
# VECTORIZERS
# ============================================================

def build_name_vectorizer():

    return HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        n_features=N_FEATURES,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )


def build_address_vectorizer():

    return HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        n_features=N_FEATURES,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )


# ============================================================
# TOP K
# ============================================================

def update_heaps(
    heaps,
    similarities,
    corpus_ids,
):

    for query_idx in range(
        similarities.shape[0]
    ):

        start = (
            similarities.indptr[
                query_idx
            ]
        )

        end = (
            similarities.indptr[
                query_idx + 1
            ]
        )

        cols = (
            similarities.indices[
                start:end
            ]
        )

        scores = (
            similarities.data[
                start:end
            ]
        )

        heap = heaps[query_idx]

        for col, score in zip(
            cols,
            scores,
        ):

            item = (
                float(score),
                corpus_ids[col],
            )

            if len(heap) < K:

                heapq.heappush(
                    heap,
                    item,
                )

            elif score > heap[0][0]:

                heapq.heapreplace(
                    heap,
                    item,
                )


# ============================================================
# SEARCH COMPLETE S2 INDIA
# ============================================================

def search_s2(
    name_query_matrix,
    address_query_matrix,
    name_vectorizer,
    address_vectorizer,
):

    query_count = (
        name_query_matrix.shape[0]
    )

    name_heaps = [
        []
        for _ in range(query_count)
    ]

    address_heaps = [
        []
        for _ in range(query_count)
    ]

    total_read = 0
    india_read = 0

    t0 = now()

    reader = pd.read_csv(
        S2_PATH,
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

    for chunk_number, chunk in enumerate(
        reader,
        start=1,
    ):

        total_read += len(chunk)

        chunk = chunk[
            chunk["country"] == COUNTRY
        ].copy()

        if chunk.empty:
            continue

        india_read += len(chunk)

        names = (
            chunk["business_name"]
            .map(normalize_name)
        )

        addresses = (
            chunk["business_address"]
            .map(normalize_address)
        )

        corpus_ids = (
            chunk["entity_id"]
            .to_numpy()
        )

        # ---------------- NAME ----------------

        name_matrix = (
            name_vectorizer.transform(
                names
            )
        )

        name_sim = (
            awesome_cossim_topn(
                name_query_matrix,
                name_matrix.T,
                ntop=K,
                lower_bound=0.0,
            )
        )

        update_heaps(
            name_heaps,
            name_sim,
            corpus_ids,
        )

        # ---------------- ADDRESS ----------------

        address_matrix = (
            address_vectorizer.transform(
                addresses
            )
        )

        address_sim = (
            awesome_cossim_topn(
                address_query_matrix,
                address_matrix.T,
                ntop=K,
                lower_bound=0.0,
            )
        )

        update_heaps(
            address_heaps,
            address_sim,
            corpus_ids,
        )

        print(
            f"\rchunks={chunk_number:<3}"
            f" read={total_read:,}"
            f" India={india_read:,}"
            f" elapsed={now() - t0:.1f}s",
            end="",
            flush=True,
        )

        del chunk
        del names
        del addresses
        del corpus_ids

        del name_matrix
        del address_matrix

        del name_sim
        del address_sim

        gc.collect()

    print()

    print(
        f"\nSearch complete in "
        f"{now() - t0:.2f}s"
    )

    return (
        name_heaps,
        address_heaps,
    )


# ============================================================
# CONVERT RESULTS
# ============================================================

def heap_to_dict(heap):

    return {
        entity_id: score
        for score, entity_id in heap
    }


def create_candidate_maps(
    query_df,
    name_heaps,
    address_heaps,
):

    result = {}

    for i, s1_id in enumerate(
        query_df["entity_id"]
    ):

        result[s1_id] = {
            "name": heap_to_dict(
                name_heaps[i]
            ),
            "address": heap_to_dict(
                address_heaps[i]
            ),
        }

    return result


# ============================================================
# IDENTIFY MISSES
# ============================================================

def find_missed_links(
    candidate_maps,
    truth,
):

    missed = []

    recovered = 0

    total = 0

    for s1_id, targets in truth.items():

        name_candidates = (
            candidate_maps[s1_id]["name"]
        )

        address_candidates = (
            candidate_maps[s1_id][
                "address"
            ]
        )

        combined = (
            set(name_candidates)
            |
            set(address_candidates)
        )

        for target_id in targets:

            total += 1

            if target_id in combined:

                recovered += 1

            else:

                missed.append(
                    (
                        s1_id,
                        target_id,
                    )
                )

    print("\n" + "=" * 75)
    print("MISS SUMMARY")
    print("=" * 75)

    print(
        f"Total true links: "
        f"{total:,}"
    )

    print(
        f"Recovered: "
        f"{recovered:,}"
    )

    print(
        f"Missed: "
        f"{len(missed):,}"
    )

    print(
        f"Recall: "
        f"{recovered / total:.4%}"
    )

    return missed


# ============================================================
# LOAD TRUE TARGET RECORDS
# ============================================================

def load_missed_s2_records(
    missed_target_ids,
):

    print(
        "\nLoading missed S2 records..."
    )

    rows = []

    reader = pd.read_csv(
        S2_PATH,
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

    for chunk in reader:

        filtered = chunk[
            chunk["entity_id"].isin(
                missed_target_ids
            )
        ]

        if not filtered.empty:
            rows.append(filtered)

    return pd.concat(
        rows,
        ignore_index=True,
    )


# ============================================================
# MISS FEATURES
# ============================================================

def build_miss_dataframe(
    query_df,
    target_df,
    missed,
):

    s1_lookup = (
        query_df
        .set_index("entity_id")
    )

    target_lookup = (
        target_df
        .set_index("entity_id")
    )

    rows = []

    for s1_id, target_id in missed:

        s1 = s1_lookup.loc[s1_id]
        s2 = target_lookup.loc[target_id]

        s1_name_norm = normalize_name(
            s1.business_name
        )

        s2_name_norm = normalize_name(
            s2.business_name
        )

        s1_addr_norm = normalize_address(
            s1.business_address
        )

        s2_addr_norm = normalize_address(
            s2.business_address
        )

        name_tokens_1 = tokenize(
            s1_name_norm
        )

        name_tokens_2 = tokenize(
            s2_name_norm
        )

        addr_tokens_1 = tokenize(
            s1_addr_norm
        )

        addr_tokens_2 = tokenize(
            s2_addr_norm
        )

        nums_1 = extract_numbers(
            s1_addr_norm
        )

        nums_2 = extract_numbers(
            s2_addr_norm
        )

        rows.append(
            {
                "s1_id": s1_id,
                "s2_id": target_id,

                "s1_name": (
                    s1.business_name
                ),

                "s2_name": (
                    s2.business_name
                ),

                "s1_address": (
                    s1.business_address
                ),

                "s2_address": (
                    s2.business_address
                ),

                "name_token_jaccard":
                    jaccard(
                        name_tokens_1,
                        name_tokens_2,
                    ),

                "address_token_jaccard":
                    jaccard(
                        addr_tokens_1,
                        addr_tokens_2,
                    ),

                "s1_address_numbers":
                    ",".join(
                        sorted(nums_1)
                    ),

                "s2_address_numbers":
                    ",".join(
                        sorted(nums_2)
                    ),

                "number_overlap":
                    len(
                        nums_1 & nums_2
                    ),

                "target_address_missing":
                    int(
                        s2_addr_norm == ""
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# ANALYSIS
# ============================================================

def analyze_misses(df):

    print("\n" + "=" * 75)
    print("MISS CHARACTERISTICS")
    print("=" * 75)

    print(
        f"Missed links: {len(df):,}"
    )

    print(
        "\nTarget address missing:"
    )

    print(
        df[
            "target_address_missing"
        ].value_counts(
            normalize=True
        )
    )

    print(
        "\nName token Jaccard:"
    )

    print(
        df[
            "name_token_jaccard"
        ].describe(
            percentiles=[
                .25,
                .5,
                .75,
                .9,
            ]
        )
    )

    print(
        "\nAddress token Jaccard:"
    )

    print(
        df[
            "address_token_jaccard"
        ].describe(
            percentiles=[
                .25,
                .5,
                .75,
                .9,
            ]
        )
    )

    has_number_overlap = (
        df["number_overlap"] > 0
    ).mean()

    print(
        "\nMissed pairs with >=1 "
        "common address number:"
    )

    print(
        f"{has_number_overlap:.2%}"
    )


def print_examples(df):

    print("\n")
    print("=" * 75)
    print("RANDOM MISSED EXAMPLES")
    print("=" * 75)

    sample = df.sample(
        n=min(
            MISS_SAMPLE_SIZE,
            len(df),
        ),
        random_state=42,
    )

    for i, row in enumerate(
        sample.itertuples(index=False),
        start=1,
    ):

        print(
            f"\n--- MISS {i} ---"
        )

        print(
            "S1 name:",
            row.s1_name,
        )

        print(
            "S2 name:",
            row.s2_name,
        )

        print(
            "Name token Jaccard:",
            f"{row.name_token_jaccard:.3f}",
        )

        print()

        print(
            "S1 address:",
            row.s1_address,
        )

        print(
            "S2 address:",
            row.s2_address,
        )

        print(
            "Address token Jaccard:",
            f"{row.address_token_jaccard:.3f}",
        )

        print(
            "S1 numbers:",
            row.s1_address_numbers,
        )

        print(
            "S2 numbers:",
            row.s2_address_numbers,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = now()

    print(
        "Blocking miss analysis"
    )

    # ------------------------------------------------
    # Sample
    # ------------------------------------------------

    val_ids = load_validation_ids()

    query_df = load_query_sample(
        val_ids
    )

    sample_ids = set(
        query_df["entity_id"]
    )

    truth = load_truth(
        sample_ids
    )

    # ------------------------------------------------
    # Query vectors
    # ------------------------------------------------

    name_vectorizer = (
        build_name_vectorizer()
    )

    address_vectorizer = (
        build_address_vectorizer()
    )

    name_query_matrix = (
        name_vectorizer.transform(
            query_df["name_norm"]
        )
    )

    address_query_matrix = (
        address_vectorizer.transform(
            query_df["address_norm"]
        )
    )

    # ------------------------------------------------
    # Search
    # ------------------------------------------------

    (
        name_heaps,
        address_heaps,
    ) = search_s2(
        name_query_matrix,
        address_query_matrix,
        name_vectorizer,
        address_vectorizer,
    )

    candidate_maps = (
        create_candidate_maps(
            query_df,
            name_heaps,
            address_heaps,
        )
    )

    # ------------------------------------------------
    # CACHE CANDIDATES
    # ------------------------------------------------

    print(
        f"\nSaving candidate cache to:"
        f"\n{CACHE_OUTPUT}"
    )

    with open(
        CACHE_OUTPUT,
        "wb",
    ) as f:

        pickle.dump(
            {
                "query_ids":
                    query_df[
                        "entity_id"
                    ].tolist(),

                "candidates":
                    candidate_maps,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # ------------------------------------------------
    # Find misses
    # ------------------------------------------------

    missed = find_missed_links(
        candidate_maps,
        truth,
    )

    missed_target_ids = {
        target_id
        for _, target_id in missed
    }

    # ------------------------------------------------
    # Fetch true S2 records
    # ------------------------------------------------

    target_df = (
        load_missed_s2_records(
            missed_target_ids
        )
    )

    # ------------------------------------------------
    # Build analysis table
    # ------------------------------------------------

    miss_df = build_miss_dataframe(
        query_df,
        target_df,
        missed,
    )

    miss_df.to_csv(
        MISS_OUTPUT,
        index=False,
    )

    print(
        f"\nSaved miss table to:"
        f"\n{MISS_OUTPUT}"
    )

    analyze_misses(
        miss_df
    )

    print_examples(
        miss_df
    )

    print(
        f"\nTotal runtime: "
        f"{now() - total_start:.2f}s"
    )


if __name__ == "__main__":
    main()