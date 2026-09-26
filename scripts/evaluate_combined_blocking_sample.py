import gc
import heapq
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


COUNTRY = "India"

SAMPLE_SIZE = 20_000
RANDOM_SEED = 42

K = 20

READ_CHUNK_SIZE = 100_000

N_FEATURES = 2 ** 18


warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
)


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


# =========================================================
# LOAD SAMPLE
# =========================================================

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

    print("\nLoading validation S1...")

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
        f"Sampled queries: "
        f"{len(df):,}"
    )

    print("Normalizing query data...")

    df["name_norm"] = (
        df["business_name"]
        .map(normalize_name)
    )

    df["address_norm"] = (
        df["business_address"]
        .map(normalize_address)
    )

    return df


# =========================================================
# GROUND TRUTH
# =========================================================

def load_truth(sample_ids):

    print("\nLoading sample ground truth...")

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    true_targets = defaultdict(set)

    for row in gt.itertuples(index=False):

        s1_id = row.source1_entity_id

        if s1_id not in sample_ids:
            continue

        for target_id in parse_matches(
            row.matched_entity_ids
        ):

            if target_id.startswith("S2-"):

                true_targets[s1_id].add(
                    target_id
                )

    total_links = sum(
        len(x)
        for x in true_targets.values()
    )

    print(
        f"True S2 links: "
        f"{total_links:,}"
    )

    print(
        f"Entities with >=1 S2 match: "
        f"{len(true_targets):,}"
    )

    return true_targets, total_links


# =========================================================
# VECTORIZERS
# =========================================================

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


# =========================================================
# EXACT BLOCK INDEX
# =========================================================

def build_exact_query_indexes(query_df):

    name_index = defaultdict(list)
    address_index = defaultdict(list)

    for i, row in enumerate(
        query_df.itertuples(index=False)
    ):

        if row.name_norm:

            name_index[
                row.name_norm
            ].append(i)

        if row.address_norm:

            address_index[
                row.address_norm
            ].append(i)

    return name_index, address_index


# =========================================================
# TOP-K MERGING
# =========================================================

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

            candidate = (
                float(score),
                corpus_ids[col],
            )

            if len(heap) < K:

                heapq.heappush(
                    heap,
                    candidate,
                )

            elif score > heap[0][0]:

                heapq.heapreplace(
                    heap,
                    candidate,
                )


# =========================================================
# STREAM S2
# =========================================================

def search_s2(
    query_df,
    name_query_matrix,
    address_query_matrix,
    name_vectorizer,
    address_vectorizer,
    exact_name_index,
    exact_address_index,
):

    n_queries = len(query_df)

    name_heaps = [
        []
        for _ in range(n_queries)
    ]

    address_heaps = [
        []
        for _ in range(n_queries)
    ]

    exact_candidates = [
        set()
        for _ in range(n_queries)
    ]

    total_read = 0
    total_india = 0

    chunk_number = 0

    start_time = now()

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

        chunk_number += 1
        total_read += len(chunk)

        chunk = chunk[
            chunk["country"] == COUNTRY
        ].copy()

        if chunk.empty:
            continue

        total_india += len(chunk)

        # ---------------------------------------------
        # Normalize this chunk once
        # ---------------------------------------------

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

        # ---------------------------------------------
        # Exact blocking
        # ---------------------------------------------

        for target_id, name, address in zip(
            corpus_ids,
            names,
            addresses,
        ):

            if name:

                for query_idx in (
                    exact_name_index.get(
                        name,
                        ()
                    )
                ):

                    exact_candidates[
                        query_idx
                    ].add(target_id)

            if address:

                for query_idx in (
                    exact_address_index.get(
                        address,
                        ()
                    )
                ):

                    exact_candidates[
                        query_idx
                    ].add(target_id)

        # ---------------------------------------------
        # Name retrieval
        # ---------------------------------------------

        name_matrix = (
            name_vectorizer.transform(
                names
            )
        )

        name_similarities = (
            awesome_cossim_topn(
                name_query_matrix,
                name_matrix.T,
                ntop=K,
                lower_bound=0.0,
            )
        )

        update_heaps(
            name_heaps,
            name_similarities,
            corpus_ids,
        )

        # ---------------------------------------------
        # Address retrieval
        # ---------------------------------------------

        address_matrix = (
            address_vectorizer.transform(
                addresses
            )
        )

        address_similarities = (
            awesome_cossim_topn(
                address_query_matrix,
                address_matrix.T,
                ntop=K,
                lower_bound=0.0,
            )
        )

        update_heaps(
            address_heaps,
            address_similarities,
            corpus_ids,
        )

        elapsed = (
            now() - start_time
        )

        print(
            f"\r"
            f"chunks={chunk_number:<3}"
            f" read={total_read:,}"
            f" India={total_india:,}"
            f" elapsed={elapsed:.1f}s",
            end="",
            flush=True,
        )

        del chunk
        del names
        del addresses
        del corpus_ids

        del name_matrix
        del address_matrix

        del name_similarities
        del address_similarities

        gc.collect()

    print()

    print(
        f"\nS2 India searched: "
        f"{total_india:,}"
    )

    print(
        f"Search time: "
        f"{now() - start_time:.2f}s"
    )

    # ---------------------------------------------
    # Convert heaps into sets
    # ---------------------------------------------

    name_candidates = []

    address_candidates = []

    for heap in name_heaps:

        name_candidates.append(
            {
                entity_id
                for score, entity_id
                in heap
            }
        )

    for heap in address_heaps:

        address_candidates.append(
            {
                entity_id
                for score, entity_id
                in heap
            }
        )

    return (
        exact_candidates,
        name_candidates,
        address_candidates,
    )


# =========================================================
# EVALUATION
# =========================================================

def evaluate_method(
    method_name,
    query_df,
    candidate_sets,
    true_targets,
    total_true_links,
):

    recovered_links = 0

    full_entities = 0
    partial_entities = 0
    zero_entities = 0

    total_candidates = 0

    entities_with_truth = len(
        true_targets
    )

    for i, row in enumerate(
        query_df.itertuples(index=False)
    ):

        s1_id = row.entity_id

        candidates = (
            candidate_sets[i]
        )

        total_candidates += len(
            candidates
        )

        true = true_targets.get(
            s1_id,
            set(),
        )

        if not true:
            continue

        recovered = (
            true & candidates
        )

        recovered_links += len(
            recovered
        )

        if len(recovered) == len(true):

            full_entities += 1

        elif len(recovered) == 0:

            zero_entities += 1

        else:

            partial_entities += 1

    pair_recall = (
        recovered_links
        / total_true_links
    )

    full_pct = (
        full_entities
        / entities_with_truth
    )

    zero_pct = (
        zero_entities
        / entities_with_truth
    )

    avg_candidates = (
        total_candidates
        / len(query_df)
    )

    return {
        "method": method_name,
        "recall": pair_recall,
        "avg_candidates": avg_candidates,
        "full_pct": full_pct,
        "zero_pct": zero_pct,
        "recovered": recovered_links,
    }


def union_sets(*candidate_lists):

    result = []

    n = len(
        candidate_lists[0]
    )

    for i in range(n):

        combined = set()

        for candidate_list in candidate_lists:

            combined.update(
                candidate_list[i]
            )

        result.append(combined)

    return result


# =========================================================
# UNIQUE CONTRIBUTION ANALYSIS
# =========================================================

def print_unique_recovery(
    query_df,
    exact,
    name,
    address,
    true_targets,
):

    exact_true = set()
    name_true = set()
    address_true = set()

    for i, row in enumerate(
        query_df.itertuples(index=False)
    ):

        s1_id = row.entity_id

        true = true_targets.get(
            s1_id,
            set(),
        )

        for target in true:

            link = (
                s1_id,
                target,
            )

            if target in exact[i]:
                exact_true.add(link)

            if target in name[i]:
                name_true.add(link)

            if target in address[i]:
                address_true.add(link)

    print("\n" + "=" * 80)
    print("UNIQUE TRUE LINKS CONTRIBUTED")
    print("=" * 80)

    print(
        "Exact only:",
        len(
            exact_true
            - name_true
            - address_true
        ),
    )

    print(
        "Name only:",
        len(
            name_true
            - exact_true
            - address_true
        ),
    )

    print(
        "Address only:",
        len(
            address_true
            - exact_true
            - name_true
        ),
    )

    print(
        "Name adds beyond address:",
        len(
            name_true
            - address_true
        ),
    )

    print(
        "Address adds beyond name:",
        len(
            address_true
            - name_true
        ),
    )


# =========================================================
# MAIN
# =========================================================

def main():

    total_start = now()

    print(
        "Combined S2 India blocking experiment"
    )

    print(
        f"Sample size: {SAMPLE_SIZE:,}"
    )

    print(
        f"Fuzzy K: {K}"
    )

    # -------------------------------------------------
    # Sample
    # -------------------------------------------------

    val_ids = load_validation_ids()

    query_df = load_query_sample(
        val_ids
    )

    sample_ids = set(
        query_df["entity_id"]
    )

    # -------------------------------------------------
    # GT
    # -------------------------------------------------

    (
        true_targets,
        total_true_links,
    ) = load_truth(
        sample_ids
    )

    # -------------------------------------------------
    # Vectorizers
    # -------------------------------------------------

    print("\nBuilding query vectors...")

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

    print(
        "Name query nnz:",
        f"{name_query_matrix.nnz:,}"
    )

    print(
        "Address query nnz:",
        f"{address_query_matrix.nnz:,}"
    )

    # -------------------------------------------------
    # Exact query indexes
    # -------------------------------------------------

    (
        exact_name_index,
        exact_address_index,
    ) = build_exact_query_indexes(
        query_df
    )

    # -------------------------------------------------
    # Search
    # -------------------------------------------------

    (
        exact,
        name,
        address,
    ) = search_s2(
        query_df=query_df,
        name_query_matrix=name_query_matrix,
        address_query_matrix=address_query_matrix,
        name_vectorizer=name_vectorizer,
        address_vectorizer=address_vectorizer,
        exact_name_index=exact_name_index,
        exact_address_index=exact_address_index,
    )

    # -------------------------------------------------
    # Unions
    # -------------------------------------------------

    exact_name = union_sets(
        exact,
        name,
    )

    exact_address = union_sets(
        exact,
        address,
    )

    name_address = union_sets(
        name,
        address,
    )

    all_three = union_sets(
        exact,
        name,
        address,
    )

    methods = [
        (
            "Exact",
            exact,
        ),
        (
            "Name K=20",
            name,
        ),
        (
            "Address K=20",
            address,
        ),
        (
            "Exact + Name",
            exact_name,
        ),
        (
            "Exact + Address",
            exact_address,
        ),
        (
            "Name + Address",
            name_address,
        ),
        (
            "Exact + Name + Address",
            all_three,
        ),
    ]

    results = []

    for method_name, candidates in methods:

        results.append(
            evaluate_method(
                method_name,
                query_df,
                candidates,
                true_targets,
                total_true_links,
            )
        )

    # -------------------------------------------------
    # Results
    # -------------------------------------------------

    print("\n")
    print("=" * 100)
    print("COMBINED BLOCKING RESULTS")
    print("=" * 100)

    print(
        f"{'Method':<30}"
        f"{'Recall':>12}"
        f"{'Avg Cand':>12}"
        f"{'Full Entity':>15}"
        f"{'Zero Recovery':>16}"
    )

    print("-" * 100)

    for result in results:

        print(
            f"{result['method']:<30}"
            f"{result['recall']:>11.2%}"
            f"{result['avg_candidates']:>12.2f}"
            f"{result['full_pct']:>14.2%}"
            f"{result['zero_pct']:>15.2%}"
        )

    print_unique_recovery(
        query_df,
        exact,
        name,
        address,
        true_targets,
    )

    print(
        f"\nTotal runtime: "
        f"{now() - total_start:.2f}s"
    )


if __name__ == "__main__":
    main()