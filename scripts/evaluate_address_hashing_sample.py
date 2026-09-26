import gc
import heapq
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import HashingVectorizer
from sparse_dot_topn import awesome_cossim_topn


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.normalize import normalize_address


TRAIN_DIR = ROOT / "data" / "train"
OUTPUT_DIR = ROOT / "outputs"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
S2_PATH = TRAIN_DIR / "train_source2.tsv"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"
SPLIT_PATH = OUTPUT_DIR / "splits.tsv"


COUNTRY = "India"

SAMPLE_SIZE = 20_000
RANDOM_SEED = 42

K_VALUES = [5, 10, 20]
MAX_K = max(K_VALUES)

READ_CHUNK_SIZE = 100_000

N_FEATURES = 2 ** 18


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


def load_india_validation_s1(val_ids):
    print("\nLoading India validation S1 records...")

    t0 = now()

    pieces = []

    reader = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_address",
            "country",
        ],
        chunksize=READ_CHUNK_SIZE,
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

    if len(df) > SAMPLE_SIZE:
        df = df.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_SEED,
        ).reset_index(drop=True)

    print(
        f"Sampled queries: "
        f"{len(df):,}"
    )

    df["address_norm"] = (
        df["business_address"]
        .map(normalize_address)
    )

    print(
        f"Load + sample time: "
        f"{now() - t0:.2f}s"
    )

    return df


def load_sample_truth(sample_ids):
    print("\nLoading ground truth for sample...")

    t0 = now()

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    true_targets = defaultdict(set)

    total_links = 0

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

                total_links += 1

    print(
        f"True S2 links for sampled "
        f"India entities: {total_links:,}"
    )

    print(
        f"Sampled S1 with >=1 S2 match: "
        f"{len(true_targets):,}"
    )

    print(
        f"Truth load time: "
        f"{now() - t0:.2f}s"
    )

    return true_targets, total_links


def build_vectorizer():
    return HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        n_features=N_FEATURES,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )


def update_global_topk(
    heaps,
    similarities,
    corpus_ids,
):
    for query_idx in range(
        similarities.shape[0]
    ):

        start = similarities.indptr[
            query_idx
        ]

        end = similarities.indptr[
            query_idx + 1
        ]

        cols = similarities.indices[
            start:end
        ]

        scores = similarities.data[
            start:end
        ]

        heap = heaps[query_idx]

        for col, score in zip(
            cols,
            scores,
        ):

            candidate = (
                float(score),
                corpus_ids[col],
            )

            if len(heap) < MAX_K:

                heapq.heappush(
                    heap,
                    candidate,
                )

            elif score > heap[0][0]:

                heapq.heapreplace(
                    heap,
                    candidate,
                )


def retrieve_from_all_s2_india(
    query_matrix,
    vectorizer,
):
    print(
        "\nStreaming complete S2 corpus..."
    )

    query_count = query_matrix.shape[0]

    heaps = [
        []
        for _ in range(query_count)
    ]

    total_read = 0
    total_india = 0

    chunk_number = 0

    t0 = now()

    reader = pd.read_csv(
        S2_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_address",
            "country",
        ],
        chunksize=READ_CHUNK_SIZE,
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

        # Missing target addresses exist in S2.
        chunk = chunk[
            chunk["business_address"] != ""
        ].copy()

        if chunk.empty:
            continue

        addresses = (
            chunk["business_address"]
            .map(normalize_address)
        )

        corpus_ids = (
            chunk["entity_id"]
            .to_numpy()
        )

        corpus_matrix = (
            vectorizer.transform(
                addresses
            )
        )

        similarities = (
            awesome_cossim_topn(
                query_matrix,
                corpus_matrix.T,
                ntop=MAX_K,
                lower_bound=0.0,
            )
        )

        update_global_topk(
            heaps,
            similarities,
            corpus_ids,
        )

        elapsed = now() - t0

        print(
            f"\r"
            f"Chunks={chunk_number:<3}"
            f" read={total_read:,}"
            f" India={total_india:,}"
            f" elapsed={elapsed:.1f}s",
            end="",
            flush=True,
        )

        del chunk
        del addresses
        del corpus_ids
        del corpus_matrix
        del similarities

        gc.collect()

    print()

    print(
        f"\nS2 India searched: "
        f"{total_india:,}"
    )

    print(
        f"Corpus search time: "
        f"{now() - t0:.2f}s"
    )

    ranked = []

    for heap in heaps:

        ranked.append(
            sorted(
                heap,
                key=lambda x: x[0],
                reverse=True,
            )
        )

    return ranked


def evaluate(
    query_df,
    ranked_candidates,
    true_targets,
    total_true_links,
):
    print("\n")
    print("=" * 80)
    print("S2 INDIA HASHING ADDRESS BLOCKER")
    print("=" * 80)

    entities_with_truth = len(
        true_targets
    )

    print(
        f"Queries: {len(query_df):,}"
    )

    print(
        f"True S2 links: "
        f"{total_true_links:,}"
    )

    print(
        f"Entities with >=1 S2 match: "
        f"{entities_with_truth:,}"
    )

    print()

    for k in K_VALUES:

        recovered_links = 0

        full_entities = 0
        partial_entities = 0
        zero_entities = 0

        for i, row in enumerate(
            query_df.itertuples(index=False)
        ):

            s1_id = row.entity_id

            true = true_targets.get(
                s1_id,
                set(),
            )

            if not true:
                continue

            candidates = {
                entity_id
                for _, entity_id
                in ranked_candidates[i][:k]
            }

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

        print(
            f"K={k}"
        )

        print(
            f"  Recovered links: "
            f"{recovered_links:,}"
            f" / {total_true_links:,}"
        )

        print(
            f"  Pair recall: "
            f"{pair_recall:.4%}"
        )

        print(
            f"  Full entity recovery: "
            f"{full_entities:,}"
            f" ({full_pct:.2%})"
        )

        print(
            f"  Partial entity recovery: "
            f"{partial_entities:,}"
        )

        print(
            f"  Zero entity recovery: "
            f"{zero_entities:,}"
            f" ({zero_pct:.2%})"
        )

        print()


def main():

    total_start = now()

    print(
        "Reduced fuzzy-address experiment"
    )

    print(
        f"Country: {COUNTRY}"
    )

    print(
        f"Sample size: {SAMPLE_SIZE:,}"
    )

    print(
        f"K values: {K_VALUES}"
    )

    print(
        f"Hash features: {N_FEATURES:,}"
    )

    val_ids = load_validation_ids()

    query_df = load_india_validation_s1(
        val_ids
    )

    sample_ids = set(
        query_df["entity_id"]
    )

    (
        true_targets,
        total_true_links,
    ) = load_sample_truth(
        sample_ids
    )

    print(
        "\nBuilding address query vectors..."
    )

    t0 = now()

    vectorizer = build_vectorizer()

    query_matrix = (
        vectorizer.transform(
            query_df["address_norm"]
        )
    )

    print(
        f"Query matrix: "
        f"{query_matrix.shape}"
    )

    print(
        f"Query nnz: "
        f"{query_matrix.nnz:,}"
    )

    print(
        f"Query vectorization time: "
        f"{now() - t0:.2f}s"
    )

    ranked_candidates = (
        retrieve_from_all_s2_india(
            query_matrix,
            vectorizer,
        )
    )

    evaluate(
        query_df,
        ranked_candidates,
        true_targets,
        total_true_links,
    )

    print(
        f"Total runtime: "
        f"{now() - total_start:.2f}s"
    )


if __name__ == "__main__":
    main()