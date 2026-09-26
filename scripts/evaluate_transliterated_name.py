import gc
import heapq
import pickle
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import HashingVectorizer
from sparse_dot_topn import awesome_cossim_topn
from unidecode import unidecode


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.normalize import normalize_name


TRAIN_DIR = ROOT / "data" / "train"
OUTPUT_DIR = ROOT / "outputs"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
S2_PATH = TRAIN_DIR / "train_source2.tsv"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"
SPLIT_PATH = OUTPUT_DIR / "splits.tsv"

BASE_CACHE_PATH = (
    OUTPUT_DIR
    / "s2_india_sample_candidates.pkl"
)

TRANSLIT_CACHE_PATH = (
    OUTPUT_DIR
    / "s2_india_translit_name_candidates.pkl"
)


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


def contains_non_ascii(text):
    return any(
        ord(ch) > 127
        for ch in str(text)
    )


def transliterate_name(text):
    if not text:
        return ""

    return normalize_name(
        unidecode(text)
    )


# ============================================================
# SAMPLE
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

    print("Loading same 20k S1 sample...")

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
                    ]
                ]
            )

    df = pd.concat(
        pieces,
        ignore_index=True,
    )

    df = df.sample(
        n=min(
            SAMPLE_SIZE,
            len(df),
        ),
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    df["name_translit"] = (
        df["business_name"]
        .map(transliterate_name)
    )

    print(
        f"Queries loaded: "
        f"{len(df):,}"
    )

    return df


# ============================================================
# TRUTH
# ============================================================

def load_truth(sample_ids):

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    truth = defaultdict(set)

    for row in gt.itertuples(
        index=False
    ):

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

    return truth, total


# ============================================================
# VECTORIZER
# ============================================================

def build_vectorizer():

    return HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        n_features=N_FEATURES,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )


# ============================================================
# TOP-K MERGE
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
# SEARCH ONLY NON-ASCII S2 NAMES
# ============================================================

def search_transliterated_names(
    query_matrix,
    vectorizer,
):

    print(
        "\nStreaming non-ASCII S2 India names..."
    )

    query_count = (
        query_matrix.shape[0]
    )

    heaps = [
        []
        for _ in range(
            query_count
        )
    ]

    total_read = 0
    india_read = 0
    non_ascii_read = 0

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

        india_read += len(chunk)

        if chunk.empty:
            continue

        mask = (
            chunk["business_name"]
            .map(contains_non_ascii)
        )

        chunk = chunk[
            mask
        ].copy()

        if chunk.empty:

            print(
                f"\r"
                f"chunks={chunk_number:<3}"
                f" read={total_read:,}"
                f" India={india_read:,}"
                f" nonASCII={non_ascii_read:,}"
                f" elapsed={now() - t0:.1f}s",
                end="",
                flush=True,
            )

            continue

        non_ascii_read += len(chunk)

        translit_names = (
            chunk["business_name"]
            .map(transliterate_name)
        )

        corpus_ids = (
            chunk["entity_id"]
            .to_numpy()
        )

        corpus_matrix = (
            vectorizer.transform(
                translit_names
            )
        )

        similarities = (
            awesome_cossim_topn(
                query_matrix,
                corpus_matrix.T,
                ntop=K,
                lower_bound=0.0,
            )
        )

        update_heaps(
            heaps,
            similarities,
            corpus_ids,
        )

        print(
            f"\r"
            f"chunks={chunk_number:<3}"
            f" read={total_read:,}"
            f" India={india_read:,}"
            f" nonASCII={non_ascii_read:,}"
            f" elapsed={now() - t0:.1f}s",
            end="",
            flush=True,
        )

        del chunk
        del translit_names
        del corpus_ids
        del corpus_matrix
        del similarities

        gc.collect()

    print()

    print(
        f"\nS2 India read: "
        f"{india_read:,}"
    )

    print(
        f"Non-ASCII names searched: "
        f"{non_ascii_read:,}"
    )

    print(
        f"Search time: "
        f"{now() - t0:.2f}s"
    )

    candidate_sets = []

    for heap in heaps:

        candidate_sets.append(
            {
                entity_id
                for score, entity_id
                in heap
            }
        )

    return candidate_sets


# ============================================================
# BASE CACHE
# ============================================================

def load_base_candidates(
    query_df,
):

    print(
        "\nLoading existing char-name/address cache..."
    )

    with open(
        BASE_CACHE_PATH,
        "rb",
    ) as f:

        cache = pickle.load(f)

    query_ids = (
        query_df["entity_id"]
        .tolist()
    )

    if (
        cache["query_ids"]
        != query_ids
    ):

        raise ValueError(
            "Base cache does not match "
            "this deterministic sample."
        )

    maps = cache[
        "candidates"
    ]

    name = []
    address = []

    for s1_id in query_ids:

        name.append(
            set(
                maps[
                    s1_id
                ]["name"]
            )
        )

        address.append(
            set(
                maps[
                    s1_id
                ]["address"]
            )
        )

    return name, address


# ============================================================
# STRUCTURED CACHE OPTIONAL
# ============================================================

def load_structured_if_present(
    query_df,
):

    path = (
        OUTPUT_DIR
        / "s2_india_structured_candidates.pkl"
    )

    if not path.exists():

        print(
            "\nStructured candidate cache "
            "not found."
        )

        print(
            "This run will evaluate "
            "transliteration against the "
            "cached name+address union only."
        )

        return None

    print(
        "\nLoading structured-address cache..."
    )

    with open(
        path,
        "rb",
    ) as f:

        cache = pickle.load(f)

    if (
        cache["query_ids"]
        != query_df[
            "entity_id"
        ].tolist()
    ):

        raise ValueError(
            "Structured cache sample mismatch."
        )

    return [
        set(x)
        for x in cache[
            "candidates"
        ]
    ]


# ============================================================
# UNION
# ============================================================

def union_lists(
    *candidate_lists,
):

    result = []

    for i in range(
        len(candidate_lists[0])
    ):

        combined = set()

        for candidate_list in (
            candidate_lists
        ):

            if candidate_list is None:
                continue

            combined.update(
                candidate_list[i]
            )

        result.append(
            combined
        )

    return result


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    label,
    query_df,
    candidate_sets,
    truth,
    total_true,
):

    recovered_links = 0

    full = 0
    partial = 0
    zero = 0

    total_candidates = 0

    entities_with_truth = len(
        truth
    )

    for i, row in enumerate(
        query_df.itertuples(
            index=False
        )
    ):

        s1_id = row.entity_id

        candidates = (
            candidate_sets[i]
        )

        total_candidates += len(
            candidates
        )

        true = truth.get(
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
            full += 1

        elif len(recovered) == 0:
            zero += 1

        else:
            partial += 1

    return {
        "label": label,

        "recall":
            recovered_links
            / total_true,

        "avg_candidates":
            total_candidates
            / len(query_df),

        "full":
            full
            / entities_with_truth,

        "zero":
            zero
            / entities_with_truth,

        "recovered":
            recovered_links,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = now()

    val_ids = load_validation_ids()

    query_df = load_query_sample(
        val_ids
    )

    sample_ids = set(
        query_df["entity_id"]
    )

    truth, total_true = load_truth(
        sample_ids
    )

    # ------------------------------------------------
    # Transliteration vectors
    # ------------------------------------------------

    print(
        "\nBuilding transliterated query vectors..."
    )

    vectorizer = (
        build_vectorizer()
    )

    query_matrix = (
        vectorizer.transform(
            query_df[
                "name_translit"
            ]
        )
    )

    print(
        f"Query nnz: "
        f"{query_matrix.nnz:,}"
    )

    # ------------------------------------------------
    # Search
    # ------------------------------------------------

    translit = (
        search_transliterated_names(
            query_matrix,
            vectorizer,
        )
    )

    # ------------------------------------------------
    # Save new channel immediately
    # ------------------------------------------------

    print(
        f"\nSaving transliteration cache:"
        f"\n{TRANSLIT_CACHE_PATH}"
    )

    with open(
        TRANSLIT_CACHE_PATH,
        "wb",
    ) as f:

        pickle.dump(
            {
                "query_ids":
                    query_df[
                        "entity_id"
                    ].tolist(),

                "candidates":
                    translit,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # ------------------------------------------------
    # Existing channels
    # ------------------------------------------------

    name, address = (
        load_base_candidates(
            query_df
        )
    )

    structured = (
        load_structured_if_present(
            query_df
        )
    )

    name_address = union_lists(
        name,
        address,
    )

    name_address_translit = union_lists(
        name,
        address,
        translit,
    )

    methods = [
        (
            "Original Name K20",
            name,
        ),
        (
            "Translit Name K20",
            translit,
        ),
        (
            "Name + Address",
            name_address,
        ),
        (
            "Name + Address + Translit",
            name_address_translit,
        ),
    ]

    # If you've added a structured cache,
    # we'll automatically evaluate the real
    # four-channel union too.
    if structured is not None:

        current_best = union_lists(
            name,
            address,
            structured,
        )

        all_four = union_lists(
            name,
            address,
            structured,
            translit,
        )

        methods.extend(
            [
                (
                    "Name + Address + Structured",
                    current_best,
                ),
                (
                    "Name + Address + Structured + Translit",
                    all_four,
                ),
            ]
        )

    results = []

    for label, candidates in methods:

        results.append(
            evaluate(
                label,
                query_df,
                candidates,
                truth,
                total_true,
            )
        )

    # ------------------------------------------------
    # OUTPUT
    # ------------------------------------------------

    print("\n")
    print("=" * 110)
    print(
        "TRANSLITERATED NAME BLOCKING RESULTS"
    )
    print("=" * 110)

    print(
        f"{'Method':<48}"
        f"{'Recall':>10}"
        f"{'Avg Cand':>12}"
        f"{'Full Entity':>15}"
        f"{'Zero Recovery':>16}"
    )

    print("-" * 110)

    for x in results:

        print(
            f"{x['label']:<48}"
            f"{x['recall']:>9.2%}"
            f"{x['avg_candidates']:>12.2f}"
            f"{x['full']:>14.2%}"
            f"{x['zero']:>15.2%}"
        )

    print(
        f"\nTotal runtime: "
        f"{now() - total_start:.2f}s"
    )


if __name__ == "__main__":
    main()