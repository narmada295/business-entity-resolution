import gc
import heapq
import math
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from rapidfuzz.fuzz import ratio, token_sort_ratio, token_set_ratio
from unidecode import unidecode


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.normalize import normalize_name, normalize_address


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

RERANK_CACHE_PATH = (
    OUTPUT_DIR
    / "s2_india_structured_reranked_candidates.pkl"
)


COUNTRY = "India"

SAMPLE_SIZE = 20_000
RANDOM_SEED = 42

# Generate a wider structured pool...
INTERNAL_K = 100

# ...but only keep this many after reranking.
FINAL_K = 25

READ_CHUNK_SIZE = 100_000

MAX_TOKEN_DF = 500
MAX_NUMBER_DF = 1000


NUMBER_RE = re.compile(r"\d+")


STOP_TOKENS = {
    "road",
    "rd",
    "street",
    "st",
    "floor",
    "plot",
    "house",
    "hno",
    "door",
    "no",
    "number",
    "near",
    "opp",
    "opposite",
    "block",
    "sector",
    "office",
    "room",
    "flat",
    "building",
    "bldg",
    "city",
    "district",
    "region",
    "india",
    "null",
}


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


def canonical_number(value):
    try:
        return str(int(value))
    except ValueError:
        return value


def extract_numbers(text):
    return {
        canonical_number(x)
        for x in NUMBER_RE.findall(text)
    }


def extract_tokens(text):
    result = set()

    for token in text.split():

        if token.isdigit():
            continue

        if len(token) < 3:
            continue

        if token in STOP_TOKENS:
            continue

        result.add(token)

    return result


def transliterate_name(text):
    if not text:
        return ""

    return normalize_name(
        unidecode(text)
    )


def lexical_similarity(a, b):
    if not a or not b:
        return 0.0

    return max(
        ratio(a, b),
        token_sort_ratio(a, b),
        token_set_ratio(a, b),
    ) / 100.0


# ============================================================
# LOAD QUERY SAMPLE
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

    print("Loading same 20k query sample...")

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

    df = df.sample(
        n=min(
            SAMPLE_SIZE,
            len(df),
        ),
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    print(
        f"Queries loaded: {len(df):,}"
    )

    print("Normalizing query names/addresses...")

    df["name_norm"] = (
        df["business_name"]
        .map(normalize_name)
    )

    df["name_translit"] = (
        df["business_name"]
        .map(transliterate_name)
    )

    df["address_norm"] = (
        df["business_address"]
        .map(normalize_address)
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
        len(v)
        for v in truth.values()
    )

    print(
        f"True S2 links: {total:,}"
    )

    return truth, total


# ============================================================
# QUERY STRUCTURED INDEX
# ============================================================

def build_query_indexes(query_df):

    print(
        "\nBuilding structured indexes..."
    )

    query_numbers = []
    query_tokens = []

    number_df = Counter()
    token_df = Counter()

    for address in query_df[
        "address_norm"
    ]:

        nums = extract_numbers(
            address
        )

        tokens = extract_tokens(
            address
        )

        query_numbers.append(nums)
        query_tokens.append(tokens)

        for n in nums:
            number_df[n] += 1

        for token in tokens:
            token_df[token] += 1

    useful_numbers = {
        n
        for n, df in number_df.items()
        if df <= MAX_NUMBER_DF
    }

    useful_tokens = {
        token
        for token, df in token_df.items()
        if df <= MAX_TOKEN_DF
    }

    number_index = defaultdict(list)
    token_index = defaultdict(list)

    for i in range(
        len(query_df)
    ):

        for n in query_numbers[i]:

            if n in useful_numbers:
                number_index[n].append(i)

        for token in query_tokens[i]:

            if token in useful_tokens:
                token_index[token].append(i)

    N = len(query_df)

    number_weight = {}

    for n in useful_numbers:

        number_weight[n] = (
            math.log(
                (N + 1)
                /
                (number_df[n] + 1)
            )
            + 1.0
        )

    token_weight = {}

    for token in useful_tokens:

        token_weight[token] = (
            math.log(
                (N + 1)
                /
                (token_df[token] + 1)
            )
            + 1.0
        )

    print(
        f"Useful numbers: "
        f"{len(number_index):,}"
    )

    print(
        f"Useful tokens: "
        f"{len(token_index):,}"
    )

    return (
        number_index,
        token_index,
        number_weight,
        token_weight,
    )


# ============================================================
# STRUCTURED GENERATION SCORE
# ============================================================

def get_structured_scores(
    address,
    number_index,
    token_index,
    number_weight,
    token_weight,
):

    numbers = extract_numbers(
        address
    )

    tokens = extract_tokens(
        address
    )

    scores = defaultdict(float)

    number_hits = defaultdict(int)
    token_hits = defaultdict(int)

    for n in numbers:

        if n not in number_index:
            continue

        w = number_weight[n]

        for query_idx in number_index[n]:

            scores[query_idx] += (
                2.5 * w
            )

            number_hits[
                query_idx
            ] += 1

    for token in tokens:

        if token not in token_index:
            continue

        w = token_weight[token]

        for query_idx in token_index[token]:

            scores[query_idx] += w

            token_hits[
                query_idx
            ] += 1

    for query_idx in list(scores):

        n_hits = number_hits.get(
            query_idx,
            0,
        )

        t_hits = token_hits.get(
            query_idx,
            0,
        )

        if n_hits >= 1 and t_hits >= 1:
            scores[query_idx] += 4.0

        if n_hits >= 2:
            scores[query_idx] += (
                2.0
                *
                (n_hits - 1)
            )

        if t_hits >= 2:
            scores[query_idx] += (
                1.0
                *
                (t_hits - 1)
            )

    return scores


# ============================================================
# BUILD TOP-100 STRUCTURED POOL
# ============================================================

def generate_structured_pool(
    query_df,
    number_index,
    token_index,
    number_weight,
    token_weight,
):

    print(
        f"\nGenerating structured "
        f"top-{INTERNAL_K} pool..."
    )

    heaps = [
        []
        for _ in range(
            len(query_df)
        )
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

    for chunk_no, chunk in enumerate(
        reader,
        start=1,
    ):

        total_read += len(chunk)

        chunk = chunk[
            chunk["country"] == COUNTRY
        ]

        india_read += len(chunk)

        for row in chunk.itertuples(
            index=False
        ):

            if not row.business_address:
                continue

            address_norm = (
                normalize_address(
                    row.business_address
                )
            )

            scores = (
                get_structured_scores(
                    address_norm,
                    number_index,
                    token_index,
                    number_weight,
                    token_weight,
                )
            )

            for query_idx, score in (
                scores.items()
            ):

                heap = heaps[
                    query_idx
                ]

                item = (
                    score,
                    row.entity_id,
                    row.business_name,
                    row.business_address,
                )

                if len(heap) < INTERNAL_K:

                    heapq.heappush(
                        heap,
                        item,
                    )

                elif score > heap[0][0]:

                    heapq.heapreplace(
                        heap,
                        item,
                    )

        print(
            f"\r"
            f"chunks={chunk_no:<3}"
            f" read={total_read:,}"
            f" India={india_read:,}"
            f" elapsed={now() - t0:.1f}s",
            end="",
            flush=True,
        )

        del chunk
        gc.collect()

    print()

    print(
        f"\nGeneration time: "
        f"{now() - t0:.2f}s"
    )

    return heaps


# ============================================================
# RERANK
# ============================================================

def rerank_structured_pool(
    query_df,
    heaps,
):

    print(
        "\nReranking structured pools..."
    )

    final_candidates = []

    for query_idx, heap in enumerate(
        heaps
    ):

        q = query_df.iloc[
            query_idx
        ]

        q_name = (
            q["name_norm"]
        )

        q_translit = (
            q["name_translit"]
        )

        if not heap:

            final_candidates.append(
                set()
            )

            continue

        # Structured scores need normalization
        # within this query's candidate pool.
        raw_structured = [
            x[0]
            for x in heap
        ]

        min_struct = min(
            raw_structured
        )

        max_struct = max(
            raw_structured
        )

        reranked = []

        for (
            struct_score,
            target_id,
            target_name,
            target_address,
        ) in heap:

            target_name_norm = (
                normalize_name(
                    target_name
                )
            )

            target_translit = (
                transliterate_name(
                    target_name
                )
            )

            normal_sim = (
                lexical_similarity(
                    q_name,
                    target_name_norm,
                )
            )

            translit_sim = (
                lexical_similarity(
                    q_translit,
                    target_translit,
                )
            )

            if max_struct > min_struct:

                structured_norm = (
                    struct_score
                    - min_struct
                ) / (
                    max_struct
                    - min_struct
                )

            else:

                structured_norm = 1.0

            # Initial simple weighting.
            final_score = (
                0.50
                * structured_norm
                +
                0.35
                * translit_sim
                +
                0.15
                * normal_sim
            )

            reranked.append(
                (
                    final_score,
                    target_id,
                )
            )

        reranked.sort(
            reverse=True
        )

        final_candidates.append(
            {
                target_id
                for _, target_id
                in reranked[
                    :FINAL_K
                ]
            }
        )

    return final_candidates


# ============================================================
# LOAD EXISTING CACHED BLOCKERS
# ============================================================

def load_base_candidates(
    query_df,
):

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
            "Base candidate cache mismatch."
        )

    name = []
    address = []

    for s1_id in query_ids:

        name.append(
            set(
                cache[
                    "candidates"
                ][s1_id]["name"]
            )
        )

        address.append(
            set(
                cache[
                    "candidates"
                ][s1_id]["address"]
            )
        )

    return name, address


def union_lists(
    *candidate_lists,
):

    result = []

    for i in range(
        len(candidate_lists[0])
    ):

        x = set()

        for lst in candidate_lists:
            x.update(
                lst[i]
            )

        result.append(x)

    return result


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    label,
    query_df,
    candidates,
    truth,
    total_true,
):

    recovered_links = 0

    full = 0
    zero = 0

    total_candidates = 0

    entities_with_truth = (
        len(truth)
    )

    for i, row in enumerate(
        query_df.itertuples(
            index=False
        )
    ):

        s1_id = row.entity_id

        cand = candidates[i]

        total_candidates += len(
            cand
        )

        true = truth.get(
            s1_id,
            set(),
        )

        if not true:
            continue

        hit = (
            true & cand
        )

        recovered_links += len(
            hit
        )

        if len(hit) == len(true):

            full += 1

        elif len(hit) == 0:

            zero += 1

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

    val_ids = (
        load_validation_ids()
    )

    query_df = (
        load_query_sample(
            val_ids
        )
    )

    truth, total_true = (
        load_truth(
            set(
                query_df[
                    "entity_id"
                ]
            )
        )
    )

    (
        number_index,
        token_index,
        number_weight,
        token_weight,
    ) = build_query_indexes(
        query_df
    )

    # --------------------------------------------
    # Generate broad structured pool
    # --------------------------------------------

    pools = (
        generate_structured_pool(
            query_df,
            number_index,
            token_index,
            number_weight,
            token_weight,
        )
    )

    # --------------------------------------------
    # Rerank down to 25
    # --------------------------------------------

    reranked = (
        rerank_structured_pool(
            query_df,
            pools,
        )
    )

    # Save the reranked channel.
    print(
        f"\nSaving reranked cache:\n"
        f"{RERANK_CACHE_PATH}"
    )

    with open(
        RERANK_CACHE_PATH,
        "wb",
    ) as f:

        pickle.dump(
            {
                "query_ids":
                    query_df[
                        "entity_id"
                    ].tolist(),

                "candidates":
                    reranked,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # --------------------------------------------
    # Existing char blockers
    # --------------------------------------------

    name, address = (
        load_base_candidates(
            query_df
        )
    )

    name_address = union_lists(
        name,
        address,
    )

    final_union = union_lists(
        name,
        address,
        reranked,
    )

    methods = [
        (
            "Name + Char Address",
            name_address,
        ),
        (
            "Reranked Structured K25",
            reranked,
        ),
        (
            "Name + Char Address + Reranked Structured",
            final_union,
        ),
    ]

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

    print("\n")
    print("=" * 110)
    print(
        "STRUCTURED TOP100 -> RERANK -> TOP25"
    )
    print("=" * 110)

    print(
        f"{'Method':<50}"
        f"{'Recall':>10}"
        f"{'Avg Cand':>12}"
        f"{'Full Entity':>15}"
        f"{'Zero Recovery':>16}"
    )

    print("-" * 110)

    for x in results:

        print(
            f"{x['label']:<50}"
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