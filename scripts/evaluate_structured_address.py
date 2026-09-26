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


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.normalize import normalize_address


TRAIN_DIR = ROOT / "data" / "train"
OUTPUT_DIR = ROOT / "outputs"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
S2_PATH = TRAIN_DIR / "train_source2.tsv"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"
SPLIT_PATH = OUTPUT_DIR / "splits.tsv"

CACHE_PATH = (
    OUTPUT_DIR /
    "s2_india_sample_candidates.pkl"
)


COUNTRY = "India"

SAMPLE_SIZE = 20_000
RANDOM_SEED = 42

STRUCTURED_K = 20

READ_CHUNK_SIZE = 100_000

# Ignore extremely common query features.
#
# Example:
# "delhi", "road", "1", etc. can occur in huge numbers
# of records and aren't useful blocking keys.
MAX_TOKEN_DF = 500
MAX_NUMBER_DF = 1000


NUMBER_RE = re.compile(r"\d+")


# Generic address vocabulary that carries little identity information.
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


# ============================================================
# ADDRESS FEATURES
# ============================================================

def canonical_number(value):
    """
    Make things such as:

        0017 -> 17
        0070 -> 70

    comparable.

    Keep "0" as "0".
    """

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
    """
    Extract non-numeric address tokens.

    Very short/common tokens are removed here because
    they are poor blocking keys.
    """

    output = set()

    for token in text.split():

        if token.isdigit():
            continue

        if len(token) < 3:
            continue

        if token in STOP_TOKENS:
            continue

        output.add(token)

    return output


# ============================================================
# LOAD SAME 20K SAMPLE
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
                        "business_address",
                    ]
                ]
            )

    df = pd.concat(
        pieces,
        ignore_index=True,
    )

    df = df.sample(
        n=min(SAMPLE_SIZE, len(df)),
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    df["address_norm"] = (
        df["business_address"]
        .map(normalize_address)
    )

    print(
        f"Queries loaded: {len(df):,}"
    )

    return df


# ============================================================
# GROUND TRUTH
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
# QUERY FEATURE INDEXES
# ============================================================

def build_query_indexes(query_df):

    print("\nBuilding structured-address indexes...")

    query_numbers = []
    query_tokens = []

    number_df = Counter()
    token_df = Counter()

    # First collect document frequency.
    for address in query_df["address_norm"]:

        numbers = extract_numbers(address)
        tokens = extract_tokens(address)

        query_numbers.append(numbers)
        query_tokens.append(tokens)

        for n in numbers:
            number_df[n] += 1

        for token in tokens:
            token_df[token] += 1

    # Only retain features that aren't extremely common.
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

    for query_idx in range(len(query_df)):

        for number in query_numbers[query_idx]:

            if number in useful_numbers:

                number_index[number].append(
                    query_idx
                )

        for token in query_tokens[query_idx]:

            if token in useful_tokens:

                token_index[token].append(
                    query_idx
                )

    N = len(query_df)

    number_weight = {}

    for number in useful_numbers:

        number_weight[number] = (
            math.log(
                (N + 1)
                /
                (number_df[number] + 1)
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
        f"Useful number keys: "
        f"{len(number_index):,}"
    )

    print(
        f"Useful token keys: "
        f"{len(token_index):,}"
    )

    return (
        number_index,
        token_index,
        number_weight,
        token_weight,
    )


# ============================================================
# STRUCTURED SCORE
# ============================================================

def candidate_query_scores(
    address,
    number_index,
    token_index,
    number_weight,
    token_weight,
):
    """
    For ONE S2 address, find potentially relevant S1 queries
    using exact address components.

    Number matches receive more weight because house/plot/
    sector numbers tend to be highly discriminative.
    """

    numbers = extract_numbers(address)
    tokens = extract_tokens(address)

    scores = defaultdict(float)

    number_hits = defaultdict(int)
    token_hits = defaultdict(int)

    # --------------------------
    # Numeric components
    # --------------------------

    for number in numbers:

        if number not in number_index:
            continue

        weight = number_weight[number]

        for query_idx in number_index[number]:

            # Numbers get stronger weight.
            scores[query_idx] += (
                2.5 * weight
            )

            number_hits[query_idx] += 1

    # --------------------------
    # Rare textual components
    # --------------------------

    for token in tokens:

        if token not in token_index:
            continue

        weight = token_weight[token]

        for query_idx in token_index[token]:

            scores[query_idx] += weight
            token_hits[query_idx] += 1

    # --------------------------
    # Structural bonuses
    # --------------------------

    for query_idx in list(scores):

        n_hits = number_hits.get(
            query_idx,
            0,
        )

        t_hits = token_hits.get(
            query_idx,
            0,
        )

        # Number + token agreement is especially useful.
        if n_hits >= 1 and t_hits >= 1:

            scores[query_idx] += 4.0

        # Multiple agreeing numbers is very strong.
        if n_hits >= 2:

            scores[query_idx] += (
                2.0 * (n_hits - 1)
            )

        # Several rare address tokens agreeing.
        if t_hits >= 2:

            scores[query_idx] += (
                1.0 * (t_hits - 1)
            )

    return scores


# ============================================================
# STREAM S2
# ============================================================

def search_structured_addresses(
    query_count,
    number_index,
    token_index,
    number_weight,
    token_weight,
):

    print(
        "\nStreaming S2 India..."
    )

    heaps = [
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
        ]

        india_read += len(chunk)

        for row in chunk.itertuples(
            index=False
        ):

            if not row.business_address:
                continue

            address = normalize_address(
                row.business_address
            )

            scores = candidate_query_scores(
                address,
                number_index,
                token_index,
                number_weight,
                token_weight,
            )

            target_id = row.entity_id

            for query_idx, score in (
                scores.items()
            ):

                heap = heaps[query_idx]

                item = (
                    score,
                    target_id,
                )

                if len(heap) < STRUCTURED_K:

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
            f"chunks={chunk_number:<3}"
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
        f"\nStructured search time: "
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
# LOAD EXISTING CHAR CANDIDATES
# ============================================================

def load_cached_candidates(
    query_df,
):

    print(
        "\nLoading cached name/address candidates..."
    )

    with open(
        CACHE_PATH,
        "rb",
    ) as f:

        cache = pickle.load(f)

    cached_ids = cache["query_ids"]

    expected_ids = (
        query_df["entity_id"]
        .tolist()
    )

    if cached_ids != expected_ids:

        raise ValueError(
            "Cached query IDs don't match "
            "the current deterministic sample."
        )

    candidate_maps = cache[
        "candidates"
    ]

    name_sets = []
    address_sets = []

    for s1_id in expected_ids:

        name_sets.append(
            set(
                candidate_maps[
                    s1_id
                ]["name"]
            )
        )

        address_sets.append(
            set(
                candidate_maps[
                    s1_id
                ]["address"]
            )
        )

    return (
        name_sets,
        address_sets,
    )


# ============================================================
# UNION
# ============================================================

def union_lists(*candidate_lists):

    result = []

    for i in range(
        len(candidate_lists[0])
    ):

        combined = set()

        for candidates in candidate_lists:

            combined.update(
                candidates[i]
            )

        result.append(combined)

    return result


# ============================================================
# EVALUATION
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
        query_df.itertuples(index=False)
    ):

        s1_id = row.entity_id

        candidates = candidate_sets[i]

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

    start = now()

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

    (
        number_index,
        token_index,
        number_weight,
        token_weight,
    ) = build_query_indexes(
        query_df
    )

    structured = (
        search_structured_addresses(
            len(query_df),
            number_index,
            token_index,
            number_weight,
            token_weight,
        )
    )

    STRUCTURED_CACHE_PATH = (
        OUTPUT_DIR
        / "s2_india_structured_candidates.pkl"
    )

    print(
        f"\nSaving structured-address cache:"
        f"\n{STRUCTURED_CACHE_PATH}"
    )

    with open(
        STRUCTURED_CACHE_PATH,
        "wb",
    ) as f:
        pickle.dump(
            {
                "query_ids":
                    query_df[
                        "entity_id"
                    ].tolist(),

                "candidates":
                    structured,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    (
        name,
        address,
    ) = load_cached_candidates(
        query_df
    )

    # Existing best blocker.
    name_address = union_lists(
        name,
        address,
    )

    # New variants.
    address_structured = union_lists(
        address,
        structured,
    )

    all_three = union_lists(
        name,
        address,
        structured,
    )

    methods = [
        (
            "Name K20",
            name,
        ),
        (
            "Char Address K20",
            address,
        ),
        (
            "Structured Address K20",
            structured,
        ),
        (
            "Name + Char Address",
            name_address,
        ),
        (
            "Char Address + Structured",
            address_structured,
        ),
        (
            "Name + Char Address + Structured",
            all_three,
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
    print("=" * 100)
    print("STRUCTURED ADDRESS BLOCKING RESULTS")
    print("=" * 100)

    print(
        f"{'Method':<40}"
        f"{'Recall':>10}"
        f"{'Avg Cand':>12}"
        f"{'Full Entity':>15}"
        f"{'Zero Recovery':>16}"
    )

    print("-" * 100)

    for x in results:

        print(
            f"{x['label']:<40}"
            f"{x['recall']:>9.2%}"
            f"{x['avg_candidates']:>12.2f}"
            f"{x['full']:>14.2%}"
            f"{x['zero']:>15.2%}"
        )

    # How many extra true links did the structured blocker add?
    old_recovered = results[3][
        "recovered"
    ]

    new_recovered = results[-1][
        "recovered"
    ]

    print(
        "\nAdditional true links from "
        "structured blocker:"
    )

    print(
        f"{new_recovered - old_recovered:,}"
    )

    print(
        "\nTotal runtime:",
        f"{now() - start:.2f}s",
    )


if __name__ == "__main__":
    main()