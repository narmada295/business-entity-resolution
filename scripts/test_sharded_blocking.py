from pathlib import Path
import gc
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from src.indexed_blocking import (
    IndexedCharBlocker,
)

from src.sharded_blocking import (
    ShardedCharBlocker,
)


TRAIN_DIR = ROOT / "data" / "train"
INDEX_DIR = ROOT / "outputs" / "indexes"

S1_PATH = (
    TRAIN_DIR
    / "train_source1.tsv"
)

GT_PATH = (
    TRAIN_DIR
    / "train_ground_truth.tsv"
)


QUERY_LIMIT = 500


if len(sys.argv) != 3:

    print(
        "Usage:\n"
        "python scripts/test_sharded_blocking.py S2 US"
    )

    sys.exit(1)


SOURCE = sys.argv[1].upper()
COUNTRY = sys.argv[2]


if SOURCE not in {
    "S2",
    "S3",
}:

    raise ValueError(
        "SOURCE must be S2 or S3"
    )


ID_PREFIX = (
    f"{SOURCE}-"
)


def parse_matches(value):

    if not value:
        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


# ============================================================
# QUERIES
# ============================================================

print(
    "Loading queries..."
)


s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)


query_df = (
    s1[
        s1["country"]
        == COUNTRY
    ]
    .head(
        QUERY_LIMIT
    )
    .copy()
    .reset_index(
        drop=True
    )
)


query_ids = set(
    query_df[
        "entity_id"
    ]
)


print(
    "Queries:",
    len(query_df),
)


# ============================================================
# GROUND TRUTH
# ============================================================

truth = {
    qid: set()
    for qid in query_ids
}


gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)


for row in gt.itertuples(
    index=False
):

    qid = (
        row.source1_entity_id
    )


    if qid not in query_ids:
        continue


    truth[qid] = {
        value
        for value in parse_matches(
            row.matched_entity_ids
        )
        if value.startswith(
            ID_PREFIX
        )
    }


# ============================================================
# FULL INDEXED BASELINE
# ============================================================

print()
print("=" * 70)
print("FULL COUNTRY BLOCKER")
print("=" * 70)


start = time.perf_counter()


full_blocker = IndexedCharBlocker(
    INDEX_DIR,
    SOURCE,
    COUNTRY,
)


full_results = (
    full_blocker.generate(
        query_df
    )
)


full_time = (
    time.perf_counter()
    - start
)


del full_blocker

gc.collect()


# ============================================================
# SHARDED BLOCKER
# ============================================================

print()
print("=" * 70)
print("SHARDED BLOCKER")
print("=" * 70)


start = time.perf_counter()


sharded_blocker = (
    ShardedCharBlocker(
        INDEX_DIR,
        SOURCE,
        COUNTRY,
        max_route_keys=3,
        n_threads=1,
    )
)


sharded_results = (
    sharded_blocker.generate(
        query_df
    )
)


sharded_time = (
    time.perf_counter()
    - start
)


# ============================================================
# EVALUATE
# ============================================================

def evaluate(results):

    total_truth = 0
    recovered = 0

    full_entities = 0
    partial_entities = 0
    zero_entities = 0

    entities_with_truth = 0


    for idx, row in query_df.iterrows():

        qid = row[
            "entity_id"
        ]


        true_set = truth.get(
            qid,
            set(),
        )


        if not true_set:
            continue


        entities_with_truth += 1


        candidates = set(
            results[
                idx
            ].keys()
        )


        hits = (
            candidates
            &
            true_set
        )


        total_truth += len(
            true_set
        )

        recovered += len(
            hits
        )


        if len(hits) == len(
            true_set
        ):

            full_entities += 1

        elif len(hits) == 0:

            zero_entities += 1

        else:

            partial_entities += 1


    recall = (
        recovered
        / total_truth
        if total_truth
        else 0
    )


    average_candidates = (
        sum(
            len(x)
            for x in results
        )
        / len(results)
    )


    return {
        "truth":
            total_truth,

        "recovered":
            recovered,

        "recall":
            recall,

        "entities":
            entities_with_truth,

        "full":
            full_entities,

        "partial":
            partial_entities,

        "zero":
            zero_entities,

        "avg_candidates":
            average_candidates,
    }


full_stats = evaluate(
    full_results
)

shard_stats = evaluate(
    sharded_results
)


print()
print("=" * 70)
print("COMPARISON")
print("=" * 70)


print(
    f"{'':22}"
    f"{'FULL':>15}"
    f"{'SHARDED':>15}"
)


print(
    f"{'Runtime (s)':22}"
    f"{full_time:>15.2f}"
    f"{sharded_time:>15.2f}"
)


print(
    f"{'Queries/sec':22}"
    f"{len(query_df)/full_time:>15.2f}"
    f"{len(query_df)/sharded_time:>15.2f}"
)


print(
    f"{'Avg candidates':22}"
    f"{full_stats['avg_candidates']:>15.2f}"
    f"{shard_stats['avg_candidates']:>15.2f}"
)


print(
    f"{'Truth links':22}"
    f"{full_stats['truth']:>15}"
    f"{shard_stats['truth']:>15}"
)


print(
    f"{'Recovered':22}"
    f"{full_stats['recovered']:>15}"
    f"{shard_stats['recovered']:>15}"
)


print(
    f"{'Pair recall':22}"
    f"{full_stats['recall']:>14.4%}"
    f"{shard_stats['recall']:>14.4%}"
)


print(
    f"{'Full entities':22}"
    f"{full_stats['full']:>15}"
    f"{shard_stats['full']:>15}"
)


print(
    f"{'Partial entities':22}"
    f"{full_stats['partial']:>15}"
    f"{shard_stats['partial']:>15}"
)


print(
    f"{'Zero entities':22}"
    f"{full_stats['zero']:>15}"
    f"{shard_stats['zero']:>15}"
)