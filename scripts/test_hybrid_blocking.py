from pathlib import Path
import gc
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from src.sharded_blocking import (
    ShardedCharBlocker,
)

from src.hybrid_blocking import (
    HybridBlocker,
)


TRAIN_DIR = ROOT / "data" / "train"
INDEX_DIR = ROOT / "outputs" / "indexes"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"


QUERY_LIMIT = 500


if len(sys.argv) != 3:

    print(
        "Usage:\n"
        "python scripts/test_hybrid_blocking.py S2 India"
    )

    sys.exit(1)


SOURCE = sys.argv[1].upper()
COUNTRY = sys.argv[2]


if SOURCE not in {"S2", "S3"}:

    raise ValueError(
        "SOURCE must be S2 or S3"
    )


ID_PREFIX = f"{SOURCE}-"


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
    query_df["entity_id"]
)


print(
    "Queries:",
    len(query_df),
)


# ============================================================
# GT
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
# EVALUATION
# ============================================================

def evaluate(results):

    total_truth = 0
    recovered = 0

    entities = 0

    full = 0
    partial = 0
    zero = 0


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


        entities += 1


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

            full += 1

        elif len(hits) == 0:

            zero += 1

        else:

            partial += 1


    return {
        "truth":
            total_truth,

        "recovered":
            recovered,

        "recall":
            (
                recovered / total_truth
                if total_truth
                else 0
            ),

        "entities":
            entities,

        "full":
            full,

        "partial":
            partial,

        "zero":
            zero,

        "avg_candidates":
            sum(
                len(x)
                for x in results
            )
            / len(results),
    }


# ============================================================
# LEXICAL
# ============================================================

print()
print("=" * 70)
print(
    "SHARDED LEXICAL"
)
print("=" * 70)


start = time.perf_counter()


lexical = ShardedCharBlocker(
    INDEX_DIR,
    SOURCE,
    COUNTRY,
    n_threads=1,
)


lexical_results = lexical.generate(
    query_df
)


lexical_time = (
    time.perf_counter()
    - start
)


lexical_stats = evaluate(
    lexical_results
)


del lexical
del lexical_results

gc.collect()


# ============================================================
# HYBRID
# ============================================================

print()
print("=" * 70)
print(
    "HYBRID LEXICAL + STRUCTURED"
)
print("=" * 70)


start = time.perf_counter()


hybrid = HybridBlocker(
    INDEX_DIR,
    SOURCE,
    COUNTRY,
)


hybrid_results = hybrid.generate(
    query_df
)


hybrid_time = (
    time.perf_counter()
    - start
)


hybrid_stats = evaluate(
    hybrid_results
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 70)
print(
    "COMPARISON"
)
print("=" * 70)


print(
    f"{'':25}"
    f"{'LEXICAL':>15}"
    f"{'HYBRID':>15}"
)


print(
    f"{'Runtime (s)':25}"
    f"{lexical_time:>15.2f}"
    f"{hybrid_time:>15.2f}"
)


print(
    f"{'Queries/sec':25}"
    f"{len(query_df)/lexical_time:>15.2f}"
    f"{len(query_df)/hybrid_time:>15.2f}"
)


print(
    f"{'Avg candidates':25}"
    f"{lexical_stats['avg_candidates']:>15.2f}"
    f"{hybrid_stats['avg_candidates']:>15.2f}"
)


print(
    f"{'Truth links':25}"
    f"{lexical_stats['truth']:>15}"
    f"{hybrid_stats['truth']:>15}"
)


print(
    f"{'Recovered':25}"
    f"{lexical_stats['recovered']:>15}"
    f"{hybrid_stats['recovered']:>15}"
)


print(
    f"{'Pair recall':25}"
    f"{lexical_stats['recall']:>14.4%}"
    f"{hybrid_stats['recall']:>14.4%}"
)


print(
    f"{'Full entities':25}"
    f"{lexical_stats['full']:>15}"
    f"{hybrid_stats['full']:>15}"
)


print(
    f"{'Partial entities':25}"
    f"{lexical_stats['partial']:>15}"
    f"{hybrid_stats['partial']:>15}"
)


print(
    f"{'Zero entities':25}"
    f"{lexical_stats['zero']:>15}"
    f"{hybrid_stats['zero']:>15}"
)