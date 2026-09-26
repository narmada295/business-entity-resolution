from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from src.streaming_blocking import (
    StreamingCountryBlocker,
)


TRAIN_DIR = (
    ROOT
    / "data"
    / "train"
)

S1_PATH = (
    TRAIN_DIR
    / "train_source1.tsv"
)

S2_PATH = (
    TRAIN_DIR
    / "train_source2.tsv"
)

GT_PATH = (
    TRAIN_DIR
    / "train_ground_truth.tsv"
)


QUERY_LIMIT = 500

S1_READ_ROWS = 100_000

TARGET_CHUNK_SIZE = 50_000


def parse_matches(value):

    if not value:
        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


# ============================================================
# LOAD S1 SAMPLE
# ============================================================

print(
    "Loading S1 sample..."
)


s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    nrows=S1_READ_ROWS,
)


# Choose most common country dynamically.
country = (
    s1[
        "country"
    ]
    .value_counts()
    .index[
        0
    ]
)


query_df = (
    s1[
        s1[
            "country"
        ]
        ==
        country
    ]
    .head(
        QUERY_LIMIT
    )
    .copy()
    .reset_index(
        drop=True
    )
)


print(
    "Country:",
    country,
)

print(
    "Queries:",
    len(
        query_df
    ),
)


query_ids = set(
    query_df[
        "entity_id"
    ]
)


# ============================================================
# INITIALIZE BLOCKER
# ============================================================

blocker = (
    StreamingCountryBlocker(
        query_df
    )
)


# ============================================================
# STREAM ENTIRE S2 FILE
# ============================================================

print(
    "\nStreaming S2..."
)


target_ids_seen = set()

total_read = 0
country_rows = 0


reader = pd.read_csv(
    S2_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    chunksize=TARGET_CHUNK_SIZE,
)


for chunk_no, chunk in enumerate(
    reader,
    start=1,
):

    total_read += len(
        chunk
    )


    country_chunk = (
        chunk[
            chunk[
                "country"
            ]
            ==
            country
        ]
        .copy()
    )


    if len(
        country_chunk
    ) == 0:

        continue


    country_rows += len(
        country_chunk
    )


    target_ids_seen.update(
        country_chunk[
            "entity_id"
        ]
    )


    blocker.process_target_chunk(
        country_chunk
    )


    print(
        f"\r"
        f"chunks={chunk_no:<3}"
        f" total_read={total_read:,}"
        f" country_rows={country_rows:,}",
        end="",
        flush=True,
    )


print()


# ============================================================
# FINALIZE
# ============================================================

results = blocker.finalize()


candidate_counts = [
    len(x)
    for x in results
]


print()

print(
    "Average candidates:",
    sum(
        candidate_counts
    )
    /
    len(
        candidate_counts
    ),
)

print(
    "Min candidates:",
    min(
        candidate_counts
    ),
)

print(
    "Max candidates:",
    max(
        candidate_counts
    ),
)

print(
    "Zero candidate queries:",
    sum(
        x == 0
        for x in candidate_counts
    ),
)


# ============================================================
# LOAD GT
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

    s1_id = (
        row.source1_entity_id
    )

    if s1_id not in query_ids:
        continue


    matches = {
        x
        for x in parse_matches(
            row.matched_entity_ids
        )
        if (
            x.startswith("S2-")
            and
            x in target_ids_seen
        )
    }


    truth[
        s1_id
    ] = matches


# ============================================================
# RECALL
# ============================================================

total_truth = 0
recovered = 0

full = 0
partial = 0
zero = 0


for idx, row in query_df.iterrows():

    qid = (
        row[
            "entity_id"
        ]
    )

    true_set = (
        truth.get(
            qid,
            set(),
        )
    )

    if not true_set:
        continue


    cand_set = set(
        results[
            idx
        ]
    )


    hit_set = (
        true_set
        &
        cand_set
    )


    total_truth += len(
        true_set
    )

    recovered += len(
        hit_set
    )


    if len(
        hit_set
    ) == len(
        true_set
    ):

        full += 1

    elif len(
        hit_set
    ) == 0:

        zero += 1

    else:

        partial += 1


print()

print(
    "=" * 70
)

print(
    "FULL-S2 STREAMING BLOCKER RECALL"
)

print(
    "=" * 70
)

print(
    "Truth links:",
    total_truth,
)

print(
    "Recovered:",
    recovered,
)

print(
    "Pair recall:",
    (
        f"{recovered/total_truth:.4%}"
        if total_truth
        else "N/A"
    ),
)

print(
    "Full entities:",
    full,
)

print(
    "Partial entities:",
    partial,
)

print(
    "Zero entities:",
    zero,
)


print()

print(
    "Streaming blocking test complete."
)