from pathlib import Path
import sys
import pickle

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.streaming_blocking import StreamingCountryBlocker


# ============================================================
# PATHS
# ============================================================

TRAIN_DIR = ROOT / "data" / "train"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"


# ============================================================
# SETTINGS
# ============================================================

QUERY_LIMIT = 500
S1_READ_ROWS = 100_000
TARGET_CHUNK_SIZE = 50_000


# ============================================================
# SOURCE ARGUMENT
# ============================================================

if len(sys.argv) != 2:
    print(
        "Usage:\n"
        "  python scripts/test_streaming_blocking.py S2\n"
        "  python scripts/test_streaming_blocking.py S3"
    )
    sys.exit(1)


SOURCE = sys.argv[1].upper()

if SOURCE not in {"S2", "S3"}:
    raise ValueError("SOURCE must be either S2 or S3")


if SOURCE == "S2":
    TARGET_PATH = TRAIN_DIR / "train_source2.tsv"
    ID_PREFIX = "S2-"

else:
    TARGET_PATH = TRAIN_DIR / "train_source3.tsv"
    ID_PREFIX = "S3-"


# ============================================================
# HELPERS
# ============================================================

def parse_matches(value):
    if not value:
        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print(f"STREAMING BLOCKER TEST: {SOURCE}")
print("=" * 70)


# ============================================================
# LOAD S1 SAMPLE
# ============================================================

print("\nLoading S1 sample...")


s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    nrows=S1_READ_ROWS,
)


# Choose country dynamically.
country = (
    s1["country"]
    .value_counts()
    .index[0]
)


query_df = (
    s1[
        s1["country"]
        == country
    ]
    .head(QUERY_LIMIT)
    .copy()
    .reset_index(drop=True)
)


print("Country:", country)
print("Queries:", len(query_df))


query_ids = set(
    query_df["entity_id"]
)


# ============================================================
# INITIALIZE BLOCKER
# ============================================================

blocker = StreamingCountryBlocker(
    query_df
)


# ============================================================
# STREAM TARGET SOURCE
# ============================================================

print(f"\nStreaming {SOURCE}...")


target_ids_seen = set()

total_read = 0
country_rows = 0


reader = pd.read_csv(
    TARGET_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    chunksize=TARGET_CHUNK_SIZE,
)


for chunk_no, chunk in enumerate(
    reader,
    start=1,
):

    total_read += len(chunk)


    country_chunk = (
        chunk[
            chunk["country"]
            == country
        ]
        .copy()
    )


    if len(country_chunk) == 0:
        continue


    country_rows += len(
        country_chunk
    )


    target_ids_seen.update(
        country_chunk["entity_id"]
    )


    blocker.process_target_chunk(
        country_chunk
    )


    print(
        f"\r"
        f"chunks={chunk_no:<3} "
        f"total_read={total_read:,} "
        f"country_rows={country_rows:,}",
        end="",
        flush=True,
    )


print()


# ============================================================
# FINALIZE
# ============================================================

results = blocker.finalize()


# ============================================================
# SAVE CACHE IMMEDIATELY
# ============================================================

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


cache_path = (
    OUTPUT_DIR
    / f"streaming_candidates_{SOURCE.lower()}.pkl"
)


with open(
    cache_path,
    "wb",
) as f:

    pickle.dump(
        {
            "source": SOURCE,
            "country": country,
            "query_df": query_df,
            "results": results,
            "target_ids_seen": target_ids_seen,
        },
        f,
        protocol=pickle.HIGHEST_PROTOCOL,
    )


print(
    f"\nSaved candidate cache to:"
    f"\n{cache_path}"
)


# ============================================================
# CANDIDATE STATISTICS
# ============================================================

candidate_counts = [
    len(x)
    for x in results
]


print()

print(
    "Average candidates:",
    sum(candidate_counts)
    / len(candidate_counts),
)

print(
    "Min candidates:",
    min(candidate_counts),
)

print(
    "Max candidates:",
    max(candidate_counts),
)

print(
    "Zero candidate queries:",
    sum(
        x == 0
        for x in candidate_counts
    ),
)


# ============================================================
# CHECK GT EXISTS
# ============================================================

if not GT_PATH.exists():

    print()
    print(
        "WARNING: ground-truth file not found:"
    )
    print(
        GT_PATH
    )

    print()
    print(
        "Blocking completed successfully and "
        "candidate cache has been saved."
    )

    sys.exit(0)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")


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
            x.startswith(ID_PREFIX)
            and
            x in target_ids_seen
        )
    }


    truth[
        s1_id
    ] = matches


# ============================================================
# RECALL + BLOCKER CONTRIBUTION
# ============================================================

total_truth = 0
recovered = 0

entities_with_truth = 0

full = 0
partial = 0
zero = 0


name_hits = 0
address_hits = 0
structured_hits = 0


name_only_hits = 0
address_only_hits = 0
structured_only_hits = 0


for idx, row in query_df.iterrows():

    qid = (
        row["entity_id"]
    )


    true_set = truth.get(
        qid,
        set(),
    )


    if not true_set:
        continue


    entities_with_truth += 1


    candidates = results[
        idx
    ]


    candidate_ids = set(
        candidates.keys()
    )


    hit_set = (
        true_set
        &
        candidate_ids
    )


    total_truth += len(
        true_set
    )

    recovered += len(
        hit_set
    )


    if len(hit_set) == len(
        true_set
    ):

        full += 1


    elif len(hit_set) == 0:

        zero += 1


    else:

        partial += 1


    # --------------------------------------------------------
    # TRUE-LINK BLOCKER CONTRIBUTION
    # --------------------------------------------------------

    for true_id in true_set:

        provenance = candidates.get(
            true_id
        )


        if provenance is None:
            continue


        from_name = int(
            provenance.get(
                "from_name_blocker",
                0,
            )
        )


        from_address = int(
            provenance.get(
                "from_address_blocker",
                0,
            )
        )


        from_structured = int(
            provenance.get(
                "from_structured_blocker",
                0,
            )
        )


        name_hits += from_name
        address_hits += from_address
        structured_hits += from_structured


        channel_count = (
            from_name
            +
            from_address
            +
            from_structured
        )


        if channel_count == 1:

            if from_name:
                name_only_hits += 1

            elif from_address:
                address_only_hits += 1

            elif from_structured:
                structured_only_hits += 1


# ============================================================
# REPORT
# ============================================================

print()

print("=" * 70)

print(
    f"FULL-{SOURCE} STREAMING BLOCKER RECALL"
)

print("=" * 70)


print(
    "Target country rows:",
    f"{country_rows:,}",
)

print(
    "Truth links:",
    total_truth,
)

print(
    "Recovered:",
    recovered,
)


if total_truth > 0:

    print(
        "Pair recall:",
        f"{recovered / total_truth:.4%}",
    )

else:

    print(
        "Pair recall: N/A"
    )


print(
    "Entities with truth:",
    entities_with_truth,
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
    "True-link blocker contribution:"
)

print(
    "Name hits:",
    name_hits,
)

print(
    "Address hits:",
    address_hits,
)

print(
    "Structured hits:",
    structured_hits,
)


print()

print(
    "Unique true-link contribution:"
)

print(
    "Name only:",
    name_only_hits,
)

print(
    "Address only:",
    address_only_hits,
)

print(
    "Structured only:",
    structured_only_hits,
)


print()

print(
    f"{SOURCE} streaming blocking test complete."
)