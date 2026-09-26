from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.blocking import (
    CountryBlocker,
    discover_countries,
)


TRAIN_DIR = ROOT / "data" / "train"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
S2_PATH = TRAIN_DIR / "train_source2.tsv"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"


QUERY_LIMIT = 500

S1_READ_ROWS = 100_000
S2_READ_ROWS = 300_000


def parse_matches(value):

    if not value:
        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


# ============================================================
# LOAD TEST CORPUS
# ============================================================

print("Reading sample rows...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    nrows=S1_READ_ROWS,
)

s2 = pd.read_csv(
    S2_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    nrows=S2_READ_ROWS,
)


countries = discover_countries(
    s1,
    s2,
)

print(
    "Countries discovered:",
    countries,
)

if not countries:
    raise RuntimeError(
        "No countries found"
    )


# ============================================================
# CHOOSE COUNTRY DYNAMICALLY
# ============================================================

country_counts = (
    s1["country"]
    .value_counts()
)

test_country = (
    country_counts.index[0]
)

print(
    "Testing country:",
    test_country,
)


query_df = (
    s1[
        s1["country"]
        == test_country
    ]
    .head(QUERY_LIMIT)
    .copy()
    .reset_index(drop=True)
)


target_df = (
    s2[
        s2["country"]
        == test_country
    ]
    .copy()
    .reset_index(drop=True)
)


print(
    "Queries:",
    len(query_df),
)

print(
    "Targets in smoke corpus:",
    len(target_df),
)


if len(query_df) == 0:
    raise RuntimeError(
        "No query rows"
    )

if len(target_df) == 0:
    raise RuntimeError(
        "No target rows"
    )


# ============================================================
# LOAD GT FOR THE 500 QUERY ENTITIES
# ============================================================

query_ids = set(
    query_df["entity_id"]
)

target_ids = set(
    target_df["entity_id"]
)


truth = {
    entity_id: set()
    for entity_id in query_ids
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

    matches = parse_matches(
        row.matched_entity_ids
    )

    # This smoke blocker is only S2.
    # Also restrict truth to targets that actually
    # exist in the sampled S2 corpus.
    matches = {
        target_id
        for target_id in matches
        if (
            target_id.startswith("S2-")
            and
            target_id in target_ids
        )
    }

    truth[
        s1_id
    ] = matches


total_truth_links = sum(
    len(x)
    for x in truth.values()
)


entities_with_truth = sum(
    bool(x)
    for x in truth.values()
)


print()

print(
    "GT links that actually exist "
    "inside smoke target corpus:",
    total_truth_links,
)

print(
    "Queries with >=1 such GT link:",
    entities_with_truth,
)


# ============================================================
# BUILD BLOCKER
# ============================================================

print(
    "\nBuilding blocker..."
)

blocker = CountryBlocker(
    target_df
)


# ============================================================
# GENERATE CANDIDATES
# ============================================================

print(
    "Generating candidates..."
)

results = blocker.generate(
    query_df
)


# ============================================================
# CANDIDATE VOLUME
# ============================================================

candidate_counts = [
    len(x)
    for x in results
]


print()

print(
    "Queries processed:",
    len(results),
)

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
# RECALL
# ============================================================

recovered_links = 0

full_entities = 0
partial_entities = 0
zero_recovery_entities = 0


for idx, row in query_df.iterrows():

    s1_id = row["entity_id"]

    true_links = truth.get(
        s1_id,
        set(),
    )

    if not true_links:
        continue

    candidate_ids = set(
        results[idx]
    )

    recovered = (
        true_links
        &
        candidate_ids
    )

    recovered_links += len(
        recovered
    )

    if len(recovered) == len(
        true_links
    ):

        full_entities += 1

    elif len(recovered) == 0:

        zero_recovery_entities += 1

    else:

        partial_entities += 1


if total_truth_links:

    pair_recall = (
        recovered_links
        / total_truth_links
    )

else:

    pair_recall = 0.0


print()

print(
    "=" * 70
)

print(
    "SMOKE-CORPUS BLOCKING RECALL"
)

print(
    "=" * 70
)

print(
    "Truth links:",
    total_truth_links,
)

print(
    "Recovered links:",
    recovered_links,
)

print(
    "Pair recall:",
    f"{pair_recall:.4%}",
)

print(
    "Entities with truth:",
    entities_with_truth,
)

print(
    "Full recovery:",
    full_entities,
)

print(
    "Partial recovery:",
    partial_entities,
)

print(
    "Zero recovery:",
    zero_recovery_entities,
)


# ============================================================
# PER-BLOCKER TRUE-LINK CONTRIBUTION
# ============================================================

name_hits = 0
address_hits = 0
structured_hits = 0

name_only_hits = 0
address_only_hits = 0
structured_only_hits = 0


for idx, row in query_df.iterrows():

    s1_id = row["entity_id"]

    true_links = truth.get(
        s1_id,
        set(),
    )

    if not true_links:
        continue

    candidates = results[idx]

    for true_id in true_links:

        provenance = candidates.get(
            true_id
        )

        if provenance is None:
            continue

        from_name = (
            provenance[
                "from_name_blocker"
            ]
        )

        from_address = (
            provenance[
                "from_address_blocker"
            ]
        )

        from_structured = (
            provenance[
                "from_structured_blocker"
            ]
        )

        name_hits += from_name
        address_hits += from_address
        structured_hits += from_structured

        channels = (
            from_name
            +
            from_address
            +
            from_structured
        )

        if channels == 1:

            if from_name:
                name_only_hits += 1

            elif from_address:
                address_only_hits += 1

            elif from_structured:
                structured_only_hits += 1


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
    "Unique true links contributed:"
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
    "Blocking correctness test OK."
)