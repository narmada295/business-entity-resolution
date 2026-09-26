from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TEST_DIR = (
    ROOT
    / "data"
    / "test"
)

INFERENCE_ROOT = (
    ROOT
    / "outputs"
    / "test_inference"
)

FINAL_DIR = (
    ROOT
    / "outputs"
    / "submission"
)

FINAL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


S1_PATH = (
    TEST_DIR
    / "test_source1.tsv"
)


def parse_ids(value):

    if not value:
        return []

    return [
        x
        for x in str(value).split(",")
        if x
    ]


def unique_preserving_order(
    values,
):

    seen = set()
    output = []

    for value in values:

        if value not in seen:

            seen.add(
                value
            )

            output.append(
                value
            )

    return output


print(
    "Loading test S1..."
)


s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    usecols=[
        "entity_id",
        "country",
    ],
)


candidate_map = {
    entity_id: []
    for entity_id in s1[
        "entity_id"
    ]
}


match_map = {
    entity_id: []
    for entity_id in s1[
        "entity_id"
    ]
}


# ============================================================
# READ ALL PARTITIONS
# ============================================================

for source in [
    "S2",
    "S3",
]:

    source_dir = (
        INFERENCE_ROOT
        / source
    )


    if not source_dir.exists():

        raise RuntimeError(
            f"Missing inference directory: "
            f"{source_dir}"
        )


    country_dirs = [
        p
        for p in source_dir.iterdir()
        if p.is_dir()
    ]


    for country_dir in (
        country_dirs
    ):

        print(
            "Reading:",
            source,
            country_dir.name,
        )


        candidate_paths = sorted(
            country_dir.glob(
                "candidates_*.parquet"
            )
        )


        match_paths = sorted(
            country_dir.glob(
                "matches_*.parquet"
            )
        )


        if not candidate_paths:

            raise RuntimeError(
                f"No candidate partitions "
                f"in {country_dir}"
            )


        if not match_paths:

            raise RuntimeError(
                f"No match partitions "
                f"in {country_dir}"
            )


        for path in candidate_paths:

            df = pd.read_parquet(
                path
            )


            for row in df.itertuples(
                index=False
            ):

                candidate_map[
                    row.source1_entity_id
                ].extend(
                    parse_ids(
                        row.candidate_entity_ids
                    )
                )


        for path in match_paths:

            df = pd.read_parquet(
                path
            )


            for row in df.itertuples(
                index=False
            ):

                match_map[
                    row.source1_entity_id
                ].extend(
                    parse_ids(
                        row.matched_entity_ids
                    )
                )


# ============================================================
# DEDUP + VERIFY SUBSET
# ============================================================

candidate_strings = []
match_strings = []


for entity_id in s1[
    "entity_id"
]:

    candidates = (
        unique_preserving_order(
            candidate_map[
                entity_id
            ]
        )
    )


    matches = (
        unique_preserving_order(
            match_map[
                entity_id
            ]
        )
    )


    candidate_set = set(
        candidates
    )


    missing = [
        match_id
        for match_id in matches
        if match_id not in candidate_set
    ]


    if missing:

        raise RuntimeError(
            f"{entity_id}: predicted matches "
            f"missing from candidate set: "
            f"{missing[:5]}"
        )


    candidate_strings.append(
        ",".join(
            candidates
        )
    )


    match_strings.append(
        ",".join(
            matches
        )
    )


# ============================================================
# WRITE EXACT CHALLENGE FORMAT
# ============================================================

candidate_output = pd.DataFrame(
    {
        "source1_entity_id":
            s1[
                "entity_id"
            ],

        "candidate_entity_ids":
            candidate_strings,
    }
)


matching_output = pd.DataFrame(
    {
        "source1_entity_id":
            s1[
                "entity_id"
            ],

        "matched_entity_ids":
            match_strings,
    }
)


candidate_path = (
    FINAL_DIR
    / "candidate_pairs.tsv"
)


matching_path = (
    FINAL_DIR
    / "matching_results.tsv"
)


candidate_output.to_csv(
    candidate_path,
    sep="\t",
    index=False,
)


matching_output.to_csv(
    matching_path,
    sep="\t",
    index=False,
)


print()
print(
    "Rows:",
    f"{len(s1):,}",
)

print(
    "Saved:",
    candidate_path,
)

print(
    "Saved:",
    matching_path,
)