import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.load_data import load_train_data


EXPECTED_SOURCE_COLUMNS = {
    "entity_id",
    "business_name",
    "business_address",
    "country",
}

EXPECTED_GT_COLUMNS = {
    "source1_entity_id",
    "matched_entity_ids",
}


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def basic_source_stats(name, df):
    print_section(name)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    print("\nMissing / empty values:")
    for col in df.columns:
        empty = (df[col].str.strip() == "").sum()
        pct = empty / len(df) * 100 if len(df) else 0

        print(
            f"  {col:<20} "
            f"{empty:>8,} "
            f"({pct:6.2f}%)"
        )

    if "entity_id" in df.columns:
        print(
            f"\nUnique entity IDs: "
            f"{df['entity_id'].nunique():,}"
        )

        duplicated_ids = df["entity_id"].duplicated().sum()

        print(
            f"Duplicate entity IDs: "
            f"{duplicated_ids:,}"
        )

    if "country" in df.columns:
        print("\nCountry distribution:")

        counts = df["country"].value_counts(
            dropna=False
        )

        for country, count in counts.items():
            pct = count / len(df) * 100

            display_country = (
                country if country else "<EMPTY>"
            )

            print(
                f"  {display_country:<20} "
                f"{count:>8,} "
                f"({pct:6.2f}%)"
            )

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))


def validate_columns(s1, s2, s3, gt):
    print_section("COLUMN VALIDATION")

    for name, df in [
        ("Source 1", s1),
        ("Source 2", s2),
        ("Source 3", s3),
    ]:

        actual = set(df.columns)

        missing = EXPECTED_SOURCE_COLUMNS - actual
        extra = actual - EXPECTED_SOURCE_COLUMNS

        print(f"\n{name}")

        print(f"  Missing expected columns: {missing}")
        print(f"  Extra columns:            {extra}")

    actual_gt = set(gt.columns)

    print("\nGround Truth")

    print(
        "  Missing expected columns:",
        EXPECTED_GT_COLUMNS - actual_gt
    )

    print(
        "  Extra columns:",
        actual_gt - EXPECTED_GT_COLUMNS
    )


def parse_matches(value):
    value = value.strip()

    if not value:
        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


def analyze_ground_truth(gt):
    print_section("GROUND TRUTH")

    gt = gt.copy()

    gt["matches"] = (
        gt["matched_entity_ids"]
        .apply(parse_matches)
    )

    gt["num_matches"] = (
        gt["matches"]
        .apply(len)
    )

    print(
        f"Ground-truth S1 entities: "
        f"{len(gt):,}"
    )

    print("\nNumber of matches per S1:")

    counts = (
        gt["num_matches"]
        .value_counts()
        .sort_index()
    )

    for num_matches, count in counts.items():
        pct = count / len(gt) * 100

        print(
            f"  {num_matches:>3} matches : "
            f"{count:>8,} "
            f"({pct:6.2f}%)"
        )

    singleton_count = (
        gt["num_matches"] == 0
    ).sum()

    single_match_count = (
        gt["num_matches"] == 1
    ).sum()

    multiple_match_count = (
        gt["num_matches"] > 1
    ).sum()

    print("\nSummary:")

    print(
        f"  Singletons:        "
        f"{singleton_count:,}"
    )

    print(
        f"  Exactly 1 match:   "
        f"{single_match_count:,}"
    )

    print(
        f"  Multiple matches: "
        f"{multiple_match_count:,}"
    )

    total_links = gt["num_matches"].sum()

    print(
        f"\nTotal positive links: "
        f"{total_links:,}"
    )

    num_s2 = 0
    num_s3 = 0

    s1_with_s2 = 0
    s1_with_s3 = 0
    s1_with_both = 0

    for matches in gt["matches"]:

        has_s2 = False
        has_s3 = False

        for entity_id in matches:

            if entity_id.startswith("S2-"):
                num_s2 += 1
                has_s2 = True

            elif entity_id.startswith("S3-"):
                num_s3 += 1
                has_s3 = True

        if has_s2:
            s1_with_s2 += 1

        if has_s3:
            s1_with_s3 += 1

        if has_s2 and has_s3:
            s1_with_both += 1

    print("\nPositive links by source:")

    print(
        f"  S2 links: "
        f"{num_s2:,}"
    )

    print(
        f"  S3 links: "
        f"{num_s3:,}"
    )

    print("\nS1 entities with:")

    print(
        f"  ≥1 S2 match: "
        f"{s1_with_s2:,}"
    )

    print(
        f"  ≥1 S3 match: "
        f"{s1_with_s3:,}"
    )

    print(
        f"  matches in both: "
        f"{s1_with_both:,}"
    )

    print("\nLargest match counts:")

    print(
        gt[
            [
                "source1_entity_id",
                "matched_entity_ids",
                "num_matches",
            ]
        ]
        .sort_values(
            "num_matches",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )


def string_stats(name, df):
    print_section(f"{name} TEXT STATISTICS")

    for col in [
        "business_name",
        "business_address",
    ]:

        values = df[col].fillna("")

        char_lengths = values.str.len()

        token_lengths = values.apply(
            lambda x: len(x.split())
        )

        print(f"\n{col}")

        print("Character length:")
        print(
            char_lengths.describe(
                percentiles=[
                    0.25,
                    0.5,
                    0.75,
                    0.9,
                    0.95,
                    0.99,
                ]
            ).to_string()
        )

        print("\nToken count:")

        print(
            token_lengths.describe(
                percentiles=[
                    0.25,
                    0.5,
                    0.75,
                    0.9,
                    0.95,
                    0.99,
                ]
            ).to_string()
        )

        print(
            f"\nUnique values: "
            f"{values.nunique():,}"
        )


def validate_ground_truth_ids(
    s1,
    s2,
    s3,
    gt,
):
    print_section("GROUND TRUTH ID VALIDATION")

    s1_ids = set(s1["entity_id"])
    s2_ids = set(s2["entity_id"])
    s3_ids = set(s3["entity_id"])

    invalid_source1 = []

    missing_targets = []

    duplicate_targets = []

    for _, row in gt.iterrows():

        source1_id = row["source1_entity_id"]

        if source1_id not in s1_ids:
            invalid_source1.append(
                source1_id
            )

        matches = parse_matches(
            row["matched_entity_ids"]
        )

        if len(matches) != len(set(matches)):
            duplicate_targets.append(
                source1_id
            )

        for candidate in matches:

            if (
                candidate not in s2_ids
                and candidate not in s3_ids
            ):
                missing_targets.append(
                    (source1_id, candidate)
                )

    print(
        f"GT source1 IDs missing from S1: "
        f"{len(invalid_source1):,}"
    )

    print(
        f"GT target IDs missing from S2/S3: "
        f"{len(missing_targets):,}"
    )

    print(
        f"GT rows containing duplicate targets: "
        f"{len(duplicate_targets):,}"
    )

    if invalid_source1[:10]:
        print(
            "\nExamples invalid S1:",
            invalid_source1[:10],
        )

    if missing_targets[:10]:
        print(
            "\nExamples missing targets:",
            missing_targets[:10],
        )


def main():

    print("Loading training data...")

    s1, s2, s3, gt = load_train_data()

    validate_columns(
        s1,
        s2,
        s3,
        gt,
    )

    basic_source_stats(
        "SOURCE 1",
        s1,
    )

    basic_source_stats(
        "SOURCE 2",
        s2,
    )

    basic_source_stats(
        "SOURCE 3",
        s3,
    )

    analyze_ground_truth(gt)

    validate_ground_truth_ids(
        s1,
        s2,
        s3,
        gt,
    )

    string_stats(
        "SOURCE 1",
        s1,
    )

    string_stats(
        "SOURCE 2",
        s2,
    )

    string_stats(
        "SOURCE 3",
        s3,
    )


if __name__ == "__main__":
    main()