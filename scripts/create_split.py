import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.split import (
    parse_matches,
    match_count_bucket,
    assign_split,
)


TRAIN_DIR = ROOT / "data" / "train"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(
    exist_ok=True
)

S1_PATH = (
    TRAIN_DIR
    / "train_source1.tsv"
)

GT_PATH = (
    TRAIN_DIR
    / "train_ground_truth.tsv"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "splits.tsv"
)


def main():

    print("Loading S1...")

    # We only need ID + country.
    s1 = pd.read_csv(
        S1_PATH,
        sep="\t",
        usecols=[
            "entity_id",
            "country",
        ],
        dtype=str,
        keep_default_na=False,
    )

    print("Loading ground truth...")

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(
        f"S1 entities: {len(s1):,}"
    )

    # -----------------------------------------------------
    # MATCH COUNTS
    # -----------------------------------------------------

    print(
        "Calculating number of matches..."
    )

    gt["num_matches"] = (
        gt["matched_entity_ids"]
        .map(
            lambda x: (
                0
                if not x.strip()
                else x.count(",") + 1
            )
        )
    )

    gt["match_bucket"] = (
        gt["num_matches"]
        .map(match_count_bucket)
    )

    # -----------------------------------------------------
    # COUNTRY
    # -----------------------------------------------------

    country_lookup = (
        s1
        .set_index("entity_id")[
            "country"
        ]
    )

    gt["country"] = (
        gt["source1_entity_id"]
        .map(country_lookup)
    )

    if gt["country"].isna().any():

        missing = (
            gt["country"]
            .isna()
            .sum()
        )

        raise ValueError(
            f"{missing:,} GT rows "
            "could not be mapped to S1."
        )

    # -----------------------------------------------------
    # SPLIT
    # -----------------------------------------------------

    print(
        "Assigning deterministic "
        "80/20 split..."
    )

    gt["split"] = (
        gt["source1_entity_id"]
        .map(assign_split)
    )

    # Useful diagnostic label.
    gt["stratum"] = (
        gt["country"]
        + "_"
        + gt["match_bucket"]
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("OVERALL SPLIT")
    print("=" * 70)

    overall = (
        gt["split"]
        .value_counts()
    )

    for split, count in overall.items():

        pct = (
            count
            / len(gt)
            * 100
        )

        print(
            f"{split:<10}"
            f"{count:>12,} "
            f"({pct:6.2f}%)"
        )

    print("\n" + "=" * 70)
    print("SPLIT BY COUNTRY")
    print("=" * 70)

    country_table = pd.crosstab(
        gt["country"],
        gt["split"],
        margins=True,
    )

    print(country_table)

    print("\n" + "=" * 70)
    print(
        "SPLIT BY MATCH-COUNT BUCKET"
    )
    print("=" * 70)

    bucket_table = pd.crosstab(
        gt["match_bucket"],
        gt["split"],
        margins=True,
    )

    print(bucket_table)

    print("\n" + "=" * 70)
    print(
        "VALIDATION STRATUM DISTRIBUTION"
    )
    print("=" * 70)

    val = gt[
        gt["split"] == "val"
    ]

    print(
        val["stratum"]
        .value_counts()
        .sort_index()
    )

    # -----------------------------------------------------
    # WRITE
    # -----------------------------------------------------

    output = gt[
        [
            "source1_entity_id",
            "country",
            "num_matches",
            "match_bucket",
            "split",
        ]
    ]

    output.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False,
    )

    print(
        f"\nSaved split file to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()