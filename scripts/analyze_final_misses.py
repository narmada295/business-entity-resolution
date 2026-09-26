from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

FINAL_MISSES = (
    OUTPUT_DIR
    / "final_cached_blocking_misses.tsv"
)

OLD_MISS_DETAILS = (
    OUTPUT_DIR
    / "s2_india_blocking_misses.csv"
)

TRANSLIT_DETAILS = (
    OUTPUT_DIR
    / "transliteration_miss_analysis.csv"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "final_blocking_miss_analysis.csv"
)


def main():

    # --------------------------------------------------
    # Load
    # --------------------------------------------------

    final_misses = pd.read_csv(
        FINAL_MISSES,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    old = pd.read_csv(
        OLD_MISS_DETAILS,
        dtype=str,
        keep_default_na=False,
    )

    translit = pd.read_csv(
        TRANSLIT_DETAILS,
        dtype=str,
        keep_default_na=False,
    )

    print(
        f"Final misses: "
        f"{len(final_misses):,}"
    )

    # --------------------------------------------------
    # Rename IDs so files line up
    # --------------------------------------------------

    final_misses = final_misses.rename(
        columns={
            "source1_entity_id": "s1_id",
            "source2_entity_id": "s2_id",
        }
    )

    # Old miss table already has:
    # s1_id, s2_id, names, addresses, jaccard, etc.

    # Keep useful transliteration fields only.
    translit = translit[
        [
            "s1_id",
            "s2_id",
            "s1_name_translit",
            "s2_name_translit",
            "original_name_score",
            "translit_name_score",
            "translit_gain",
            "cross_script_like",
        ]
    ]

    # --------------------------------------------------
    # Join
    # --------------------------------------------------

    df = final_misses.merge(
        old,
        on=[
            "s1_id",
            "s2_id",
        ],
        how="left",
    )

    df = df.merge(
        translit,
        on=[
            "s1_id",
            "s2_id",
        ],
        how="left",
    )

    print(
        f"Joined rows: "
        f"{len(df):,}"
    )

    # --------------------------------------------------
    # Numeric columns
    # --------------------------------------------------

    numeric_cols = [
        "name_token_jaccard",
        "address_token_jaccard",
        "number_overlap",
        "target_address_missing",
        "original_name_score",
        "translit_name_score",
        "translit_gain",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # bool came back as text
    df["cross_script_like"] = (
        df["cross_script_like"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    # --------------------------------------------------
    # Main statistics
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL MISS CHARACTERISTICS")
    print("=" * 80)

    print(
        f"Remaining true links: "
        f"{len(df):,}"
    )

    print()

    print(
        "Missing target address:"
    )

    missing_addr = (
        df[
            "target_address_missing"
        ]
        .fillna(0)
        .mean()
    )

    print(
        f"{missing_addr:.2%}"
    )

    print()

    print(
        "Cross-script-like:"
    )

    print(
        f"{df['cross_script_like'].mean():.2%}"
    )

    print()

    print(
        "At least one shared address number:"
    )

    has_number = (
        df["number_overlap"]
        .fillna(0)
        > 0
    ).mean()

    print(
        f"{has_number:.2%}"
    )

    print()

    print(
        "Name token Jaccard:"
    )

    print(
        df[
            "name_token_jaccard"
        ].describe(
            percentiles=[
                .25,
                .5,
                .75,
                .9,
            ]
        )
    )

    print()

    print(
        "Address token Jaccard:"
    )

    print(
        df[
            "address_token_jaccard"
        ].describe(
            percentiles=[
                .25,
                .5,
                .75,
                .9,
            ]
        )
    )

    print()

    print(
        "Transliterated-name score:"
    )

    print(
        df[
            "translit_name_score"
        ].describe(
            percentiles=[
                .25,
                .5,
                .75,
                .9,
            ]
        )
    )

    # --------------------------------------------------
    # Useful failure buckets
    # --------------------------------------------------

    df["bucket_missing_address"] = (
        df["target_address_missing"]
        == 1
    )

    df["bucket_cross_script"] = (
        df["cross_script_like"]
    )

    df["bucket_number_overlap"] = (
        df["number_overlap"]
        .fillna(0)
        > 0
    )

    df["bucket_high_address_overlap"] = (
        df[
            "address_token_jaccard"
        ]
        .fillna(0)
        >= 0.5
    )

    df["bucket_high_name_overlap"] = (
        df[
            "name_token_jaccard"
        ]
        .fillna(0)
        >= 0.5
    )

    df["bucket_good_translit"] = (
        df[
            "translit_name_score"
        ]
        .fillna(0)
        >= 70
    )

    print()
    print("=" * 80)
    print("FAILURE BUCKETS")
    print("=" * 80)

    buckets = [
        "bucket_missing_address",
        "bucket_cross_script",
        "bucket_number_overlap",
        "bucket_high_address_overlap",
        "bucket_high_name_overlap",
        "bucket_good_translit",
    ]

    for col in buckets:

        count = int(
            df[col].sum()
        )

        pct = (
            count / len(df)
        )

        print(
            f"{col:<32}"
            f"{count:>6,}"
            f"  ({pct:.2%})"
        )

    # --------------------------------------------------
    # More specific combinations
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("IMPORTANT COMBINATIONS")
    print("=" * 80)

    combinations = {
        "number + high address overlap":
            (
                df["bucket_number_overlap"]
                &
                df[
                    "bucket_high_address_overlap"
                ]
            ),

        "cross-script + good translit":
            (
                df["bucket_cross_script"]
                &
                df[
                    "bucket_good_translit"
                ]
            ),

        "missing address + good name":
            (
                df[
                    "bucket_missing_address"
                ]
                &
                df[
                    "bucket_high_name_overlap"
                ]
            ),

        "no number + low address + low name":
            (
                ~df[
                    "bucket_number_overlap"
                ]
                &
                ~df[
                    "bucket_high_address_overlap"
                ]
                &
                ~df[
                    "bucket_high_name_overlap"
                ]
            ),
    }

    for label, mask in combinations.items():

        count = int(
            mask.sum()
        )

        print(
            f"{label:<40}"
            f"{count:>6,}"
            f"  ({count / len(df):.2%})"
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Saved:\n{OUTPUT_PATH}"
    )

    # --------------------------------------------------
    # Examples from useful buckets
    # --------------------------------------------------

    print()
    print("=" * 80)
    print("NUMBER + HIGH ADDRESS OVERLAP EXAMPLES")
    print("=" * 80)

    subset = df[
        df["bucket_number_overlap"]
        &
        df["bucket_high_address_overlap"]
    ].head(10)

    for row in subset.itertuples(
        index=False
    ):

        print()
        print(
            "S1:",
            row.s1_name,
        )

        print(
            "S2:",
            row.s2_name,
        )

        print(
            "S1 addr:",
            row.s1_address,
        )

        print(
            "S2 addr:",
            row.s2_address,
        )

        print(
            "Address Jaccard:",
            row.address_token_jaccard,
        )

        print(
            "Number overlap:",
            row.number_overlap,
        )

    print()
    print("=" * 80)
    print("CROSS-SCRIPT + GOOD TRANSLITERATION EXAMPLES")
    print("=" * 80)

    subset = df[
        df["bucket_cross_script"]
        &
        df["bucket_good_translit"]
    ].head(10)

    for row in subset.itertuples(
        index=False
    ):

        print()
        print(
            "S1:",
            row.s1_name,
        )

        print(
            "S2:",
            row.s2_name,
        )

        print(
            "S1 translit:",
            row.s1_name_translit,
        )

        print(
            "S2 translit:",
            row.s2_name_translit,
        )

        print(
            "Translit score:",
            row.translit_name_score,
        )


if __name__ == "__main__":
    main()