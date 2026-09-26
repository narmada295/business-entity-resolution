from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "data" / "train"
OUTPUT_DIR = ROOT / "outputs"

FEATURE_PATH = (
    OUTPUT_DIR
    / "matcher_features_s2_india_sample.parquet"
)

PREDICTION_PATH = (
    OUTPUT_DIR
    / "lightgbm_validation_predictions.parquet"
)

S1_PATH = (
    TRAIN_DIR
    / "train_source1.tsv"
)

S2_PATH = (
    TRAIN_DIR
    / "train_source2.tsv"
)

FN_PATH = (
    OUTPUT_DIR
    / "matcher_false_negatives.csv"
)

FP_PATH = (
    OUTPUT_DIR
    / "matcher_false_positives.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "matcher_error_summary.txt"
)


# Best threshold from our previous run
THRESHOLD = 0.527

CHUNK_SIZE = 100_000


# ============================================================
# LOAD DATA
# ============================================================

def load_validation_pairs():

    print("Loading feature table...")

    features = pd.read_parquet(
        FEATURE_PATH
    )

    print(
        f"Feature rows: "
        f"{len(features):,}"
    )

    print(
        "\nLoading validation predictions..."
    )

    preds = pd.read_parquet(
        PREDICTION_PATH
    )

    print(
        f"Validation prediction rows: "
        f"{len(preds):,}"
    )

    # Predictions contain metadata + probability.
    # Merge probabilities back into the full feature rows.

    df = features.merge(
        preds[
            [
                "source1_entity_id",
                "candidate_entity_id",
                "probability",
            ]
        ],
        on=[
            "source1_entity_id",
            "candidate_entity_id",
        ],
        how="inner",
        validate="one_to_one",
    )

    print(
        f"Merged validation rows: "
        f"{len(df):,}"
    )

    if len(df) != len(preds):

        raise ValueError(
            "Merged validation rows do not match "
            "prediction rows."
        )

    return df


# ============================================================
# CLASSIFY ERRORS
# ============================================================

def classify_rows(df):

    print(
        f"\nUsing threshold: "
        f"{THRESHOLD:.3f}"
    )

    df = df.copy()

    df["predicted_label"] = (
        df["probability"]
        >= THRESHOLD
    ).astype(int)

    false_negatives = df[
        (df["label"] == 1)
        &
        (df["predicted_label"] == 0)
    ].copy()

    false_positives = df[
        (df["label"] == 0)
        &
        (df["predicted_label"] == 1)
    ].copy()

    true_positives = df[
        (df["label"] == 1)
        &
        (df["predicted_label"] == 1)
    ].copy()

    true_negatives = df[
        (df["label"] == 0)
        &
        (df["predicted_label"] == 0)
    ].copy()

    print()

    print(
        f"True positives: "
        f"{len(true_positives):,}"
    )

    print(
        f"False negatives: "
        f"{len(false_negatives):,}"
    )

    print(
        f"False positives: "
        f"{len(false_positives):,}"
    )

    print(
        f"True negatives: "
        f"{len(true_negatives):,}"
    )

    return (
        false_negatives,
        false_positives,
        true_positives,
        true_negatives,
    )


# ============================================================
# LOAD ORIGINAL S1 TEXT
# ============================================================

def load_s1_text(s1_ids):

    print(
        "\nLoading original S1 records..."
    )

    needed = set(
        s1_ids
    )

    result = {}

    reader = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        chunksize=CHUNK_SIZE,
    )

    for chunk in reader:

        selected = chunk[
            chunk[
                "entity_id"
            ].isin(
                needed
            )
        ]

        for row in selected.itertuples(
            index=False
        ):

            result[
                row.entity_id
            ] = {
                "s1_business_name":
                    row.business_name,

                "s1_business_address":
                    row.business_address,

                "s1_country":
                    row.country,
            }

    print(
        f"S1 records loaded: "
        f"{len(result):,}"
    )

    return result


# ============================================================
# LOAD ORIGINAL S2 TEXT
# ============================================================

def load_s2_text(s2_ids):

    print(
        "\nLoading original S2 records..."
    )

    needed = set(
        s2_ids
    )

    result = {}

    total_read = 0

    reader = pd.read_csv(
        S2_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
        chunksize=CHUNK_SIZE,
    )

    for chunk_no, chunk in enumerate(
        reader,
        start=1,
    ):

        total_read += len(chunk)

        selected = chunk[
            chunk[
                "entity_id"
            ].isin(
                needed
            )
        ]

        for row in selected.itertuples(
            index=False
        ):

            result[
                row.entity_id
            ] = {
                "s2_business_name":
                    row.business_name,

                "s2_business_address":
                    row.business_address,

                "s2_country":
                    row.country,
            }

        print(
            f"\r"
            f"chunks={chunk_no:<3}"
            f" read={total_read:,}"
            f" found={len(result):,}"
            f"/{len(needed):,}",
            end="",
            flush=True,
        )

    print()

    print(
        f"S2 records loaded: "
        f"{len(result):,}"
    )

    return result


# ============================================================
# ADD RAW TEXT
# ============================================================

def add_original_text(
    df,
    s1_lookup,
    s2_lookup,
):

    df = df.copy()

    df[
        "s1_business_name"
    ] = df[
        "source1_entity_id"
    ].map(
        lambda x:
            s1_lookup.get(
                x,
                {},
            ).get(
                "s1_business_name",
                "",
            )
    )

    df[
        "s1_business_address"
    ] = df[
        "source1_entity_id"
    ].map(
        lambda x:
            s1_lookup.get(
                x,
                {},
            ).get(
                "s1_business_address",
                "",
            )
    )

    df[
        "s2_business_name"
    ] = df[
        "candidate_entity_id"
    ].map(
        lambda x:
            s2_lookup.get(
                x,
                {},
            ).get(
                "s2_business_name",
                "",
            )
    )

    df[
        "s2_business_address"
    ] = df[
        "candidate_entity_id"
    ].map(
        lambda x:
            s2_lookup.get(
                x,
                {},
            ).get(
                "s2_business_address",
                "",
            )
    )

    return df


# ============================================================
# ERROR SUMMARY
# ============================================================

def summarize_error_group(
    df,
    name,
):

    lines = []

    lines.append(
        f"\n{name}"
    )

    lines.append(
        "-" * len(name)
    )

    lines.append(
        f"Rows: {len(df):,}"
    )

    if len(df) == 0:
        return lines

    features = [
        "probability",

        "name_ratio",
        "name_token_set",
        "name_token_jaccard",

        "translit_name_ratio",
        "translit_name_token_set",

        "address_ratio",
        "address_token_set",
        "address_token_jaccard",

        "shared_number_count",
        "number_jaccard",

        "name_length_ratio",
        "address_length_ratio",

        "blocker_count",
    ]

    for feature in features:

        if feature not in df.columns:
            continue

        values = df[
            feature
        ]

        lines.append(
            f"{feature:28s} "
            f"mean={values.mean():.4f} "
            f"median={values.median():.4f} "
            f"p25={values.quantile(.25):.4f} "
            f"p75={values.quantile(.75):.4f}"
        )

    # --------------------------------------------
    # Important buckets
    # --------------------------------------------

    buckets = {
        "name_ratio >= 0.8":
            df["name_ratio"] >= 0.8,

        "translit_name_ratio >= 0.8":
            df["translit_name_ratio"] >= 0.8,

        "address_ratio >= 0.8":
            df["address_ratio"] >= 0.8,

        "address_token_jaccard >= 0.5":
            df["address_token_jaccard"] >= 0.5,

        "number_jaccard >= 0.5":
            df["number_jaccard"] >= 0.5,

        "shared number":
            df["shared_number_count"] > 0,

        "missing S2 address":
            df["s2_address_missing"] == 1,

        "name blocker":
            df["from_name_blocker"] == 1,

        "address blocker":
            df["from_address_blocker"] == 1,

        "structured blocker":
            df["from_structured_blocker"] == 1,

        "embedding blocker":
            df["from_embedding_blocker"] == 1,
    }

    lines.append(
        "\nBuckets:"
    )

    for bucket_name, mask in buckets.items():

        count = int(
            mask.sum()
        )

        pct = (
            count
            / len(df)
            * 100
        )

        lines.append(
            f"{bucket_name:32s} "
            f"{count:6,d} "
            f"({pct:6.2f}%)"
        )

    return lines


# ============================================================
# SAVE USEFUL ERROR TABLES
# ============================================================

def reorder_columns(df):

    preferred = [
        "source1_entity_id",
        "candidate_entity_id",

        "s1_business_name",
        "s2_business_name",

        "s1_business_address",
        "s2_business_address",

        "label",
        "predicted_label",
        "probability",

        "name_ratio",
        "name_token_sort",
        "name_token_set",
        "name_token_jaccard",
        "name_length_ratio",

        "translit_name_ratio",
        "translit_name_token_sort",
        "translit_name_token_set",

        "address_ratio",
        "address_token_sort",
        "address_token_set",
        "address_token_jaccard",
        "address_length_ratio",

        "shared_number_count",
        "number_jaccard",
        "number_exact_set",

        "from_name_blocker",
        "from_address_blocker",
        "from_structured_blocker",
        "from_embedding_blocker",

        "blocker_count",

        "name_retrieval_score",
        "address_retrieval_score",
        "embedding_rank",
    ]

    remaining = [
        col
        for col in df.columns
        if col not in preferred
    ]

    cols = [
        col
        for col in preferred
        if col in df.columns
    ]

    cols += remaining

    return df[
        cols
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_validation_pairs()

    (
        false_negatives,
        false_positives,
        true_positives,
        true_negatives,
    ) = classify_rows(
        df
    )

    # --------------------------------------------
    # We only need raw records appearing in errors
    # --------------------------------------------

    error_df = pd.concat(
        [
            false_negatives,
            false_positives,
        ],
        ignore_index=True,
    )

    s1_ids = (
        error_df[
            "source1_entity_id"
        ]
        .unique()
        .tolist()
    )

    s2_ids = (
        error_df[
            "candidate_entity_id"
        ]
        .unique()
        .tolist()
    )

    s1_lookup = load_s1_text(
        s1_ids
    )

    s2_lookup = load_s2_text(
        s2_ids
    )

    false_negatives = (
        add_original_text(
            false_negatives,
            s1_lookup,
            s2_lookup,
        )
    )

    false_positives = (
        add_original_text(
            false_positives,
            s1_lookup,
            s2_lookup,
        )
    )

    # --------------------------------------------
    # Sort errors by confidence
    # --------------------------------------------

    # Most painful FN first:
    # true matches given the LOWEST probabilities.
    false_negatives = (
        false_negatives
        .sort_values(
            "probability",
            ascending=True,
        )
    )

    # Most dangerous FP first:
    # false matches given the HIGHEST probabilities.
    false_positives = (
        false_positives
        .sort_values(
            "probability",
            ascending=False,
        )
    )

    false_negatives = (
        reorder_columns(
            false_negatives
        )
    )

    false_positives = (
        reorder_columns(
            false_positives
        )
    )

    # --------------------------------------------
    # Write CSVs
    # --------------------------------------------

    false_negatives.to_csv(
        FN_PATH,
        index=False,
    )

    false_positives.to_csv(
        FP_PATH,
        index=False,
    )

    # --------------------------------------------
    # Summary
    # --------------------------------------------

    lines = []

    lines.append(
        "MATCHER ERROR ANALYSIS"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        f"Threshold: "
        f"{THRESHOLD:.3f}"
    )

    lines.append(
        f"Validation pairs: "
        f"{len(df):,}"
    )

    lines.append(
        f"True positives: "
        f"{len(true_positives):,}"
    )

    lines.append(
        f"False negatives: "
        f"{len(false_negatives):,}"
    )

    lines.append(
        f"False positives: "
        f"{len(false_positives):,}"
    )

    lines.append(
        f"True negatives: "
        f"{len(true_negatives):,}"
    )

    lines.extend(
        summarize_error_group(
            false_negatives,
            "FALSE NEGATIVES",
        )
    )

    lines.extend(
        summarize_error_group(
            false_positives,
            "FALSE POSITIVES",
        )
    )

    # --------------------------------------------
    # Near-threshold errors
    # --------------------------------------------

    fn_near = false_negatives[
        false_negatives[
            "probability"
        ] >= 0.30
    ]

    fp_near = false_positives[
        false_positives[
            "probability"
        ] <= 0.60
    ]

    lines.append(
        "\nNEAR-THRESHOLD ERRORS"
    )

    lines.append(
        "-" * 30
    )

    lines.append(
        f"FN probability >= 0.30: "
        f"{len(fn_near):,}"
    )

    lines.append(
        f"FP probability <= 0.60: "
        f"{len(fp_near):,}"
    )

    # --------------------------------------------
    # Very confident mistakes
    # --------------------------------------------

    confident_fn = false_negatives[
        false_negatives[
            "probability"
        ] <= 0.05
    ]

    confident_fp = false_positives[
        false_positives[
            "probability"
        ] >= 0.90
    ]

    lines.append(
        "\nHIGH-CONFIDENCE MISTAKES"
    )

    lines.append(
        "-" * 30
    )

    lines.append(
        f"FN probability <= 0.05: "
        f"{len(confident_fn):,}"
    )

    lines.append(
        f"FP probability >= 0.90: "
        f"{len(confident_fp):,}"
    )

    summary_text = "\n".join(
        lines
    )

    print()
    print(summary_text)

    SUMMARY_PATH.write_text(
        summary_text,
        encoding="utf-8",
    )

    print()

    print(
        f"Saved false negatives:\n"
        f"{FN_PATH}"
    )

    print()

    print(
        f"Saved false positives:\n"
        f"{FP_PATH}"
    )

    print()

    print(
        f"Saved summary:\n"
        f"{SUMMARY_PATH}"
    )

    # --------------------------------------------
    # Show example mistakes directly
    # --------------------------------------------

    display_cols = [
        "source1_entity_id",
        "candidate_entity_id",
        "probability",

        "s1_business_name",
        "s2_business_name",

        "s1_business_address",
        "s2_business_address",

        "translit_name_ratio",
        "address_token_jaccard",
        "number_jaccard",
    ]

    print()

    print(
        "=" * 80
    )

    print(
        "10 MOST CONFIDENT FALSE NEGATIVES"
    )

    print(
        "=" * 80
    )

    print(
        false_negatives[
            display_cols
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print()

    print(
        "=" * 80
    )

    print(
        "10 MOST CONFIDENT FALSE POSITIVES"
    )

    print(
        "=" * 80
    )

    print(
        false_positives[
            display_cols
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()