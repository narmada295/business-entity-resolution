from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.hybrid_blocking import HybridBlocker

from src.features import (
    prepare_record,
    compute_pair_features,
    MODEL_FEATURE_COLUMNS,
)


TRAIN_DIR = ROOT / "data" / "train"
INDEX_DIR = ROOT / "outputs" / "indexes"
OUTPUT_ROOT = ROOT / "outputs" / "training_pairs"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"

SPLIT_PATH = ROOT / "outputs" / "splits.tsv"


# ============================================================
# HELPERS
# ============================================================

def safe_country_name(country):
    return (
        country
        .replace("/", "_")
        .replace(" ", "_")
    )


def parse_matches(value):

    if not value:
        return set()

    return {
        x.strip()
        for x in value.split(",")
        if x.strip()
    }


def deterministic_split(entity_id):

    value = hashlib.md5(
        f"42:{entity_id}".encode()
    ).hexdigest()

    number = int(
        value,
        16,
    )

    ratio = number / (16 ** 32)

    return (
        "train"
        if ratio < 0.8
        else "val"
    )


def normalize_split_name(value):

    value = str(
        value
    ).strip().lower()

    if value in {
        "val",
        "valid",
        "validation",
        "dev",
    }:
        return "val"

    return "train"


# ============================================================
# SPLITS
# ============================================================

def assign_splits(query_df):

    fallback = {
        entity_id:
            deterministic_split(
                entity_id
            )
        for entity_id
        in query_df["entity_id"]
    }


    if not SPLIT_PATH.exists():

        query_df["split"] = (
            query_df["entity_id"]
            .map(fallback)
        )

        return query_df


    split_df = pd.read_csv(
        SPLIT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )


    id_candidates = [
        "entity_id",
        "source1_entity_id",
    ]

    split_candidates = [
        "split",
        "set",
        "partition",
    ]


    id_col = next(
        (
            x
            for x in id_candidates
            if x in split_df.columns
        ),
        None,
    )


    split_col = next(
        (
            x
            for x in split_candidates
            if x in split_df.columns
        ),
        None,
    )


    if (
        id_col is None
        or split_col is None
    ):

        print(
            "Could not understand splits.tsv; "
            "using deterministic 80/20 split."
        )

        query_df["split"] = (
            query_df["entity_id"]
            .map(fallback)
        )

        return query_df


    split_map = dict(
        zip(
            split_df[id_col],
            split_df[split_col]
            .map(normalize_split_name),
        )
    )


    query_df["split"] = [
        split_map.get(
            entity_id,
            fallback[entity_id],
        )
        for entity_id
        in query_df["entity_id"]
    ]


    return query_df


# ============================================================
# LOAD QUERIES
# ============================================================

def load_queries(
    country,
    max_entities,
    seed,
):

    print(
        "Loading S1..."
    )


    s1 = pd.read_csv(
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
    )


    query_df = (
        s1[
            s1["country"]
            == country
        ]
        .copy()
    )


    del s1


    print(
        "Available S1 entities:",
        f"{len(query_df):,}",
    )


    if (
        max_entities is not None
        and
        max_entities > 0
        and
        len(query_df) > max_entities
    ):

        query_df = (
            query_df.sample(
                n=max_entities,
                random_state=seed,
            )
        )


    # Deterministic ordering after sample.
    query_df = (
        query_df
        .sort_values(
            "entity_id"
        )
        .reset_index(
            drop=True
        )
    )


    query_df = assign_splits(
        query_df
    )


    print(
        "Selected:",
        f"{len(query_df):,}",
    )

    print(
        query_df[
            "split"
        ].value_counts()
    )


    return query_df


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth(
    query_ids,
):

    print(
        "Loading GT for selected S1 entities..."
    )


    query_ids = set(
        query_ids
    )


    result = {
        qid: set()
        for qid in query_ids
    }


    reader = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
        chunksize=200_000,
    )


    for chunk in reader:

        chunk = chunk[
            chunk[
                "source1_entity_id"
            ].isin(
                query_ids
            )
        ]


        for row in chunk.itertuples(
            index=False
        ):

            result[
                row.source1_entity_id
            ] = parse_matches(
                row.matched_entity_ids
            )


    return result


# ============================================================
# HARD NEGATIVE SELECTION
# ============================================================

def candidate_hardness(
    provenance,
):

    name_score = float(
        provenance.get(
            "name_retrieval_score",
            0.0,
        )
    )

    address_score = float(
        provenance.get(
            "address_retrieval_score",
            0.0,
        )
    )

    structured_score = float(
        provenance.get(
            "structured_retrieval_score",
            0.0,
        )
    )


    blocker_count = (
        int(
            provenance.get(
                "from_name_blocker",
                0,
            )
        )
        +
        int(
            provenance.get(
                "from_address_blocker",
                0,
            )
        )
        +
        int(
            provenance.get(
                "from_structured_blocker",
                0,
            )
        )
    )


    return (
        max(
            name_score,
            address_score,
            structured_score,
        )
        +
        0.05 * blocker_count
    )


def select_candidates(
    candidates,
    true_ids,
    split,
    max_train_negatives,
):

    positives = []
    negatives = []


    for candidate_id, provenance in (
        candidates.items()
    ):

        item = (
            candidate_id,
            provenance,
        )


        if candidate_id in true_ids:

            positives.append(
                item
            )

        else:

            negatives.append(
                item
            )


    # Validation MUST retain all candidates.
    if split == "val":

        return (
            positives
            +
            negatives
        )


    # Training: retain all positives +
    # hardest negatives.
    negatives.sort(
        key=lambda item:
            candidate_hardness(
                item[1]
            ),
        reverse=True,
    )


    negatives = negatives[
        :max_train_negatives
    ]


    return (
        positives
        +
        negatives
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--source",
        required=True,
        choices=[
            "S2",
            "S3",
        ],
    )


    parser.add_argument(
        "--country",
        required=True,
    )


    parser.add_argument(
        "--max-entities",
        type=int,
        default=25_000,
    )


    parser.add_argument(
        "--batch-size",
        type=int,
        default=1_000,
    )


    parser.add_argument(
        "--max-train-negatives",
        type=int,
        default=20,
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )


    args = parser.parse_args()


    SOURCE = (
        args.source.upper()
    )

    COUNTRY = (
        args.country
    )


    source_prefix = (
        f"{SOURCE}-"
    )


    safe_country = (
        safe_country_name(
            COUNTRY
        )
    )


    output_dir = (
        OUTPUT_ROOT
        / SOURCE
        / safe_country
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # QUERIES
    # ========================================================

    query_df = load_queries(
        COUNTRY,
        args.max_entities,
        args.seed,
    )


    query_manifest = (
        output_dir
        / "queries.parquet"
    )


    query_df[
        [
            "entity_id",
            "country",
            "split",
        ]
    ].to_parquet(
        query_manifest,
        index=False,
        compression="zstd",
    )


    # ========================================================
    # GT
    # ========================================================

    gt_map = load_ground_truth(
        query_df[
            "entity_id"
        ]
    )


    # ========================================================
    # BLOCKER
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        f"Loading HybridBlocker "
        f"{SOURCE}/{COUNTRY}"
    )

    print(
        "=" * 70
    )


    blocker = HybridBlocker(
        INDEX_DIR,
        SOURCE,
        COUNTRY,
    )


    # Index target record cache by entity ID.
    #
    # We keep original columns available.
    blocker.records.set_index(
        "entity_id",
        inplace=True,
        drop=False,
    )


    # ========================================================
    # BATCHES
    # ========================================================

    num_batches = (
        len(query_df)
        + args.batch_size
        - 1
    ) // args.batch_size


    print()
    print(
        "Batches:",
        num_batches,
    )


    for batch_no in range(
        num_batches
    ):

        start_idx = (
            batch_no
            * args.batch_size
        )

        end_idx = min(
            len(query_df),
            start_idx
            + args.batch_size,
        )


        part_path = (
            output_dir
            / f"part_{batch_no:05d}.parquet"
        )


        stats_path = (
            output_dir
            / f"part_{batch_no:05d}.json"
        )


        # Resume-safe.
        if part_path.exists():

            print(
                f"[{batch_no + 1}/{num_batches}] "
                f"already exists → skip"
            )

            continue


        batch_started = (
            time.perf_counter()
        )


        batch_df = (
            query_df.iloc[
                start_idx:end_idx
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )


        print()
        print(
            "=" * 70
        )

        print(
            f"BATCH "
            f"{batch_no + 1}/{num_batches}"
        )

        print(
            f"Rows "
            f"{start_idx:,}"
            f"–"
            f"{end_idx - 1:,}"
        )

        print(
            "=" * 70
        )


        # ====================================================
        # CANDIDATES
        # ====================================================

        retrieval_started = (
            time.perf_counter()
        )


        results = blocker.generate(
            batch_df
        )


        retrieval_time = (
            time.perf_counter()
            - retrieval_started
        )


        # ====================================================
        # SELECT TRAIN/VAL CANDIDATES
        # ====================================================

        selected_per_query = []

        unique_target_ids = set()

        truth_links = 0
        recovered_truth_links = 0

        candidate_count_before = 0
        candidate_count_after = 0


        for idx, row in batch_df.iterrows():

            qid = row[
                "entity_id"
            ]

            split = row[
                "split"
            ]


            all_truth = gt_map.get(
                qid,
                set(),
            )


            source_truth = {
                candidate_id
                for candidate_id
                in all_truth
                if candidate_id.startswith(
                    source_prefix
                )
            }


            candidates = results[
                idx
            ]


            truth_links += len(
                source_truth
            )


            recovered_truth_links += len(
                source_truth
                &
                set(
                    candidates.keys()
                )
            )


            candidate_count_before += len(
                candidates
            )


            selected = select_candidates(
                candidates,
                source_truth,
                split,
                args.max_train_negatives,
            )


            candidate_count_after += len(
                selected
            )


            selected_per_query.append(
                selected
            )


            for candidate_id, _ in selected:

                unique_target_ids.add(
                    candidate_id
                )


        # ====================================================
        # PREPARE TARGET RECORDS ONCE
        # ====================================================

        target_prepared = {}


        if unique_target_ids:

            target_ids_list = list(
                unique_target_ids
            )


            target_subset = (
                blocker.records.loc[
                    target_ids_list
                ]
            )


            for target_row in (
                target_subset.itertuples(
                    index=False
                )
            ):

                target_prepared[
                    target_row.entity_id
                ] = prepare_record(
                    target_row.name_norm,
                    target_row.address_norm,
                )


        # ====================================================
        # COMPUTE FEATURES
        # ====================================================

        feature_started = (
            time.perf_counter()
        )


        rows = []


        for idx, row in batch_df.iterrows():

            qid = row[
                "entity_id"
            ]


            split = row[
                "split"
            ]


            query_record = prepare_record(
                row[
                    "business_name"
                ],
                row[
                    "business_address"
                ],
            )


            all_truth = gt_map.get(
                qid,
                set(),
            )


            for (
                candidate_id,
                provenance,
            ) in selected_per_query[
                idx
            ]:

                target_record = (
                    target_prepared.get(
                        candidate_id
                    )
                )


                if target_record is None:

                    raise RuntimeError(
                        "Target record missing for "
                        f"{candidate_id}"
                    )


                features = (
                    compute_pair_features(
                        query_record,
                        target_record,
                        provenance,
                    )
                )


                output_row = {
                    "source1_entity_id":
                        qid,

                    "candidate_entity_id":
                        candidate_id,

                    "source":
                        SOURCE,

                    "country":
                        COUNTRY,

                    "split":
                        split,

                    "label":
                        int(
                            candidate_id
                            in all_truth
                        ),
                }


                output_row.update(
                    features
                )


                rows.append(
                    output_row
                )


        feature_time = (
            time.perf_counter()
            - feature_started
        )


        output_df = pd.DataFrame(
            rows
        )


        # Explicit ordering.
        output_columns = [
            "source1_entity_id",
            "candidate_entity_id",
            "source",
            "country",
            "split",
            "label",
        ] + list(
            MODEL_FEATURE_COLUMNS
        )


        output_df = output_df[
            output_columns
        ]


        output_df.to_parquet(
            part_path,
            index=False,
            compression="zstd",
        )


        batch_time = (
            time.perf_counter()
            - batch_started
        )


        stats = {
            "source":
                SOURCE,

            "country":
                COUNTRY,

            "batch":
                batch_no,

            "query_count":
                len(batch_df),

            "pair_count":
                len(output_df),

            "positive_pairs":
                int(
                    output_df[
                        "label"
                    ].sum()
                ),

            "candidates_before_sampling":
                candidate_count_before,

            "candidates_after_sampling":
                candidate_count_after,

            "truth_links":
                truth_links,

            "recovered_truth_links":
                recovered_truth_links,

            "blocking_recall":
                (
                    recovered_truth_links
                    / truth_links
                    if truth_links
                    else None
                ),

            "retrieval_seconds":
                retrieval_time,

            "feature_seconds":
                feature_time,

            "total_seconds":
                batch_time,
        }


        with open(
            stats_path,
            "w",
        ) as f:

            json.dump(
                stats,
                f,
                indent=2,
            )


        print(
            "Queries:",
            len(batch_df),
        )

        print(
            "Pairs written:",
            f"{len(output_df):,}",
        )

        print(
            "Positive pairs:",
            f"{int(output_df['label'].sum()):,}",
        )

        print(
            "Candidate reduction:",
            f"{candidate_count_before:,}"
            " → "
            f"{candidate_count_after:,}",
        )


        if truth_links:

            print(
                "Blocking recall:",
                f"{recovered_truth_links / truth_links:.4%}",
            )


        print(
            "Retrieval:",
            f"{retrieval_time:.2f}s",
        )

        print(
            "Features:",
            f"{feature_time:.2f}s",
        )

        print(
            "Batch total:",
            f"{batch_time:.2f}s",
        )


    print()
    print(
        "=" * 70
    )

    print(
        f"FINISHED "
        f"{SOURCE}/{COUNTRY}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()