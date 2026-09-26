from pathlib import Path
import argparse
import gc
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from src.hybrid_blocking import HybridBlocker

from src.features import (
    prepare_record,
    compute_pair_features,
    MODEL_FEATURE_COLUMNS,
)


TEST_DIR = (
    ROOT
    / "data"
    / "test"
)

INDEX_DIR = (
    ROOT
    / "outputs"
    / "test_indexes"
)

MODEL_PATH = (
    ROOT
    / "outputs"
    / "models"
    / "lightgbm_v2.txt"
)

METRICS_PATH = (
    ROOT
    / "outputs"
    / "models"
    / "metrics.json"
)

OUTPUT_ROOT = (
    ROOT
    / "outputs"
    / "test_inference"
)

S1_PATH = (
    TEST_DIR
    / "test_source1.tsv"
)


def safe_country_name(country):
    return (
        country
        .replace("/", "_")
        .replace(" ", "_")
    )


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
        "--batch-size",
        type=int,
        default=1000,
    )


    parser.add_argument(
        "--max-entities",
        type=int,
        default=0,
        help=(
            "0 means all entities. "
            "Useful for benchmarking."
        ),
    )


    args = parser.parse_args()


    source = args.source.upper()
    country = args.country


    safe_country = safe_country_name(
        country
    )


    output_dir = (
        OUTPUT_ROOT
        / source
        / safe_country
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # MODEL
    # ========================================================

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            MODEL_PATH
        )


    model = lgb.Booster(
        model_file=str(
            MODEL_PATH
        )
    )


    with open(
        METRICS_PATH
    ) as f:

        metrics = json.load(
            f
        )


    threshold = float(
        metrics[
            "best_threshold"
        ]
    )


    print(
        "Threshold:",
        threshold,
    )


    # ========================================================
    # S1
    # ========================================================

    print(
        "Loading test S1..."
    )


    query_df = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )


    query_df = (
        query_df[
            query_df[
                "country"
            ]
            == country
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    if (
        args.max_entities > 0
        and
        len(query_df)
        > args.max_entities
    ):

        query_df = (
            query_df.iloc[
                :args.max_entities
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )


    print(
        "Queries:",
        f"{len(query_df):,}",
    )


    # ========================================================
    # BLOCKER
    # ========================================================

    blocker = HybridBlocker(
        INDEX_DIR,
        source,
        country,
    )


    blocker.records.set_index(
        "entity_id",
        inplace=True,
        drop=False,
    )


    num_batches = (
        len(query_df)
        + args.batch_size
        - 1
    ) // args.batch_size


    print(
        "Batches:",
        num_batches,
    )


    # ========================================================
    # BATCHES
    # ========================================================

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


        candidate_path = (
            output_dir
            / (
                f"candidates_"
                f"{batch_no:05d}.parquet"
            )
        )


        match_path = (
            output_dir
            / (
                f"matches_"
                f"{batch_no:05d}.parquet"
            )
        )


        stats_path = (
            output_dir
            / (
                f"stats_"
                f"{batch_no:05d}.json"
            )
        )


        # Resume.
        if (
            candidate_path.exists()
            and
            match_path.exists()
        ):

            print(
                f"[{batch_no + 1}/{num_batches}] "
                f"exists → skip"
            )

            continue


        started = time.perf_counter()


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
            "=" * 70
        )


        # ====================================================
        # RETRIEVE
        # ====================================================

        retrieval_started = (
            time.perf_counter()
        )


        candidate_results = (
            blocker.generate(
                batch_df
            )
        )


        retrieval_seconds = (
            time.perf_counter()
            - retrieval_started
        )


        # ====================================================
        # LOAD ONLY TARGET RECORDS ACTUALLY NEEDED
        # ====================================================

        needed_ids = set()


        for candidates in (
            candidate_results
        ):

            needed_ids.update(
                candidates.keys()
            )


        prepared_targets = {}


        if needed_ids:

            target_subset = (
                blocker.records.loc[
                    list(
                        needed_ids
                    )
                ]
            )


            for target in (
                target_subset
                .itertuples(
                    index=False
                )
            ):

                prepared_targets[
                    target.entity_id
                ] = prepare_record(
                    target.name_norm,
                    target.address_norm,
                )


        # ====================================================
        # FEATURES
        # ====================================================

        feature_rows = []

        pair_metadata = []


        feature_started = (
            time.perf_counter()
        )


        for q_idx, query in enumerate(
            batch_df.itertuples(
                index=False
            )
        ):

            left = prepare_record(
                query.business_name,
                query.business_address,
            )


            for (
                candidate_id,
                provenance,
            ) in candidate_results[
                q_idx
            ].items():

                right = (
                    prepared_targets[
                        candidate_id
                    ]
                )


                features = (
                    compute_pair_features(
                        left,
                        right,
                        provenance,
                    )
                )


                feature_rows.append(
                    [
                        features[col]
                        for col in
                        MODEL_FEATURE_COLUMNS
                    ]
                )


                pair_metadata.append(
                    (
                        query.entity_id,
                        candidate_id,
                    )
                )


        feature_seconds = (
            time.perf_counter()
            - feature_started
        )


        # ====================================================
        # SCORE
        # ====================================================

        if feature_rows:

            X = np.asarray(
                feature_rows,
                dtype=np.float32,
            )


            probabilities = (
                model.predict(
                    X
                )
            )

        else:

            probabilities = np.asarray(
                [],
                dtype=float,
            )


        # ====================================================
        # WRITE SOURCE-LEVEL BATCH PARTITIONS
        # ====================================================

        candidate_map = {
            qid: []
            for qid in batch_df[
                "entity_id"
            ]
        }


        match_map = {
            qid: []
            for qid in batch_df[
                "entity_id"
            ]
        }


        for (
            (
                qid,
                candidate_id,
            ),
            probability,
        ) in zip(
            pair_metadata,
            probabilities,
        ):

            candidate_map[
                qid
            ].append(
                candidate_id
            )


            if probability >= threshold:

                match_map[
                    qid
                ].append(
                    candidate_id
                )


        candidate_output = pd.DataFrame(
            {
                "source1_entity_id":
                    batch_df[
                        "entity_id"
                    ],

                "candidate_entity_ids":
                    [
                        ",".join(
                            candidate_map[
                                qid
                            ]
                        )
                        for qid in
                        batch_df[
                            "entity_id"
                        ]
                    ],
            }
        )


        match_output = pd.DataFrame(
            {
                "source1_entity_id":
                    batch_df[
                        "entity_id"
                    ],

                "matched_entity_ids":
                    [
                        ",".join(
                            match_map[
                                qid
                            ]
                        )
                        for qid in
                        batch_df[
                            "entity_id"
                        ]
                    ],
            }
        )


        candidate_output.to_parquet(
            candidate_path,
            index=False,
            compression="zstd",
        )


        match_output.to_parquet(
            match_path,
            index=False,
            compression="zstd",
        )


        elapsed = (
            time.perf_counter()
            - started
        )


        stats = {
            "source":
                source,

            "country":
                country,

            "batch":
                batch_no,

            "queries":
                len(batch_df),

            "candidate_pairs":
                len(pair_metadata),

            "predicted_matches":
                int(
                    np.sum(
                        probabilities
                        >= threshold
                    )
                )
                if len(
                    probabilities
                )
                else 0,

            "retrieval_seconds":
                retrieval_seconds,

            "feature_seconds":
                feature_seconds,

            "total_seconds":
                elapsed,
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
            "Candidate pairs:",
            f"{len(pair_metadata):,}",
        )

        print(
            "Predicted matches:",
            stats[
                "predicted_matches"
            ],
        )

        print(
            "Retrieval:",
            f"{retrieval_seconds:.2f}s",
        )

        print(
            "Features:",
            f"{feature_seconds:.2f}s",
        )

        print(
            "Total:",
            f"{elapsed:.2f}s",
        )


        del feature_rows
        del pair_metadata
        del prepared_targets
        del candidate_results

        gc.collect()


if __name__ == "__main__":
    main()