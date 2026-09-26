import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from src.blocking import (
    build_exact_block_indexes,
    exact_candidates_for_record,
)


TRAIN_DIR = ROOT / "data" / "train"
OUTPUT_DIR = ROOT / "outputs"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
S2_PATH = TRAIN_DIR / "train_source2.tsv"
S3_PATH = TRAIN_DIR / "train_source3.tsv"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"

SPLIT_PATH = OUTPUT_DIR / "splits.tsv"


CHUNK_SIZE = 50_000


def parse_matches(value):
    if not value:
        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


def load_validation_ids():
    print("Loading split information...")

    splits = pd.read_csv(
        SPLIT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "source1_entity_id",
            "split",
        ],
    )

    val_ids = set(
        splits.loc[
            splits["split"] == "val",
            "source1_entity_id",
        ]
    )

    print(
        f"Validation S1 entities: "
        f"{len(val_ids):,}"
    )

    return val_ids


def load_validation_s1(val_ids):
    """
    Stream S1 rather than loading all 2.2M rows.
    """

    print("\nLoading validation S1 records...")

    pieces = []

    reader = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    )

    for chunk in reader:

        filtered = chunk[
            chunk["entity_id"].isin(val_ids)
        ]

        if not filtered.empty:
            pieces.append(filtered)

    validation_s1 = pd.concat(
        pieces,
        ignore_index=True,
    )

    print(
        f"Loaded validation records: "
        f"{len(validation_s1):,}"
    )

    return validation_s1


def load_validation_ground_truth(val_ids):
    print("\nLoading validation ground truth...")

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    gt = gt[
        gt["source1_entity_id"].isin(
            val_ids
        )
    ].copy()

    # target_id -> true S1 id
    #
    # We expect each target to belong to one S1 entity.
    true_target_to_s1 = {}

    true_links_per_s1 = defaultdict(int)

    s2_true_links = 0
    s3_true_links = 0

    for row in gt.itertuples(index=False):

        matches = parse_matches(
            row.matched_entity_ids
        )

        for target_id in matches:

            true_target_to_s1[
                target_id
            ] = row.source1_entity_id

            true_links_per_s1[
                row.source1_entity_id
            ] += 1

            if target_id.startswith("S2-"):
                s2_true_links += 1

            elif target_id.startswith("S3-"):
                s3_true_links += 1

    print(
        f"Validation true links: "
        f"{len(true_target_to_s1):,}"
    )

    print(
        f"  S2 true links: "
        f"{s2_true_links:,}"
    )

    print(
        f"  S3 true links: "
        f"{s3_true_links:,}"
    )

    return (
        true_target_to_s1,
        true_links_per_s1,
        s2_true_links,
        s3_true_links,
    )


def process_secondary_source(
    filepath,
    source_name,
    name_index,
    address_index,
    true_target_to_s1,
    candidate_counts,
    recovered_per_s1,
):
    """
    Stream an entire secondary source.

    For each S2/S3 record:
      1. retrieve exact-name/address candidate S1s
      2. count generated candidate pairs
      3. check whether the true S1 was recovered
    """

    print(
        f"\nProcessing {source_name}..."
    )

    total_records = 0

    true_links_seen = 0
    true_links_recovered = 0

    total_candidate_pairs = 0

    reader = pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    )

    for chunk in reader:

        for row in chunk.itertuples(index=False):

            candidates = exact_candidates_for_record(
                country=row.country,
                business_name=row.business_name,
                business_address=row.business_address,
                name_index=name_index,
                address_index=address_index,
            )

            # Each target record is processed once,
            # so there is no duplicate-pair problem
            # across chunks.
            for s1_id in candidates:

                candidate_counts[s1_id] += 1

                total_candidate_pairs += 1

            true_s1 = true_target_to_s1.get(
                row.entity_id
            )

            if true_s1 is not None:

                true_links_seen += 1

                if true_s1 in candidates:

                    true_links_recovered += 1

                    recovered_per_s1[
                        true_s1
                    ] += 1

        total_records += len(chunk)

        print(
            f"\r{source_name} records: "
            f"{total_records:,}"
            f" | candidates: "
            f"{total_candidate_pairs:,}"
            f" | recovered: "
            f"{true_links_recovered:,}"
            f"/{true_links_seen:,}",
            end="",
            flush=True,
        )

    print()

    recall = (
        true_links_recovered
        / true_links_seen
        if true_links_seen
        else 0
    )

    print(
        f"\n{source_name} blocking recall: "
        f"{recall:.4%}"
    )

    print(
        f"{source_name} candidate pairs: "
        f"{total_candidate_pairs:,}"
    )

    return {
        "true_links": true_links_seen,
        "recovered_links": true_links_recovered,
        "candidate_pairs": total_candidate_pairs,
    }


def percentile(values, p):
    return np.percentile(
        values,
        p
    )


def main():

    # ---------------------------------------------------------
    # Validation entities
    # ---------------------------------------------------------

    val_ids = load_validation_ids()

    validation_s1 = load_validation_s1(
        val_ids
    )

    # ---------------------------------------------------------
    # Ground truth
    # ---------------------------------------------------------

    (
        true_target_to_s1,
        true_links_per_s1,
        s2_true_links,
        s3_true_links,
    ) = load_validation_ground_truth(
        val_ids
    )

    # ---------------------------------------------------------
    # Build exact indexes
    # ---------------------------------------------------------

    print("\nBuilding exact blocking indexes...")

    (
        name_index,
        address_index,
    ) = build_exact_block_indexes(
        validation_s1
    )

    print(
        f"Unique name blocks: "
        f"{len(name_index):,}"
    )

    print(
        f"Unique address blocks: "
        f"{len(address_index):,}"
    )

    # We no longer need the dataframe itself.
    del validation_s1

    # ---------------------------------------------------------
    # Candidate counters
    # ---------------------------------------------------------

    candidate_counts = defaultdict(int)

    recovered_per_s1 = defaultdict(int)

    # ---------------------------------------------------------
    # S2
    # ---------------------------------------------------------

    s2_results = process_secondary_source(
        filepath=S2_PATH,
        source_name="S2",
        name_index=name_index,
        address_index=address_index,
        true_target_to_s1=true_target_to_s1,
        candidate_counts=candidate_counts,
        recovered_per_s1=recovered_per_s1,
    )

    # ---------------------------------------------------------
    # S3
    # ---------------------------------------------------------

    s3_results = process_secondary_source(
        filepath=S3_PATH,
        source_name="S3",
        name_index=name_index,
        address_index=address_index,
        true_target_to_s1=true_target_to_s1,
        candidate_counts=candidate_counts,
        recovered_per_s1=recovered_per_s1,
    )

    # ---------------------------------------------------------
    # Overall pair recall
    # ---------------------------------------------------------

    total_true = (
        s2_results["true_links"]
        +
        s3_results["true_links"]
    )

    total_recovered = (
        s2_results["recovered_links"]
        +
        s3_results["recovered_links"]
    )

    total_candidates = (
        s2_results["candidate_pairs"]
        +
        s3_results["candidate_pairs"]
    )

    overall_recall = (
        total_recovered / total_true
    )

    # ---------------------------------------------------------
    # Candidate-count statistics
    # ---------------------------------------------------------

    counts = np.array(
        [
            candidate_counts[s1_id]
            for s1_id in val_ids
        ]
    )

    # ---------------------------------------------------------
    # Entity-level blocking completeness
    # ---------------------------------------------------------

    entities_with_truth = 0
    entities_fully_recovered = 0

    entities_partially_recovered = 0
    entities_zero_recovered = 0

    for s1_id in val_ids:

        total = true_links_per_s1.get(
            s1_id,
            0
        )

        # Singleton entities have no true links,
        # so don't include them in this measure.
        if total == 0:
            continue

        entities_with_truth += 1

        recovered = recovered_per_s1.get(
            s1_id,
            0
        )

        if recovered == total:

            entities_fully_recovered += 1

        elif recovered == 0:

            entities_zero_recovered += 1

        else:

            entities_partially_recovered += 1

    # ---------------------------------------------------------
    # FINAL RESULTS
    # ---------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("EXACT BLOCKING BASELINE")
    print("=" * 75)

    print(
        f"Validation S1 entities: "
        f"{len(val_ids):,}"
    )

    print(
        f"True validation links: "
        f"{total_true:,}"
    )

    print(
        f"Recovered true links: "
        f"{total_recovered:,}"
    )

    print(
        f"\nPair-level blocking recall: "
        f"{overall_recall:.4%}"
    )

    print(
        f"\nCandidate pairs generated: "
        f"{total_candidates:,}"
    )

    print(
        f"Average candidates / S1: "
        f"{counts.mean():.2f}"
    )

    print(
        f"Median candidates / S1: "
        f"{percentile(counts, 50):.0f}"
    )

    print(
        f"P95 candidates / S1: "
        f"{percentile(counts, 95):.0f}"
    )

    print(
        f"P99 candidates / S1: "
        f"{percentile(counts, 99):.0f}"
    )

    print(
        f"Maximum candidates / S1: "
        f"{counts.max():,}"
    )

    zero_candidates = (
        counts == 0
    ).sum()

    print(
        f"\nS1 entities with zero candidates: "
        f"{zero_candidates:,} "
        f"({zero_candidates / len(counts):.2%})"
    )

    print("\nEntity recovery:")

    print(
        f"Entities with ≥1 true match: "
        f"{entities_with_truth:,}"
    )

    print(
        f"All true links recovered: "
        f"{entities_fully_recovered:,} "
        f"("
        f"{entities_fully_recovered / entities_with_truth:.2%}"
        f")"
    )

    print(
        f"Some true links recovered: "
        f"{entities_partially_recovered:,} "
        f"("
        f"{entities_partially_recovered / entities_with_truth:.2%}"
        f")"
    )

    print(
        f"No true links recovered: "
        f"{entities_zero_recovered:,} "
        f"("
        f"{entities_zero_recovered / entities_with_truth:.2%}"
        f")"
    )


if __name__ == "__main__":
    main()