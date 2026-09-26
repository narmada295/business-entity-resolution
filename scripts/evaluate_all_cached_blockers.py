import pickle
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "data" / "train"
OUTPUT_DIR = ROOT / "outputs"

GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"

BASE_CACHE = (
    OUTPUT_DIR
    / "s2_india_sample_candidates.pkl"
)

STRUCTURED_CACHE = (
    OUTPUT_DIR
    / "s2_india_structured_candidates.pkl"
)

TRANSLIT_CACHE = (
    OUTPUT_DIR
    / "s2_india_translit_name_candidates.pkl"
)

MISS_OUTPUT = (
    OUTPUT_DIR
    / "final_cached_blocking_misses.tsv"
)


def parse_matches(value):
    if not value:
        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_truth(query_ids):
    query_set = set(query_ids)

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    truth = defaultdict(set)

    for row in gt.itertuples(index=False):
        s1_id = row.source1_entity_id

        if s1_id not in query_set:
            continue

        for target_id in parse_matches(
            row.matched_entity_ids
        ):
            if target_id.startswith("S2-"):
                truth[s1_id].add(target_id)

    return truth


def evaluate(
    label,
    query_ids,
    candidates,
    truth,
):
    total_true = sum(
        len(x)
        for x in truth.values()
    )

    recovered = 0
    full = 0
    partial = 0
    zero = 0

    total_candidates = 0

    entities_with_truth = len(truth)

    for i, s1_id in enumerate(query_ids):
        cand = candidates[i]

        total_candidates += len(cand)

        true = truth.get(
            s1_id,
            set(),
        )

        if not true:
            continue

        hit = true & cand

        recovered += len(hit)

        if len(hit) == len(true):
            full += 1

        elif len(hit) == 0:
            zero += 1

        else:
            partial += 1

    return {
        "label": label,
        "recall": recovered / total_true,
        "recovered": recovered,
        "avg_candidates":
            total_candidates / len(query_ids),
        "full":
            full / entities_with_truth,
        "zero":
            zero / entities_with_truth,
    }


def union_lists(*lists):
    result = []

    for i in range(len(lists[0])):
        s = set()

        for lst in lists:
            s.update(lst[i])

        result.append(s)

    return result


def main():
    print("Loading caches...")

    base = load_pickle(BASE_CACHE)
    structured_cache = load_pickle(
        STRUCTURED_CACHE
    )
    translit_cache = load_pickle(
        TRANSLIT_CACHE
    )

    query_ids = base["query_ids"]

    if structured_cache["query_ids"] != query_ids:
        raise ValueError(
            "Structured cache query mismatch"
        )

    if translit_cache["query_ids"] != query_ids:
        raise ValueError(
            "Translit cache query mismatch"
        )

    name = []
    address = []

    for s1_id in query_ids:
        name.append(
            set(
                base["candidates"][
                    s1_id
                ]["name"]
            )
        )

        address.append(
            set(
                base["candidates"][
                    s1_id
                ]["address"]
            )
        )

    structured = [
        set(x)
        for x in structured_cache[
            "candidates"
        ]
    ]

    translit = [
        set(x)
        for x in translit_cache[
            "candidates"
        ]
    ]

    print(
        f"Queries: {len(query_ids):,}"
    )

    truth = load_truth(
        query_ids
    )

    total_true = sum(
        len(x)
        for x in truth.values()
    )

    print(
        f"True S2 links: {total_true:,}"
    )

    # ------------------------------------------------
    # Unions
    # ------------------------------------------------

    name_address = union_lists(
        name,
        address,
    )

    current_best = union_lists(
        name,
        address,
        structured,
    )

    final_union = union_lists(
        name,
        address,
        structured,
        translit,
    )

    methods = [
        (
            "Name + Address",
            name_address,
        ),
        (
            "Name + Address + Structured",
            current_best,
        ),
        (
            "Name + Address + Structured + Translit",
            final_union,
        ),
    ]

    results = []

    for label, candidates in methods:
        results.append(
            evaluate(
                label,
                query_ids,
                candidates,
                truth,
            )
        )

    print()
    print("=" * 105)
    print("ALL CACHED BLOCKERS")
    print("=" * 105)

    print(
        f"{'Method':<48}"
        f"{'Recall':>10}"
        f"{'Avg Cand':>12}"
        f"{'Full Entity':>15}"
        f"{'Zero Recovery':>16}"
    )

    print("-" * 105)

    for x in results:
        print(
            f"{x['label']:<48}"
            f"{x['recall']:>9.2%}"
            f"{x['avg_candidates']:>12.2f}"
            f"{x['full']:>14.2%}"
            f"{x['zero']:>15.2%}"
        )

    # ------------------------------------------------
    # Contribution of transliteration beyond current best
    # ------------------------------------------------

    old = results[1]["recovered"]
    new = results[2]["recovered"]

    print()
    print(
        "Additional true links from transliteration "
        "beyond structured union:",
        f"{new - old:,}",
    )

    # ------------------------------------------------
    # Final misses
    # ------------------------------------------------

    missed_rows = []

    for i, s1_id in enumerate(query_ids):
        candidates = final_union[i]

        for target_id in truth.get(
            s1_id,
            set(),
        ):
            if target_id not in candidates:
                missed_rows.append(
                    {
                        "source1_entity_id":
                            s1_id,
                        "source2_entity_id":
                            target_id,
                    }
                )

    miss_df = pd.DataFrame(
        missed_rows
    )

    miss_df.to_csv(
        MISS_OUTPUT,
        sep="\t",
        index=False,
    )

    print(
        f"\nFinal missed true links: "
        f"{len(miss_df):,}"
    )

    print(
        f"Saved to:\n{MISS_OUTPUT}"
    )


if __name__ == "__main__":
    main()