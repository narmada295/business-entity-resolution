from pathlib import Path
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from src.indexed_blocking import (
    IndexedCharBlocker,
)


TRAIN_DIR = ROOT / "data" / "train"
INDEX_DIR = ROOT / "outputs" / "indexes"


S1_PATH = (
    TRAIN_DIR
    / "train_source1.tsv"
)


QUERY_LIMIT = 5000


if len(sys.argv) != 3:

    print(
        "Usage:\n"
        "python scripts/test_indexed_blocking.py S2 US\n"
        "python scripts/test_indexed_blocking.py S3 India"
    )

    sys.exit(1)


SOURCE = sys.argv[1].upper()
COUNTRY = sys.argv[2]


# ============================================================
# LOAD QUERIES
# ============================================================

print(
    "Loading S1..."
)


s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)


query_df = (
    s1[
        s1["country"]
        == COUNTRY
    ]
    .head(
        QUERY_LIMIT
    )
    .copy()
    .reset_index(
        drop=True
    )
)


print(
    "Queries:",
    len(query_df),
)


# ============================================================
# LOAD INDEX
# ============================================================

start = time.perf_counter()


blocker = IndexedCharBlocker(
    INDEX_DIR,
    SOURCE,
    COUNTRY,
)


load_time = (
    time.perf_counter()
    - start
)


print(
    f"Index load time: "
    f"{load_time:.2f}s"
)


# ============================================================
# QUERY
# ============================================================

start = time.perf_counter()


results = blocker.generate(
    query_df
)


query_time = (
    time.perf_counter()
    - start
)


counts = [
    len(x)
    for x in results
]


print()

print(
    "Average candidates:",
    sum(counts)
    / len(counts),
)

print(
    "Min candidates:",
    min(counts),
)

print(
    "Max candidates:",
    max(counts),
)

print(
    "Zero candidate queries:",
    sum(
        x == 0
        for x in counts
    ),
)


print()

print(
    f"Query runtime: "
    f"{query_time:.2f}s"
)

print(
    f"Queries/sec: "
    f"{len(query_df) / query_time:.2f}"
)