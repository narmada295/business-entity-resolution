import sys
import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.normalize import normalize_name, normalize_address


TRAIN_DIR = ROOT / "data" / "train"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
S2_PATH = TRAIN_DIR / "train_source2.tsv"
S3_PATH = TRAIN_DIR / "train_source3.tsv"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DB_PATH = OUTPUT_DIR / "normalization_analysis.sqlite"


# Keep chunks modest because pandas string columns have
# significant Python-object overhead.
CHUNK_SIZE = 50_000

# SQLite insert batch size.
INSERT_BATCH_SIZE = 20_000


def parse_matches(value):
    if not value:
        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


def connect_database():
    if DB_PATH.exists():
        print(f"Removing old database: {DB_PATH}")
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)

    # These settings favour speed while remaining safe enough for
    # a temporary analysis database that we can always regenerate.
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA temp_store = FILE")
    conn.execute("PRAGMA cache_size = -200000")  # ~200 MB cache

    return conn


def create_tables(conn):
    conn.executescript(
        """
        CREATE TABLE s1 (
            entity_id TEXT PRIMARY KEY,
            raw_name TEXT NOT NULL,
            raw_address TEXT NOT NULL,
            name_norm TEXT NOT NULL,
            address_norm TEXT NOT NULL,
            country TEXT NOT NULL
        );

        CREATE TABLE positives (
            s1_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            source INTEGER NOT NULL
        );

        CREATE TABLE targets (
            entity_id TEXT PRIMARY KEY,
            raw_name TEXT NOT NULL,
            raw_address TEXT NOT NULL,
            name_norm TEXT NOT NULL,
            address_norm TEXT NOT NULL,
            country TEXT NOT NULL
        );
        """
    )

    conn.commit()


def insert_s1(conn):
    print("\nLoading + normalizing Source 1...")

    total = 0

    reader = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    )

    query = """
        INSERT INTO s1 (
            entity_id,
            raw_name,
            raw_address,
            name_norm,
            address_norm,
            country
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """

    for chunk_number, chunk in enumerate(reader, start=1):

        rows = []

        for row in chunk.itertuples(index=False):
            rows.append(
                (
                    row.entity_id,
                    row.business_name,
                    row.business_address,
                    normalize_name(row.business_name),
                    normalize_address(row.business_address),
                    row.country,
                )
            )

        conn.executemany(query, rows)
        conn.commit()

        total += len(rows)

        print(
            f"\rS1 processed: {total:,}",
            end="",
            flush=True,
        )

    print()


def insert_ground_truth(conn):
    print("\nExpanding ground-truth links...")

    total_entities = 0
    total_pairs = 0

    reader = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    )

    query = """
        INSERT INTO positives (
            s1_id,
            target_id,
            source
        )
        VALUES (?, ?, ?)
    """

    batch = []

    for chunk in reader:

        for row in chunk.itertuples(index=False):

            total_entities += 1

            matches = parse_matches(
                row.matched_entity_ids
            )

            for target_id in matches:

                if target_id.startswith("S2-"):
                    source = 2

                elif target_id.startswith("S3-"):
                    source = 3

                else:
                    raise ValueError(
                        f"Unexpected entity ID: {target_id}"
                    )

                batch.append(
                    (
                        row.source1_entity_id,
                        target_id,
                        source,
                    )
                )

                total_pairs += 1

                if len(batch) >= INSERT_BATCH_SIZE:
                    conn.executemany(
                        query,
                        batch,
                    )

                    batch.clear()

        conn.commit()

        print(
            f"\rGT entities: {total_entities:,}"
            f" | positive pairs: {total_pairs:,}",
            end="",
            flush=True,
        )

    if batch:
        conn.executemany(
            query,
            batch,
        )

    conn.commit()

    print()


def create_positive_indexes(conn):
    print("\nCreating ground-truth indexes...")

    conn.executescript(
        """
        CREATE INDEX idx_positives_source_target
        ON positives(source, target_id);

        CREATE INDEX idx_positives_s1
        ON positives(s1_id);
        """
    )

    conn.commit()


def clear_targets(conn):
    conn.execute("DELETE FROM targets")
    conn.commit()


def insert_secondary_source(
    conn,
    filepath,
    source_name,
):
    print(
        f"\nLoading + normalizing {source_name}..."
    )

    clear_targets(conn)

    total = 0

    reader = pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    )

    query = """
        INSERT INTO targets (
            entity_id,
            raw_name,
            raw_address,
            name_norm,
            address_norm,
            country
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """

    for chunk in reader:

        rows = []

        for row in chunk.itertuples(index=False):

            rows.append(
                (
                    row.entity_id,
                    row.business_name,
                    row.business_address,
                    normalize_name(row.business_name),
                    normalize_address(row.business_address),
                    row.country,
                )
            )

        conn.executemany(query, rows)

        conn.commit()

        total += len(rows)

        print(
            f"\r{source_name} processed: {total:,}",
            end="",
            flush=True,
        )

    print()


def calculate_stats(conn, source):
    print(
        f"\nCalculating positive-pair overlap for S{source}..."
    )

    query = """
        SELECT
            COUNT(*) AS total,

            SUM(
                CASE
                    WHEN a.raw_name = b.raw_name
                    THEN 1 ELSE 0
                END
            ) AS raw_name,

            SUM(
                CASE
                    WHEN a.name_norm = b.name_norm
                    THEN 1 ELSE 0
                END
            ) AS norm_name,

            SUM(
                CASE
                    WHEN a.raw_address = b.raw_address
                         AND b.raw_address != ''
                    THEN 1 ELSE 0
                END
            ) AS raw_address,

            SUM(
                CASE
                    WHEN a.address_norm = b.address_norm
                         AND b.address_norm != ''
                    THEN 1 ELSE 0
                END
            ) AS norm_address,

            SUM(
                CASE
                    WHEN
                        a.name_norm = b.name_norm
                        OR (
                            a.address_norm = b.address_norm
                            AND b.address_norm != ''
                        )
                    THEN 1 ELSE 0
                END
            ) AS name_or_address,

            SUM(
                CASE
                    WHEN
                        a.name_norm = b.name_norm
                        AND
                        a.address_norm = b.address_norm
                        AND b.address_norm != ''
                    THEN 1 ELSE 0
                END
            ) AS name_and_address,

            SUM(
                CASE
                    WHEN a.country = b.country
                    THEN 1 ELSE 0
                END
            ) AS country

        FROM positives p

        JOIN s1 a
            ON a.entity_id = p.s1_id

        JOIN targets b
            ON b.entity_id = p.target_id

        WHERE p.source = ?
    """

    row = conn.execute(
        query,
        (source,),
    ).fetchone()

    keys = [
        "total",
        "raw_name",
        "norm_name",
        "raw_address",
        "norm_address",
        "name_or_address",
        "name_and_address",
        "country",
    ]

    return dict(zip(keys, row))


def merge_stats(a, b):
    return {
        key: a[key] + b[key]
        for key in a
    }


def print_stats(title, stats):
    total = stats["total"]

    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)

    print(
        f"Positive pairs: {total:,}\n"
    )

    labels = [
        ("Raw exact name", "raw_name"),
        ("Normalized exact name", "norm_name"),
        ("Raw exact address", "raw_address"),
        (
            "Normalized exact address",
            "norm_address",
        ),
        (
            "Normalized name OR address",
            "name_or_address",
        ),
        (
            "Normalized name AND address",
            "name_and_address",
        ),
        ("Country equal", "country"),
    ]

    for label, key in labels:

        count = stats[key]

        pct = (
            100 * count / total
            if total
            else 0
        )

        print(
            f"{label:<35}"
            f"{count:>12,} "
            f"({pct:6.2f}%)"
        )


def main():

    print(
        f"Temporary analysis DB:\n{DB_PATH}"
    )

    conn = connect_database()

    try:

        create_tables(conn)

        # -----------------------------------
        # S1
        # -----------------------------------

        insert_s1(conn)

        # -----------------------------------
        # Ground truth
        # -----------------------------------

        insert_ground_truth(conn)

        create_positive_indexes(conn)

        # -----------------------------------
        # Source 2
        # -----------------------------------

        insert_secondary_source(
            conn,
            S2_PATH,
            "S2",
        )

        s2_stats = calculate_stats(
            conn,
            source=2,
        )

        print_stats(
            "SOURCE 2 POSITIVE PAIRS",
            s2_stats,
        )

        # Completely delete S2 from our
        # temporary target table before S3.
        clear_targets(conn)

        # -----------------------------------
        # Source 3
        # -----------------------------------

        insert_secondary_source(
            conn,
            S3_PATH,
            "S3",
        )

        s3_stats = calculate_stats(
            conn,
            source=3,
        )

        print_stats(
            "SOURCE 3 POSITIVE PAIRS",
            s3_stats,
        )

        # -----------------------------------
        # Combined
        # -----------------------------------

        combined = merge_stats(
            s2_stats,
            s3_stats,
        )

        print_stats(
            "ALL POSITIVE PAIRS",
            combined,
        )

    finally:
        conn.close()

    print(
        "\nAnalysis complete."
    )

    print(
        f"\nSQLite database retained at:\n"
        f"{DB_PATH}"
    )

    print(
        "\nYou can delete it after we inspect "
        "the results."
    )


if __name__ == "__main__":
    main()