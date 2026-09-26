import gc
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "data" / "train"
OUTPUT_DIR = ROOT / "outputs"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
S2_PATH = TRAIN_DIR / "train_source2.tsv"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"
SPLIT_PATH = OUTPUT_DIR / "splits.tsv"

BASE_CACHE = (
    OUTPUT_DIR
    / "s2_india_sample_candidates.pkl"
)

RERANK_CACHE = (
    OUTPUT_DIR
    / "s2_india_structured_reranked_candidates.pkl"
)

EMBED_CACHE = (
    OUTPUT_DIR
    / "s2_india_embedding_name_candidates.pkl"
)


COUNTRY = "India"

SAMPLE_SIZE = 20_000
RANDOM_SEED = 42

MODEL_NAME = (
    "intfloat/multilingual-e5-small"
)

K_VALUES = [10, 20]
MAX_K = max(K_VALUES)

READ_CHUNK_SIZE = 100_000
ENCODE_BATCH_SIZE = 256


def now():
    return time.perf_counter()


def parse_matches(value):
    if not value:
        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


def contains_non_ascii(text):
    return any(
        ord(ch) > 127
        for ch in str(text)
    )


# ============================================================
# SAME SAMPLE
# ============================================================

def load_validation_ids():

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

    return set(
        splits.loc[
            splits["split"] == "val",
            "source1_entity_id",
        ]
    )


def load_query_sample(val_ids):

    print("Loading same 20k query sample...")

    pieces = []

    reader = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=READ_CHUNK_SIZE,
        usecols=[
            "entity_id",
            "business_name",
            "country",
        ],
    )

    for chunk in reader:

        filtered = chunk[
            (chunk["country"] == COUNTRY)
            &
            (chunk["entity_id"].isin(val_ids))
        ]

        if not filtered.empty:

            pieces.append(
                filtered[
                    [
                        "entity_id",
                        "business_name",
                    ]
                ].copy()
            )

    df = pd.concat(
        pieces,
        ignore_index=True,
    )

    df = df.sample(
        n=min(
            SAMPLE_SIZE,
            len(df),
        ),
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    print(
        f"Queries: {len(df):,}"
    )

    return df


# ============================================================
# TRUTH
# ============================================================

def load_truth(query_ids):

    query_ids = set(query_ids)

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    truth = defaultdict(set)

    for row in gt.itertuples(index=False):

        s1_id = row.source1_entity_id

        if s1_id not in query_ids:
            continue

        for target_id in parse_matches(
            row.matched_entity_ids
        ):

            if target_id.startswith("S2-"):
                truth[s1_id].add(
                    target_id
                )

    total = sum(
        len(v)
        for v in truth.values()
    )

    print(
        f"True S2 links: {total:,}"
    )

    return truth, total


# ============================================================
# LOAD NON-ASCII CORPUS
# ============================================================

def load_non_ascii_s2():

    print(
        "\nLoading non-ASCII S2 India names..."
    )

    ids = []
    names = []

    total_read = 0
    india_read = 0

    t0 = now()

    reader = pd.read_csv(
        S2_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=READ_CHUNK_SIZE,
        usecols=[
            "entity_id",
            "business_name",
            "country",
        ],
    )

    for chunk_no, chunk in enumerate(
        reader,
        start=1,
    ):

        total_read += len(chunk)

        chunk = chunk[
            chunk["country"] == COUNTRY
        ]

        india_read += len(chunk)

        mask = (
            chunk["business_name"]
            .map(contains_non_ascii)
        )

        chunk = chunk[mask]

        if not chunk.empty:

            ids.extend(
                chunk["entity_id"]
                .tolist()
            )

            names.extend(
                chunk["business_name"]
                .tolist()
            )

        print(
            f"\r"
            f"chunks={chunk_no:<3}"
            f" read={total_read:,}"
            f" India={india_read:,}"
            f" nonASCII={len(ids):,}"
            f" elapsed={now()-t0:.1f}s",
            end="",
            flush=True,
        )

    print()

    print(
        f"Non-ASCII S2 names: "
        f"{len(ids):,}"
    )

    return ids, names


# ============================================================
# ENCODE
# ============================================================

def encode_names(
    model,
    names,
    label,
):

    print(
        f"\nEncoding {label}: "
        f"{len(names):,}"
    )

    t0 = now()

    embeddings = model.encode(
        names,
        batch_size=ENCODE_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    print(
        f"{label} embeddings: "
        f"{embeddings.shape}"
    )

    print(
        f"{label} encode time: "
        f"{now() - t0:.2f}s"
    )

    return embeddings


# ============================================================
# FAISS HNSW
# ============================================================

def build_index(
    corpus_embeddings,
):

    dim = corpus_embeddings.shape[1]

    print(
        "\nBuilding FAISS HNSW index..."
    )

    t0 = now()

    index = faiss.IndexHNSWFlat(
        dim,
        32,
        faiss.METRIC_INNER_PRODUCT,
    )

    index.hnsw.efConstruction = 80
    index.hnsw.efSearch = 128

    index.add(
        corpus_embeddings
    )

    print(
        f"Indexed vectors: "
        f"{index.ntotal:,}"
    )

    print(
        f"Index build time: "
        f"{now() - t0:.2f}s"
    )

    return index


# ============================================================
# RETRIEVE
# ============================================================

def retrieve(
    index,
    query_embeddings,
    corpus_ids,
):

    print(
        f"\nRetrieving top {MAX_K}..."
    )

    t0 = now()

    scores, indices = index.search(
        query_embeddings,
        MAX_K,
    )

    results = []

    for row in indices:

        candidates = []

        for idx in row:

            if idx < 0:
                continue

            candidates.append(
                corpus_ids[idx]
            )

        results.append(
            candidates
        )

    print(
        f"Retrieval time: "
        f"{now() - t0:.2f}s"
    )

    return results


# ============================================================
# LOAD CURRENT BLOCKER
# ============================================================

def load_current_candidates(
    query_df,
):

    query_ids = (
        query_df["entity_id"]
        .tolist()
    )

    with open(
        BASE_CACHE,
        "rb",
    ) as f:

        base = pickle.load(f)

    with open(
        RERANK_CACHE,
        "rb",
    ) as f:

        rerank = pickle.load(f)

    if (
        base["query_ids"]
        != query_ids
    ):

        raise ValueError(
            "Base cache mismatch"
        )

    if (
        rerank["query_ids"]
        != query_ids
    ):

        raise ValueError(
            "Rerank cache mismatch"
        )

    current = []

    for i, s1_id in enumerate(
        query_ids
    ):

        combined = set()

        combined.update(
            base["candidates"][
                s1_id
            ]["name"]
        )

        combined.update(
            base["candidates"][
                s1_id
            ]["address"]
        )

        combined.update(
            rerank[
                "candidates"
            ][i]
        )

        current.append(
            combined
        )

    return current


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    label,
    query_ids,
    candidate_sets,
    truth,
    total_true,
):

    recovered = 0

    full = 0
    zero = 0

    total_candidates = 0

    entities_with_truth = len(
        truth
    )

    for i, s1_id in enumerate(
        query_ids
    ):

        candidates = (
            candidate_sets[i]
        )

        total_candidates += len(
            candidates
        )

        true = truth.get(
            s1_id,
            set(),
        )

        if not true:
            continue

        hit = (
            true & candidates
        )

        recovered += len(
            hit
        )

        if len(hit) == len(true):
            full += 1

        elif len(hit) == 0:
            zero += 1

    return {
        "label": label,

        "recall":
            recovered
            / total_true,

        "avg_candidates":
            total_candidates
            / len(query_ids),

        "full":
            full
            / entities_with_truth,

        "zero":
            zero
            / entities_with_truth,

        "recovered":
            recovered,
    }


def main():

    total_start = now()

    val_ids = load_validation_ids()

    query_df = load_query_sample(
        val_ids
    )

    query_ids = (
        query_df["entity_id"]
        .tolist()
    )

    truth, total_true = load_truth(
        query_ids
    )

    # ------------------------------------------------
    # Model
    # ------------------------------------------------

    print(
        f"\nLoading model:\n"
        f"{MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    # ------------------------------------------------
    # Corpus
    # ------------------------------------------------

    corpus_ids, corpus_names = (
        load_non_ascii_s2()
    )

    # ------------------------------------------------
    # Embeddings
    # ------------------------------------------------

    query_embeddings = (
        encode_names(
            model,
            query_df[
                "business_name"
            ].tolist(),
            "queries",
        )
    )

    corpus_embeddings = (
        encode_names(
            model,
            corpus_names,
            "corpus",
        )
    )

    # We don't need raw names anymore.
    del corpus_names
    gc.collect()

    # ------------------------------------------------
    # ANN index
    # ------------------------------------------------

    index = build_index(
        corpus_embeddings
    )

    # HNSW stores its own copy.
    del corpus_embeddings
    gc.collect()

    embedding_candidates = retrieve(
        index,
        query_embeddings,
        corpus_ids,
    )

    # ------------------------------------------------
    # Save
    # ------------------------------------------------

    print(
        f"\nSaving embedding candidates:\n"
        f"{EMBED_CACHE}"
    )

    with open(
        EMBED_CACHE,
        "wb",
    ) as f:

        pickle.dump(
            {
                "query_ids":
                    query_ids,

                "candidates":
                    embedding_candidates,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # ------------------------------------------------
    # Existing best blocker
    # ------------------------------------------------

    current = (
        load_current_candidates(
            query_df
        )
    )

    methods = []

    for k in K_VALUES:

        embedding_k = [
            set(row[:k])
            for row
            in embedding_candidates
        ]

        union_k = []

        for i in range(
            len(query_ids)
        ):

            x = set(
                current[i]
            )

            x.update(
                embedding_k[i]
            )

            union_k.append(x)

        methods.append(
            (
                f"Embedding K{k}",
                embedding_k,
            )
        )

        methods.append(
            (
                f"Current + Embedding K{k}",
                union_k,
            )
        )

    results = []

    for label, candidates in methods:

        results.append(
            evaluate(
                label,
                query_ids,
                candidates,
                truth,
                total_true,
            )
        )

    print()
    print("=" * 110)
    print("MULTILINGUAL EMBEDDING NAME BLOCKER")
    print("=" * 110)

    print(
        f"{'Method':<42}"
        f"{'Recall':>10}"
        f"{'Avg Cand':>12}"
        f"{'Full Entity':>15}"
        f"{'Zero Recovery':>16}"
    )

    print("-" * 110)

    for x in results:

        print(
            f"{x['label']:<42}"
            f"{x['recall']:>9.2%}"
            f"{x['avg_candidates']:>12.2f}"
            f"{x['full']:>14.2%}"
            f"{x['zero']:>15.2%}"
        )

    print(
        f"\nTotal runtime: "
        f"{now() - total_start:.2f}s"
    )


if __name__ == "__main__":
    main()