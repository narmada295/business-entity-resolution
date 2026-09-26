import gc
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from rapidfuzz.fuzz import (
    ratio,
    token_sort_ratio,
    token_set_ratio,
)
from unidecode import unidecode


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT))

from src.normalize import (
    normalize_name,
    normalize_address,
)

TRAIN_DIR = ROOT / "data" / "train"
OUTPUT_DIR = ROOT / "outputs"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
S2_PATH = TRAIN_DIR / "train_source2.tsv"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"

BASE_CACHE = (
    OUTPUT_DIR
    / "s2_india_sample_candidates.pkl"
)

STRUCTURED_CACHE = (
    OUTPUT_DIR
    / "s2_india_structured_reranked_candidates.pkl"
)

EMBED_CACHE = (
    OUTPUT_DIR
    / "s2_india_embedding_name_candidates.pkl"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "matcher_features_s2_india_sample.parquet"
)


# ============================================================
# SETTINGS
# ============================================================

READ_CHUNK_SIZE = 100_000
WRITE_BUFFER_SIZE = 50_000

EMBEDDING_K = 10

NUMBER_RE = re.compile(r"\d+")


LEGAL_SUFFIXES = {
    "private",
    "limited",
    "ltd",
    "llp",
    "pvt",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
}


GENERIC_NAME_TOKENS = {
    "private",
    "limited",
    "ltd",
    "llp",
    "pvt",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
}


# ============================================================
# HELPERS
# ============================================================

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


def transliterate_name(text):

    if not text:
        return ""

    return normalize_name(
        unidecode(text)
    )


def token_set(text):

    if not text:
        return set()

    return set(
        text.split()
    )


def meaningful_tokens(text):

    if not text:
        return set()

    return {
        token
        for token in text.split()
        if token not in GENERIC_NAME_TOKENS
    }


def strip_legal_suffix_tokens(text):

    if not text:
        return ""

    tokens = text.split()

    kept = [
        token
        for token in tokens
        if token not in LEGAL_SUFFIXES
    ]

    return " ".join(
        kept
    )


def char_ngrams(
    text,
    n=3,
):

    if not text:
        return set()

    compact = text.replace(
        " ",
        "",
    )

    if not compact:
        return set()

    if len(compact) < n:
        return {
            compact
        }

    return {
        compact[i:i+n]
        for i in range(
            len(compact) - n + 1
        )
    }


def extract_numbers(text):

    if not text:
        return set()

    result = set()

    for value in NUMBER_RE.findall(
        text
    ):

        try:
            result.add(
                str(
                    int(value)
                )
            )
        except ValueError:
            result.add(
                value
            )

    return result


def ordered_numbers(text):

    if not text:
        return []

    result = []

    for value in NUMBER_RE.findall(
        text
    ):

        try:
            result.append(
                str(
                    int(value)
                )
            )
        except ValueError:
            result.append(
                value
            )

    return result


def jaccard(
    a,
    b,
):

    if not a and not b:
        return 0.0

    union = a | b

    if not union:
        return 0.0

    return (
        len(
            a & b
        )
        / len(
            union
        )
    )


def safe_containment(
    shared,
    base,
):

    if not base:
        return 0.0

    return (
        len(shared)
        / len(base)
    )


def length_ratio(
    a,
    b,
):

    la = len(a)
    lb = len(b)

    if la == 0 and lb == 0:
        return 1.0

    if la == 0 or lb == 0:
        return 0.0

    return (
        min(
            la,
            lb,
        )
        / max(
            la,
            lb,
        )
    )


# ============================================================
# LOAD CANDIDATE CACHES
# ============================================================

def load_candidate_caches():

    print(
        "Loading candidate caches..."
    )

    with open(
        BASE_CACHE,
        "rb",
    ) as f:
        base = pickle.load(f)

    with open(
        STRUCTURED_CACHE,
        "rb",
    ) as f:
        structured = pickle.load(f)

    with open(
        EMBED_CACHE,
        "rb",
    ) as f:
        embedding = pickle.load(f)

    query_ids = (
        base[
            "query_ids"
        ]
    )

    if (
        structured[
            "query_ids"
        ]
        != query_ids
    ):
        raise ValueError(
            "Structured cache query mismatch"
        )

    if (
        embedding[
            "query_ids"
        ]
        != query_ids
    ):
        raise ValueError(
            "Embedding cache query mismatch"
        )

    print(
        f"Queries in caches: "
        f"{len(query_ids):,}"
    )

    return (
        query_ids,
        base,
        structured,
        embedding,
    )


# ============================================================
# LOAD S1 RECORDS
# ============================================================

def load_s1_records(
    query_ids,
):

    print(
        "\nLoading S1 query records..."
    )

    query_set = set(
        query_ids
    )

    records = {}

    reader = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=READ_CHUNK_SIZE,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
        ],
    )

    for chunk in reader:

        chunk = chunk[
            chunk[
                "entity_id"
            ].isin(
                query_set
            )
        ]

        for row in chunk.itertuples(
            index=False
        ):

            name_norm = (
                normalize_name(
                    row.business_name
                )
            )

            addr_norm = (
                normalize_address(
                    row.business_address
                )
            )

            name_tokens = (
                token_set(
                    name_norm
                )
            )

            meaningful_name = (
                meaningful_tokens(
                    name_norm
                )
            )

            name_no_suffix = (
                strip_legal_suffix_tokens(
                    name_norm
                )
            )

            records[
                row.entity_id
            ] = {

                "business_name":
                    row.business_name,

                "business_address":
                    row.business_address,

                "name_norm":
                    name_norm,

                "name_translit":
                    transliterate_name(
                        row.business_name
                    ),

                "name_no_suffix":
                    name_no_suffix,

                "name_tokens":
                    name_tokens,

                "meaningful_name_tokens":
                    meaningful_name,

                "name_char_3grams":
                    char_ngrams(
                        name_norm,
                        3,
                    ),

                "address_norm":
                    addr_norm,

                "address_tokens":
                    token_set(
                        addr_norm
                    ),

                "numbers":
                    extract_numbers(
                        addr_norm
                    ),

                "ordered_numbers":
                    ordered_numbers(
                        addr_norm
                    ),
            }

    if (
        len(records)
        != len(query_ids)
    ):

        raise ValueError(
            f"Expected "
            f"{len(query_ids):,} "
            f"S1 records but found "
            f"{len(records):,}"
        )

    print(
        f"S1 records loaded: "
        f"{len(records):,}"
    )

    return records


# ============================================================
# GROUND TRUTH
# ============================================================

def load_truth(
    query_ids,
):

    print(
        "\nLoading ground truth..."
    )

    query_set = set(
        query_ids
    )

    truth = defaultdict(
        set
    )

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    for row in gt.itertuples(
        index=False
    ):

        s1_id = (
            row.source1_entity_id
        )

        if (
            s1_id
            not in query_set
        ):
            continue

        for target_id in parse_matches(
            row.matched_entity_ids
        ):

            if target_id.startswith(
                "S2-"
            ):
                truth[
                    s1_id
                ].add(
                    target_id
                )

    total = sum(
        len(x)
        for x in truth.values()
    )

    print(
        f"True S2 links: "
        f"{total:,}"
    )

    return truth


# ============================================================
# BUILD CANDIDATE INDEX
# ============================================================

def build_candidate_pair_index(
    query_ids,
    base,
    structured,
    embedding,
):

    print(
        "\nBuilding candidate-pair index..."
    )

    t0 = now()

    target_to_pairs = defaultdict(
        list
    )

    total_pairs = 0

    for (
        query_idx,
        s1_id,
    ) in enumerate(
        query_ids
    ):

        name_scores = (
            base[
                "candidates"
            ][
                s1_id
            ][
                "name"
            ]
        )

        address_scores = (
            base[
                "candidates"
            ][
                s1_id
            ][
                "address"
            ]
        )

        structured_set = set(
            structured[
                "candidates"
            ][
                query_idx
            ]
        )

        embedding_list = (
            embedding[
                "candidates"
            ][
                query_idx
            ][
                :EMBEDDING_K
            ]
        )

        embedding_rank = {
            entity_id: rank
            for (
                rank,
                entity_id,
            )
            in enumerate(
                embedding_list,
                start=1,
            )
        }

        candidate_ids = (
            set(
                name_scores
            )
            |
            set(
                address_scores
            )
            |
            structured_set
            |
            set(
                embedding_list
            )
        )

        for target_id in candidate_ids:

            from_name = int(
                target_id
                in name_scores
            )

            from_address = int(
                target_id
                in address_scores
            )

            from_structured = int(
                target_id
                in structured_set
            )

            from_embedding = int(
                target_id
                in embedding_rank
            )

            target_to_pairs[
                target_id
            ].append(
                (
                    s1_id,

                    from_name,
                    from_address,
                    from_structured,
                    from_embedding,

                    float(
                        name_scores.get(
                            target_id,
                            0.0,
                        )
                    ),

                    float(
                        address_scores.get(
                            target_id,
                            0.0,
                        )
                    ),

                    int(
                        embedding_rank.get(
                            target_id,
                            0,
                        )
                    ),
                )
            )

            total_pairs += 1

    print(
        f"Candidate pairs: "
        f"{total_pairs:,}"
    )

    print(
        f"Unique S2 candidate IDs: "
        f"{len(target_to_pairs):,}"
    )

    print(
        f"Index build time: "
        f"{now() - t0:.2f}s"
    )

    return (
        target_to_pairs,
        total_pairs,
    )


# ============================================================
# COMPUTE FEATURES
# ============================================================

def compute_features(
    s1_id,
    s1,
    s2_id,
    s2_name,
    s2_address,
    truth,
    provenance,
):

    (
        from_name,
        from_address,
        from_structured,
        from_embedding,
        name_retrieval_score,
        address_retrieval_score,
        embedding_rank,
    ) = provenance


    # ========================================================
    # NORMALIZE S2
    # ========================================================

    s2_name_norm = (
        normalize_name(
            s2_name
        )
    )

    s2_addr_norm = (
        normalize_address(
            s2_address
        )
    )

    s2_translit = (
        transliterate_name(
            s2_name
        )
    )


    # ========================================================
    # NAME REPRESENTATIONS
    # ========================================================

    s2_name_tokens = (
        token_set(
            s2_name_norm
        )
    )

    s2_meaningful_tokens = (
        meaningful_tokens(
            s2_name_norm
        )
    )

    s2_name_no_suffix = (
        strip_legal_suffix_tokens(
            s2_name_norm
        )
    )

    s2_name_3grams = (
        char_ngrams(
            s2_name_norm,
            3,
        )
    )


    # ========================================================
    # ADDRESS REPRESENTATIONS
    # ========================================================

    s2_addr_tokens = (
        token_set(
            s2_addr_norm
        )
    )

    s2_numbers = (
        extract_numbers(
            s2_addr_norm
        )
    )

    s2_ordered_numbers = (
        ordered_numbers(
            s2_addr_norm
        )
    )


    # ========================================================
    # NAME TOKEN OVERLAPS
    # ========================================================

    shared_name_tokens = (
        s1[
            "name_tokens"
        ]
        &
        s2_name_tokens
    )

    shared_meaningful_tokens = (
        s1[
            "meaningful_name_tokens"
        ]
        &
        s2_meaningful_tokens
    )

    extra_meaningful_s1 = (
        s1[
            "meaningful_name_tokens"
        ]
        -
        s2_meaningful_tokens
    )

    extra_meaningful_s2 = (
        s2_meaningful_tokens
        -
        s1[
            "meaningful_name_tokens"
        ]
    )


    # ========================================================
    # NAME CHAR NGRAMS
    # ========================================================

    shared_name_3grams = (
        s1[
            "name_char_3grams"
        ]
        &
        s2_name_3grams
    )

    union_name_3grams = (
        s1[
            "name_char_3grams"
        ]
        |
        s2_name_3grams
    )

    name_char_3gram_jaccard = (
        len(
            shared_name_3grams
        )
        /
        len(
            union_name_3grams
        )
        if union_name_3grams
        else 0.0
    )


    # ========================================================
    # BASIC NAME SIMILARITY
    # ========================================================

    name_ratio = (
        ratio(
            s1[
                "name_norm"
            ],
            s2_name_norm,
        )
        / 100.0
    )

    name_token_sort = (
        token_sort_ratio(
            s1[
                "name_norm"
            ],
            s2_name_norm,
        )
        / 100.0
    )

    name_token_set_score = (
        token_set_ratio(
            s1[
                "name_norm"
            ],
            s2_name_norm,
        )
        / 100.0
    )

    name_token_jaccard = (
        jaccard(
            s1[
                "name_tokens"
            ],
            s2_name_tokens,
        )
    )


    # ========================================================
    # LEGAL-SUFFIX-STRIPPED NAME
    # ========================================================

    legal_suffix_stripped_ratio = (
        ratio(
            s1[
                "name_no_suffix"
            ],
            s2_name_no_suffix,
        )
        / 100.0
    )

    legal_suffix_stripped_token_set = (
        token_set_ratio(
            s1[
                "name_no_suffix"
            ],
            s2_name_no_suffix,
        )
        / 100.0
    )


    # ========================================================
    # TRANSLITERATED NAME
    # ========================================================

    translit_ratio = (
        ratio(
            s1[
                "name_translit"
            ],
            s2_translit,
        )
        / 100.0
    )

    translit_token_sort = (
        token_sort_ratio(
            s1[
                "name_translit"
            ],
            s2_translit,
        )
        / 100.0
    )

    translit_token_set = (
        token_set_ratio(
            s1[
                "name_translit"
            ],
            s2_translit,
        )
        / 100.0
    )


    # ========================================================
    # ADDRESS SIMILARITY
    # ========================================================

    address_ratio = (
        ratio(
            s1[
                "address_norm"
            ],
            s2_addr_norm,
        )
        / 100.0
    )

    address_token_sort = (
        token_sort_ratio(
            s1[
                "address_norm"
            ],
            s2_addr_norm,
        )
        / 100.0
    )

    address_token_set_score = (
        token_set_ratio(
            s1[
                "address_norm"
            ],
            s2_addr_norm,
        )
        / 100.0
    )

    address_token_jaccard = (
        jaccard(
            s1[
                "address_tokens"
            ],
            s2_addr_tokens,
        )
    )


    # ========================================================
    # ADDRESS NUMBERS
    # ========================================================

    shared_numbers = (
        s1[
            "numbers"
        ]
        &
        s2_numbers
    )

    number_union = (
        s1[
            "numbers"
        ]
        |
        s2_numbers
    )

    number_jaccard = (
        len(
            shared_numbers
        )
        /
        len(
            number_union
        )
        if number_union
        else 0.0
    )

    both_have_numbers = (
        bool(
            s1[
                "numbers"
            ]
        )
        and
        bool(
            s2_numbers
        )
    )

    number_conflict = int(
        both_have_numbers
        and
        len(
            shared_numbers
        ) == 0
    )

    ordered_number_exact = int(
        bool(
            s1[
                "ordered_numbers"
            ]
        )
        and
        (
            s1[
                "ordered_numbers"
            ]
            ==
            s2_ordered_numbers
        )
    )

    first_number_match = 0
    first_number_conflict = 0

    if (
        s1[
            "ordered_numbers"
        ]
        and
        s2_ordered_numbers
    ):

        if (
            s1[
                "ordered_numbers"
            ][0]
            ==
            s2_ordered_numbers[0]
        ):
            first_number_match = 1
        else:
            first_number_conflict = 1


    # ========================================================
    # CONTAINMENT VALUES
    # ========================================================

    s1_name_containment = (
        safe_containment(
            shared_name_tokens,
            s1[
                "name_tokens"
            ],
        )
    )

    s2_name_containment = (
        safe_containment(
            shared_name_tokens,
            s2_name_tokens,
        )
    )

    meaningful_s1_containment = (
        safe_containment(
            shared_meaningful_tokens,
            s1[
                "meaningful_name_tokens"
            ],
        )
    )

    meaningful_s2_containment = (
        safe_containment(
            shared_meaningful_tokens,
            s2_meaningful_tokens,
        )
    )


    # ========================================================
    # INTERACTION FEATURES
    # ========================================================

    strong_address_weak_name = int(
        address_token_jaccard
        >= 0.80
        and
        name_ratio
        < 0.60
    )

    strong_name_weak_address = int(
        name_token_set_score
        >= 0.90
        and
        address_token_jaccard
        < 0.30
    )

    address_number_agreement = int(
        address_token_jaccard
        >= 0.80
        and
        number_jaccard
        >= 0.80
    )

    high_address_low_name = int(
        address_token_set_score
        >= 0.90
        and
        name_token_set_score
        < 0.60
    )

    translit_rescue = int(
        translit_ratio
        >= 0.80
        and
        name_ratio
        < 0.50
    )

    translit_address_rescue = int(
        translit_token_set
        >= 0.80
        and
        address_token_set_score
        >= 0.80
        and
        name_ratio
        < 0.60
    )

    strong_address_number_name_conflict = int(
        address_token_jaccard
        >= 0.80
        and
        number_jaccard
        >= 0.80
        and
        name_token_set_score
        < 0.50
    )

    strong_name_number_conflict = int(
        name_token_set_score
        >= 0.90
        and
        number_conflict
        == 1
    )

    address_x_number = (
        address_token_jaccard
        *
        number_jaccard
    )

    address_set_x_number = (
        address_token_set_score
        *
        number_jaccard
    )

    name_x_address = (
        name_token_set_score
        *
        address_token_set_score
    )

    name_jaccard_x_address_jaccard = (
        name_token_jaccard
        *
        address_token_jaccard
    )

    translit_x_address = (
        translit_token_set
        *
        address_token_set_score
    )

    translit_ratio_x_address_jaccard = (
        translit_ratio
        *
        address_token_jaccard
    )

    meaningful_name_x_address = (
        meaningful_s1_containment
        *
        meaningful_s2_containment
        *
        address_token_set_score
    )

    name_address_min = min(
        name_token_set_score,
        address_token_set_score,
    )

    name_address_max = max(
        name_token_set_score,
        address_token_set_score,
    )

    name_address_abs_diff = abs(
        name_token_set_score
        -
        address_token_set_score
    )


    # ========================================================
    # BLOCKER PROVENANCE
    # ========================================================

    blocker_count = (
        from_name
        +
        from_address
        +
        from_structured
        +
        from_embedding
    )


    # ========================================================
    # LABEL
    # ========================================================

    label = int(
        s2_id
        in truth.get(
            s1_id,
            set(),
        )
    )


    # ========================================================
    # RETURN ROW
    # ========================================================

    return {

        # metadata
        "source1_entity_id":
            s1_id,

        "candidate_entity_id":
            s2_id,

        "label":
            label,

        # basic name
        "name_exact":
            int(
                s1[
                    "name_norm"
                ]
                ==
                s2_name_norm
                and
                s1[
                    "name_norm"
                ] != ""
            ),

        "name_ratio":
            name_ratio,

        "name_token_sort":
            name_token_sort,

        "name_token_set":
            name_token_set_score,

        "name_token_jaccard":
            name_token_jaccard,

        "name_length_ratio":
            length_ratio(
                s1[
                    "name_norm"
                ],
                s2_name_norm,
            ),

        # name token structure
        "name_token_count_s1":
            len(
                s1[
                    "name_tokens"
                ]
            ),

        "name_token_count_s2":
            len(
                s2_name_tokens
            ),

        "shared_name_token_count":
            len(
                shared_name_tokens
            ),

        "s1_name_containment":
            s1_name_containment,

        "s2_name_containment":
            s2_name_containment,

        # meaningful name tokens
        "meaningful_name_token_count_s1":
            len(
                s1[
                    "meaningful_name_tokens"
                ]
            ),

        "meaningful_name_token_count_s2":
            len(
                s2_meaningful_tokens
            ),

        "shared_meaningful_name_tokens":
            len(
                shared_meaningful_tokens
            ),

        "extra_meaningful_tokens_s1":
            len(
                extra_meaningful_s1
            ),

        "extra_meaningful_tokens_s2":
            len(
                extra_meaningful_s2
            ),

        "meaningful_s1_containment":
            meaningful_s1_containment,

        "meaningful_s2_containment":
            meaningful_s2_containment,

        # legal suffix stripped
        "legal_suffix_stripped_ratio":
            legal_suffix_stripped_ratio,

        "legal_suffix_stripped_token_set":
            legal_suffix_stripped_token_set,

        # char ngram
        "name_char_3gram_jaccard":
            name_char_3gram_jaccard,

        # transliteration
        "translit_name_ratio":
            translit_ratio,

        "translit_name_token_sort":
            translit_token_sort,

        "translit_name_token_set":
            translit_token_set,

        # address
        "address_exact":
            int(
                s1[
                    "address_norm"
                ]
                ==
                s2_addr_norm
                and
                s1[
                    "address_norm"
                ] != ""
            ),

        "address_ratio":
            address_ratio,

        "address_token_sort":
            address_token_sort,

        "address_token_set":
            address_token_set_score,

        "address_token_jaccard":
            address_token_jaccard,

        "address_length_ratio":
            length_ratio(
                s1[
                    "address_norm"
                ],
                s2_addr_norm,
            ),

        "s1_address_missing":
            int(
                s1[
                    "address_norm"
                ]
                == ""
            ),

        "s2_address_missing":
            int(
                s2_addr_norm
                == ""
            ),

        # address number features
        "shared_number_count":
            len(
                shared_numbers
            ),

        "number_jaccard":
            number_jaccard,

        "number_exact_set":
            int(
                bool(
                    s1[
                        "numbers"
                    ]
                )
                and
                s1[
                    "numbers"
                ]
                ==
                s2_numbers
            ),

        "both_have_numbers":
            int(
                both_have_numbers
            ),

        "number_conflict":
            number_conflict,

        "ordered_number_exact":
            ordered_number_exact,

        "first_number_match":
            first_number_match,

        "first_number_conflict":
            first_number_conflict,

        # interaction features
        "strong_address_weak_name":
            strong_address_weak_name,

        "strong_name_weak_address":
            strong_name_weak_address,

        "address_number_agreement":
            address_number_agreement,

        "high_address_low_name":
            high_address_low_name,

        "translit_rescue":
            translit_rescue,

        "translit_address_rescue":
            translit_address_rescue,

        "strong_address_number_name_conflict":
            strong_address_number_name_conflict,

        "strong_name_number_conflict":
            strong_name_number_conflict,

        "address_x_number":
            address_x_number,

        "address_set_x_number":
            address_set_x_number,

        "name_x_address":
            name_x_address,

        "name_jaccard_x_address_jaccard":
            name_jaccard_x_address_jaccard,

        "translit_x_address":
            translit_x_address,

        "translit_ratio_x_address_jaccard":
            translit_ratio_x_address_jaccard,

        "meaningful_name_x_address":
            meaningful_name_x_address,

        "name_address_min":
            name_address_min,

        "name_address_max":
            name_address_max,

        "name_address_abs_diff":
            name_address_abs_diff,

        # blocker provenance
        "from_name_blocker":
            from_name,

        "from_address_blocker":
            from_address,

        "from_structured_blocker":
            from_structured,

        "from_embedding_blocker":
            from_embedding,

        "blocker_count":
            blocker_count,

        # retrieval information
        "name_retrieval_score":
            name_retrieval_score,

        "address_retrieval_score":
            address_retrieval_score,

        "embedding_rank":
            embedding_rank,
    }


# ============================================================
# PARQUET WRITING
# ============================================================

def flush_buffer(
    buffer,
    writer,
):

    df = pd.DataFrame(
        buffer
    )

    table = pa.Table.from_pandas(
        df,
        preserve_index=False,
    )

    if writer is None:

        writer = (
            pq.ParquetWriter(
                OUTPUT_PATH,
                table.schema,
                compression="zstd",
            )
        )

    writer.write_table(
        table
    )

    return writer


# ============================================================
# GENERATE FEATURES
# ============================================================

def generate_features(
    query_records,
    target_to_pairs,
    truth,
):

    print(
        "\nStreaming S2 and computing features..."
    )

    buffer = []

    writer = None

    total_read = 0
    candidate_rows = 0
    positives = 0
    negatives = 0

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
            "business_address",
        ],
    )

    for (
        chunk_no,
        chunk,
    ) in enumerate(
        reader,
        start=1,
    ):

        total_read += len(
            chunk
        )

        for row in chunk.itertuples(
            index=False
        ):

            target_id = (
                row.entity_id
            )

            pair_list = (
                target_to_pairs.get(
                    target_id
                )
            )

            if not pair_list:
                continue

            for pair in pair_list:

                (
                    s1_id,
                    *provenance,
                ) = pair

                feature_row = (
                    compute_features(
                        s1_id=s1_id,
                        s1=query_records[
                            s1_id
                        ],
                        s2_id=target_id,
                        s2_name=
                            row.business_name,
                        s2_address=
                            row.business_address,
                        truth=truth,
                        provenance=
                            provenance,
                    )
                )

                buffer.append(
                    feature_row
                )

                candidate_rows += 1

                if (
                    feature_row[
                        "label"
                    ]
                ):
                    positives += 1
                else:
                    negatives += 1

                if (
                    len(buffer)
                    >= WRITE_BUFFER_SIZE
                ):

                    writer = (
                        flush_buffer(
                            buffer,
                            writer,
                        )
                    )

                    buffer.clear()

        print(
            f"\r"
            f"chunks={chunk_no:<3}"
            f" read={total_read:,}"
            f" pairs={candidate_rows:,}"
            f" positives={positives:,}"
            f" elapsed={now()-t0:.1f}s",
            end="",
            flush=True,
        )

        del chunk
        gc.collect()

    if buffer:

        writer = (
            flush_buffer(
                buffer,
                writer,
            )
        )

    if writer is not None:
        writer.close()

    print()

    print(
        f"\nFeature rows written: "
        f"{candidate_rows:,}"
    )

    print(
        f"Positive pairs: "
        f"{positives:,}"
    )

    print(
        f"Negative pairs: "
        f"{negatives:,}"
    )

    print(
        f"Positive rate: "
        f"{positives/candidate_rows:.4%}"
    )

    print(
        f"Feature generation time: "
        f"{now()-t0:.2f}s"
    )

    return (
        candidate_rows,
        positives,
        negatives,
    )


# ============================================================
# INSPECT OUTPUT
# ============================================================

def inspect_output():

    print(
        "\nReading feature summary..."
    )

    df = pd.read_parquet(
        OUTPUT_PATH
    )

    print(
        f"Rows: "
        f"{len(df):,}"
    )

    print(
        f"Columns: "
        f"{len(df.columns)}"
    )

    print(
        "\nLabel distribution:"
    )

    print(
        df[
            "label"
        ].value_counts()
    )

    feature_cols = [
        "name_ratio",
        "name_token_set",
        "translit_name_token_set",

        "legal_suffix_stripped_ratio",
        "legal_suffix_stripped_token_set",

        "name_char_3gram_jaccard",

        "s1_name_containment",
        "s2_name_containment",

        "meaningful_s1_containment",
        "meaningful_s2_containment",

        "extra_meaningful_tokens_s1",
        "extra_meaningful_tokens_s2",

        "address_ratio",
        "address_token_set",
        "address_token_jaccard",

        "shared_number_count",
        "number_jaccard",
        "number_conflict",

        "strong_address_weak_name",
        "strong_name_weak_address",
        "address_number_agreement",
        "high_address_low_name",
        "translit_rescue",
        "translit_address_rescue",
        "strong_address_number_name_conflict",
        "strong_name_number_conflict",

        "address_x_number",
        "address_set_x_number",
        "name_x_address",
        "name_jaccard_x_address_jaccard",
        "translit_x_address",
        "translit_ratio_x_address_jaccard",
        "meaningful_name_x_address",

        "name_address_min",
        "name_address_max",
        "name_address_abs_diff",

        "blocker_count",
    ]

    print(
        "\nMean features by label:"
    )

    print(
        df.groupby(
            "label"
        )[
            feature_cols
        ]
        .mean()
        .T
    )


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = now()

    (
        query_ids,
        base,
        structured,
        embedding,
    ) = load_candidate_caches()

    query_records = (
        load_s1_records(
            query_ids
        )
    )

    truth = (
        load_truth(
            query_ids
        )
    )

    (
        target_to_pairs,
        expected_pairs,
    ) = build_candidate_pair_index(
        query_ids,
        base,
        structured,
        embedding,
    )

    (
        rows,
        positives,
        negatives,
    ) = generate_features(
        query_records,
        target_to_pairs,
        truth,
    )

    if (
        rows
        != expected_pairs
    ):

        raise ValueError(
            f"Expected "
            f"{expected_pairs:,} "
            f"pairs but generated "
            f"{rows:,}"
        )

    print(
        f"\nSaved matcher dataset:\n"
        f"{OUTPUT_PATH}"
    )

    inspect_output()

    print(
        f"\nTotal runtime: "
        f"{now()-total_start:.2f}s"
    )


if __name__ == "__main__":
    main()