import re

from rapidfuzz.fuzz import (
    ratio,
    token_sort_ratio,
    token_set_ratio,
)
from unidecode import unidecode

from src.normalize import (
    normalize_name,
    normalize_address,
)


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


GENERIC_NAME_TOKENS = LEGAL_SUFFIXES


# ============================================================
# BASIC HELPERS
# ============================================================

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

    return " ".join(
        token
        for token in text.split()
        if token not in LEGAL_SUFFIXES
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

    union = (
        a | b
    )

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

    if (
        la == 0
        and lb == 0
    ):
        return 1.0

    if (
        la == 0
        or lb == 0
    ):
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
# PRECOMPUTE ONE RECORD
# ============================================================

def prepare_record(
    business_name,
    business_address,
):

    name_norm = normalize_name(
        business_name
    )

    address_norm = normalize_address(
        business_address
    )

    return {

        "name_norm":
            name_norm,

        "name_translit":
            transliterate_name(
                business_name
            ),

        "name_no_suffix":
            strip_legal_suffix_tokens(
                name_norm
            ),

        "name_tokens":
            token_set(
                name_norm
            ),

        "meaningful_name_tokens":
            meaningful_tokens(
                name_norm
            ),

        "name_char_3grams":
            char_ngrams(
                name_norm,
                3,
            ),

        "address_norm":
            address_norm,

        "address_tokens":
            token_set(
                address_norm
            ),

        "numbers":
            extract_numbers(
                address_norm
            ),

        "ordered_numbers":
            ordered_numbers(
                address_norm
            ),
    }


# ============================================================
# PAIR FEATURES
# ============================================================

def compute_pair_features(
    left,
    right,
    provenance=None,
):

    """
    left/right are outputs of prepare_record().

    provenance is optional and may contain:

        from_name_blocker
        from_address_blocker
        from_structured_blocker
        from_embedding_blocker
        name_retrieval_score
        address_retrieval_score
        embedding_rank

    Source dataset (S2/S3) is deliberately NOT a model feature.
    """

    if provenance is None:

        provenance = {}


    # ========================================================
    # NAME TOKEN OVERLAPS
    # ========================================================

    shared_name_tokens = (
        left[
            "name_tokens"
        ]
        &
        right[
            "name_tokens"
        ]
    )

    shared_meaningful = (
        left[
            "meaningful_name_tokens"
        ]
        &
        right[
            "meaningful_name_tokens"
        ]
    )

    extra_meaningful_left = (
        left[
            "meaningful_name_tokens"
        ]
        -
        right[
            "meaningful_name_tokens"
        ]
    )

    extra_meaningful_right = (
        right[
            "meaningful_name_tokens"
        ]
        -
        left[
            "meaningful_name_tokens"
        ]
    )


    # ========================================================
    # BASIC NAME
    # ========================================================

    name_ratio = (
        ratio(
            left[
                "name_norm"
            ],
            right[
                "name_norm"
            ],
        )
        / 100.0
    )

    name_token_sort = (
        token_sort_ratio(
            left[
                "name_norm"
            ],
            right[
                "name_norm"
            ],
        )
        / 100.0
    )

    name_token_set_score = (
        token_set_ratio(
            left[
                "name_norm"
            ],
            right[
                "name_norm"
            ],
        )
        / 100.0
    )

    name_token_jaccard = (
        jaccard(
            left[
                "name_tokens"
            ],
            right[
                "name_tokens"
            ],
        )
    )


    # ========================================================
    # LEGAL-SUFFIX-STRIPPED NAME
    # ========================================================

    legal_suffix_stripped_ratio = (
        ratio(
            left[
                "name_no_suffix"
            ],
            right[
                "name_no_suffix"
            ],
        )
        / 100.0
    )

    legal_suffix_stripped_token_set = (
        token_set_ratio(
            left[
                "name_no_suffix"
            ],
            right[
                "name_no_suffix"
            ],
        )
        / 100.0
    )


    # ========================================================
    # CHAR 3-GRAM
    # ========================================================

    name_char_3gram_jaccard = (
        jaccard(
            left[
                "name_char_3grams"
            ],
            right[
                "name_char_3grams"
            ],
        )
    )


    # ========================================================
    # TRANSLITERATED NAME
    # ========================================================

    translit_name_ratio = (
        ratio(
            left[
                "name_translit"
            ],
            right[
                "name_translit"
            ],
        )
        / 100.0
    )

    translit_name_token_sort = (
        token_sort_ratio(
            left[
                "name_translit"
            ],
            right[
                "name_translit"
            ],
        )
        / 100.0
    )

    translit_name_token_set = (
        token_set_ratio(
            left[
                "name_translit"
            ],
            right[
                "name_translit"
            ],
        )
        / 100.0
    )


    # ========================================================
    # ADDRESS
    # ========================================================

    address_ratio = (
        ratio(
            left[
                "address_norm"
            ],
            right[
                "address_norm"
            ],
        )
        / 100.0
    )

    address_token_sort = (
        token_sort_ratio(
            left[
                "address_norm"
            ],
            right[
                "address_norm"
            ],
        )
        / 100.0
    )

    address_token_set_score = (
        token_set_ratio(
            left[
                "address_norm"
            ],
            right[
                "address_norm"
            ],
        )
        / 100.0
    )

    address_token_jaccard = (
        jaccard(
            left[
                "address_tokens"
            ],
            right[
                "address_tokens"
            ],
        )
    )


    # ========================================================
    # NUMBERS
    # ========================================================

    shared_numbers = (
        left[
            "numbers"
        ]
        &
        right[
            "numbers"
        ]
    )

    number_union = (
        left[
            "numbers"
        ]
        |
        right[
            "numbers"
        ]
    )

    number_jaccard = (
        len(
            shared_numbers
        )
        / len(
            number_union
        )
        if number_union
        else 0.0
    )

    both_have_numbers = (
        bool(
            left[
                "numbers"
            ]
        )
        and
        bool(
            right[
                "numbers"
            ]
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
            left[
                "ordered_numbers"
            ]
        )
        and
        (
            left[
                "ordered_numbers"
            ]
            ==
            right[
                "ordered_numbers"
            ]
        )
    )

    first_number_match = 0
    first_number_conflict = 0

    if (
        left[
            "ordered_numbers"
        ]
        and
        right[
            "ordered_numbers"
        ]
    ):

        if (
            left[
                "ordered_numbers"
            ][0]
            ==
            right[
                "ordered_numbers"
            ][0]
        ):

            first_number_match = 1

        else:

            first_number_conflict = 1


    # ========================================================
    # PROVENANCE
    # ========================================================

    from_name = int(
        provenance.get(
            "from_name_blocker",
            0,
        )
    )

    from_address = int(
        provenance.get(
            "from_address_blocker",
            0,
        )
    )

    from_structured = int(
        provenance.get(
            "from_structured_blocker",
            0,
        )
    )

    from_embedding = int(
        provenance.get(
            "from_embedding_blocker",
            0,
        )
    )

    blocker_count = (
        from_name
        + from_address
        + from_structured
        + from_embedding
    )


    # ========================================================
    # RETURN V2 FEATURES
    # ========================================================

    return {

        "name_exact":
            int(
                left[
                    "name_norm"
                ]
                ==
                right[
                    "name_norm"
                ]
                and
                left[
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
                left[
                    "name_norm"
                ],
                right[
                    "name_norm"
                ],
            ),


        # ----------------------------------------------------
        # Name structure
        # ----------------------------------------------------

        "name_token_count_s1":
            len(
                left[
                    "name_tokens"
                ]
            ),

        "name_token_count_s2":
            len(
                right[
                    "name_tokens"
                ]
            ),

        "shared_name_token_count":
            len(
                shared_name_tokens
            ),

        "s1_name_containment":
            safe_containment(
                shared_name_tokens,
                left[
                    "name_tokens"
                ],
            ),

        "s2_name_containment":
            safe_containment(
                shared_name_tokens,
                right[
                    "name_tokens"
                ],
            ),


        # ----------------------------------------------------
        # Meaningful name structure
        # ----------------------------------------------------

        "meaningful_name_token_count_s1":
            len(
                left[
                    "meaningful_name_tokens"
                ]
            ),

        "meaningful_name_token_count_s2":
            len(
                right[
                    "meaningful_name_tokens"
                ]
            ),

        "shared_meaningful_name_tokens":
            len(
                shared_meaningful
            ),

        "extra_meaningful_tokens_s1":
            len(
                extra_meaningful_left
            ),

        "extra_meaningful_tokens_s2":
            len(
                extra_meaningful_right
            ),

        "meaningful_s1_containment":
            safe_containment(
                shared_meaningful,
                left[
                    "meaningful_name_tokens"
                ],
            ),

        "meaningful_s2_containment":
            safe_containment(
                shared_meaningful,
                right[
                    "meaningful_name_tokens"
                ],
            ),


        # ----------------------------------------------------
        # Improved name representations
        # ----------------------------------------------------

        "legal_suffix_stripped_ratio":
            legal_suffix_stripped_ratio,

        "legal_suffix_stripped_token_set":
            legal_suffix_stripped_token_set,

        "name_char_3gram_jaccard":
            name_char_3gram_jaccard,

        "translit_name_ratio":
            translit_name_ratio,

        "translit_name_token_sort":
            translit_name_token_sort,

        "translit_name_token_set":
            translit_name_token_set,


        # ----------------------------------------------------
        # Address
        # ----------------------------------------------------

        "address_exact":
            int(
                left[
                    "address_norm"
                ]
                ==
                right[
                    "address_norm"
                ]
                and
                left[
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
                left[
                    "address_norm"
                ],
                right[
                    "address_norm"
                ],
            ),

        "s1_address_missing":
            int(
                left[
                    "address_norm"
                ]
                == ""
            ),

        "s2_address_missing":
            int(
                right[
                    "address_norm"
                ]
                == ""
            ),


        # ----------------------------------------------------
        # Address numbers
        # ----------------------------------------------------

        "shared_number_count":
            len(
                shared_numbers
            ),

        "number_jaccard":
            number_jaccard,

        "number_exact_set":
            int(
                bool(
                    left[
                        "numbers"
                    ]
                )
                and
                (
                    left[
                        "numbers"
                    ]
                    ==
                    right[
                        "numbers"
                    ]
                )
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


        # ----------------------------------------------------
        # Retrieval provenance
        # ----------------------------------------------------

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

        "name_retrieval_score":
            float(
                provenance.get(
                    "name_retrieval_score",
                    0.0,
                )
            ),

        "address_retrieval_score":
            float(
                provenance.get(
                    "address_retrieval_score",
                    0.0,
                )
            ),

        "embedding_rank":
            int(
                provenance.get(
                    "embedding_rank",
                    0,
                )
            ),
    }


# ============================================================
# MODEL FEATURE LIST
# ============================================================

MODEL_FEATURE_COLUMNS = [

    "name_exact",
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "name_token_jaccard",
    "name_length_ratio",

    "name_token_count_s1",
    "name_token_count_s2",
    "shared_name_token_count",
    "s1_name_containment",
    "s2_name_containment",

    "meaningful_name_token_count_s1",
    "meaningful_name_token_count_s2",
    "shared_meaningful_name_tokens",
    "extra_meaningful_tokens_s1",
    "extra_meaningful_tokens_s2",
    "meaningful_s1_containment",
    "meaningful_s2_containment",

    "legal_suffix_stripped_ratio",
    "legal_suffix_stripped_token_set",

    "name_char_3gram_jaccard",

    "translit_name_ratio",
    "translit_name_token_sort",
    "translit_name_token_set",

    "address_exact",
    "address_ratio",
    "address_token_sort",
    "address_token_set",
    "address_token_jaccard",
    "address_length_ratio",

    "s1_address_missing",
    "s2_address_missing",

    "shared_number_count",
    "number_jaccard",
    "number_exact_set",
    "both_have_numbers",
    "number_conflict",
    "ordered_number_exact",
    "first_number_match",
    "first_number_conflict",

    "from_name_blocker",
    "from_address_blocker",
    "from_structured_blocker",
    "from_embedding_blocker",

    "blocker_count",

    "name_retrieval_score",
    "address_retrieval_score",
    "embedding_rank",
]