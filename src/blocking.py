import math
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import (
    HashingVectorizer,
)

from sparse_dot_topn import sp_matmul_topn

from src.normalize import (
    normalize_name,
    normalize_address,
)


# ============================================================
# DEFAULT SETTINGS
# ============================================================

NAME_TOP_K = 20
ADDRESS_TOP_K = 20

STRUCTURED_POOL_K = 100
STRUCTURED_KEEP_K = 25

HASH_FEATURES = 2 ** 18


NUMBER_RE = re.compile(r"\d+")


STRUCTURED_STOP_TOKENS = {
    "road",
    "rd",
    "street",
    "st",
    "lane",
    "ln",
    "floor",
    "fl",
    "building",
    "bldg",
    "house",
    "h",
    "no",
    "number",
    "plot",
    "shop",
    "door",
    "near",
    "opp",
    "opposite",
    "district",
    "city",
    "state",
    "india",
}


# ============================================================
# HELPERS
# ============================================================

def canonical_number(value):

    try:
        return str(
            int(value)
        )

    except ValueError:
        return value


def extract_numbers(text):

    if not text:
        return []

    return [
        canonical_number(x)
        for x in NUMBER_RE.findall(
            text
        )
    ]


def tokenize_address(text):

    if not text:
        return []

    return [
        token
        for token in text.split()
        if (
            token
            and
            token
            not in STRUCTURED_STOP_TOKENS
            and
            not token.isdigit()
        )
    ]


def lexical_name_similarity(
    query_name,
    candidate_name,
):

    from rapidfuzz.fuzz import (
        ratio,
        token_sort_ratio,
        token_set_ratio,
    )

    if (
        not query_name
        or
        not candidate_name
    ):
        return 0.0

    return (
        max(
            ratio(
                query_name,
                candidate_name,
            ),
            token_sort_ratio(
                query_name,
                candidate_name,
            ),
            token_set_ratio(
                query_name,
                candidate_name,
            ),
        )
        / 100.0
    )


# ============================================================
# COUNTRY DISCOVERY
# ============================================================

def discover_countries(
    *dataframes,
):

    """
    Return all countries appearing in the provided dataframes.

    No country names are hardcoded.
    """

    countries = set()

    for df in dataframes:

        if df is None:
            continue

        if "country" not in df.columns:
            continue

        values = (
            df[
                "country"
            ]
            .astype(str)
            .str.strip()
        )

        countries.update(
            value
            for value in values
            if value
        )

    return sorted(
        countries
    )


# ============================================================
# PREPARE COUNTRY DATA
# ============================================================

def prepare_blocking_frame(
    df,
):

    """
    Add normalized fields needed by blocking.

    Expected columns:
        entity_id
        business_name
        business_address
        country
    """

    result = df[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ]
    ].copy()

    result[
        "name_norm"
    ] = result[
        "business_name"
    ].map(
        normalize_name
    )

    result[
        "address_norm"
    ] = result[
        "business_address"
    ].map(
        normalize_address
    )

    return result


# ============================================================
# HASHING VECTORIZERS
# ============================================================

def make_name_vectorizer():

    return HashingVectorizer(

        analyzer="char_wb",

        ngram_range=(
            3,
            4,
        ),

        n_features=HASH_FEATURES,

        alternate_sign=False,

        norm="l2",

        lowercase=False,

        dtype=np.float32,
    )


def make_address_vectorizer():

    return HashingVectorizer(

        analyzer="char_wb",

        ngram_range=(
            3,
            5,
        ),

        n_features=HASH_FEATURES,

        alternate_sign=False,

        norm="l2",

        lowercase=False,

        dtype=np.float32,
    )


# ============================================================
# CHAR HASH BLOCKER
# ============================================================

def char_hash_candidates(
    query_texts,
    corpus_texts,
    corpus_ids,
    vectorizer,
    top_k,
):

    """
    Return:

        list[
            dict(
                candidate_id -> similarity_score
            )
        ]

    one dict per query.
    """

    if len(query_texts) == 0:

        return []

    if len(corpus_texts) == 0:

        return [
            {}
            for _ in query_texts
        ]

    query_matrix = vectorizer.transform(
        query_texts
    )

    corpus_matrix = vectorizer.transform(
        corpus_texts
    )

    similarities = sp_matmul_topn(

        query_matrix,

        corpus_matrix.T,

        top_n=top_k,

        threshold=0.0,

        sort=True,
    )

    results = []

    for query_idx in range(
        similarities.shape[0]
    ):

        row = similarities.getrow(
            query_idx
        )

        candidates = {}

        for (
            col_idx,
            score,
        ) in zip(
            row.indices,
            row.data,
        ):

            candidates[
                corpus_ids[
                    col_idx
                ]
            ] = float(
                score
            )

        results.append(
            candidates
        )

    return results


# ============================================================
# STRUCTURED ADDRESS INDEX
# ============================================================

class StructuredAddressIndex:

    def __init__(
        self,
        target_df,
        max_token_df=500,
        max_number_df=1000,
    ):

        self.target_df = (
            target_df.reset_index(
                drop=True
            )
        )

        self.max_token_df = (
            max_token_df
        )

        self.max_number_df = (
            max_number_df
        )

        self.token_index = defaultdict(
            list
        )

        self.number_index = defaultdict(
            list
        )

        self.token_df = Counter()
        self.number_df = Counter()

        self.target_tokens = []
        self.target_numbers = []

        self._build()


    def _build(self):

        for idx, row in self.target_df.iterrows():

            address = row[
                "address_norm"
            ]

            tokens = set(
                tokenize_address(
                    address
                )
            )

            numbers = set(
                extract_numbers(
                    address
                )
            )

            self.target_tokens.append(
                tokens
            )

            self.target_numbers.append(
                numbers
            )

            for token in tokens:

                self.token_df[
                    token
                ] += 1

            for number in numbers:

                self.number_df[
                    number
                ] += 1


        for idx in range(
            len(
                self.target_df
            )
        ):

            for token in self.target_tokens[
                idx
            ]:

                if (
                    self.token_df[
                        token
                    ]
                    <=
                    self.max_token_df
                ):

                    self.token_index[
                        token
                    ].append(
                        idx
                    )


            for number in self.target_numbers[
                idx
            ]:

                if (
                    self.number_df[
                        number
                    ]
                    <=
                    self.max_number_df
                ):

                    self.number_index[
                        number
                    ].append(
                        idx
                    )


    def _idf_weight(
        self,
        df_value,
    ):

        n = max(
            1,
            len(
                self.target_df
            ),
        )

        return math.log(
            1.0
            +
            n
            /
            max(
                1,
                df_value,
            )
        )


    def retrieve(
        self,
        query_address,
        pool_k=100,
    ):

        query_tokens = set(
            tokenize_address(
                query_address
            )
        )

        query_numbers = set(
            extract_numbers(
                query_address
            )
        )

        scores = defaultdict(
            float
        )


        # ----------------------------------------------------
        # Token evidence
        # ----------------------------------------------------

        for token in query_tokens:

            postings = (
                self.token_index.get(
                    token
                )
            )

            if not postings:
                continue

            weight = self._idf_weight(
                self.token_df[
                    token
                ]
            )

            for idx in postings:

                scores[
                    idx
                ] += weight


        # ----------------------------------------------------
        # Number evidence
        # ----------------------------------------------------

        for number in query_numbers:

            postings = (
                self.number_index.get(
                    number
                )
            )

            if not postings:
                continue

            weight = self._idf_weight(
                self.number_df[
                    number
                ]
            )

            for idx in postings:

                scores[
                    idx
                ] += (
                    2.5
                    *
                    weight
                )


        # ----------------------------------------------------
        # Bonuses
        # ----------------------------------------------------

        if scores:

            for idx in list(
                scores.keys()
            ):

                target_tokens = (
                    self.target_tokens[
                        idx
                    ]
                )

                target_numbers = (
                    self.target_numbers[
                        idx
                    ]
                )

                shared_tokens = (
                    query_tokens
                    &
                    target_tokens
                )

                shared_numbers = (
                    query_numbers
                    &
                    target_numbers
                )

                if (
                    shared_tokens
                    and
                    shared_numbers
                ):
                    scores[
                        idx
                    ] += 2.0

                if len(
                    shared_numbers
                ) >= 2:

                    scores[
                        idx
                    ] += 2.0

                if len(
                    shared_tokens
                ) >= 2:

                    scores[
                        idx
                    ] += 1.0


        best = sorted(

            scores.items(),

            key=lambda x:
                x[1],

            reverse=True,

        )[
            :pool_k
        ]


        return [
            (
                idx,
                score,
            )
            for (
                idx,
                score,
            ) in best
        ]


# ============================================================
# STRUCTURED RERANKING
# ============================================================

def structured_rerank(
    query_row,
    target_df,
    structured_pool,
    keep_k=25,
):

    if not structured_pool:
        return {}


    if not structured_pool:
        max_structured = 1.0

    else:

        max_structured = max(
            score
            for (
                _,
                score,
            )
            in structured_pool
        )

        if max_structured <= 0:
            max_structured = 1.0


    reranked = []


    for (
        target_idx,
        structured_score,
    ) in structured_pool:

        target_row = (
            target_df.iloc[
                target_idx
            ]
        )

        normalized_structured = (
            structured_score
            /
            max_structured
        )


        lexical_name = (
            lexical_name_similarity(
                query_row[
                    "name_norm"
                ],
                target_row[
                    "name_norm"
                ],
            )
        )


        # Original-name lexical score is deliberately
        # separated so we can later replace one side
        # with transliteration without changing API.
        normal_name_score = lexical_name


        final_score = (
            0.50
            *
            normalized_structured
            +
            0.35
            *
            lexical_name
            +
            0.15
            *
            normal_name_score
        )


        reranked.append(
            (
                target_row[
                    "entity_id"
                ],
                float(
                    final_score
                ),
            )
        )


    reranked.sort(
        key=lambda x:
            x[1],
        reverse=True,
    )


    return {
        entity_id: score
        for (
            entity_id,
            score,
        ) in reranked[
            :keep_k
        ]
    }


# ============================================================
# COUNTRY BLOCKER
# ============================================================

class CountryBlocker:

    """
    Candidate generator for one target source
    and one country.

    It makes no assumptions about what the country name is.
    """

    def __init__(
        self,
        target_df,
        name_top_k=NAME_TOP_K,
        address_top_k=ADDRESS_TOP_K,
        structured_pool_k=STRUCTURED_POOL_K,
        structured_keep_k=STRUCTURED_KEEP_K,
    ):

        self.target_df = (
            prepare_blocking_frame(
                target_df
            )
            .reset_index(
                drop=True
            )
        )

        self.name_top_k = (
            name_top_k
        )

        self.address_top_k = (
            address_top_k
        )

        self.structured_pool_k = (
            structured_pool_k
        )

        self.structured_keep_k = (
            structured_keep_k
        )


        self.target_ids = (
            self.target_df[
                "entity_id"
            ]
            .tolist()
        )


        self.name_vectorizer = (
            make_name_vectorizer()
        )

        self.address_vectorizer = (
            make_address_vectorizer()
        )


        self.name_matrix = (
            self.name_vectorizer.transform(
                self.target_df[
                    "name_norm"
                ].tolist()
            )
        )


        self.address_matrix = (
            self.address_vectorizer.transform(
                self.target_df[
                    "address_norm"
                ].tolist()
            )
        )


        self.structured_index = (
            StructuredAddressIndex(
                self.target_df
            )
        )


    def _char_candidates(
        self,
        query_df,
    ):

        query_names = (
            query_df[
                "name_norm"
            ]
            .tolist()
        )

        query_addresses = (
            query_df[
                "address_norm"
            ]
            .tolist()
        )


        name_query_matrix = (
            self.name_vectorizer.transform(
                query_names
            )
        )

        address_query_matrix = (
            self.address_vectorizer.transform(
                query_addresses
            )
        )


        name_sim = sp_matmul_topn(

            name_query_matrix,

            self.name_matrix.T,

            top_n=
                self.name_top_k,

            threshold=0.0,

            sort=True,
        )


        address_sim = sp_matmul_topn(

            address_query_matrix,

            self.address_matrix.T,

            top_n=
                self.address_top_k,

            threshold=0.0,

            sort=True,
        )


        name_results = []
        address_results = []


        for idx in range(
            len(
                query_df
            )
        ):

            row = name_sim.getrow(
                idx
            )

            name_candidates = {
                self.target_ids[
                    col
                ]:
                float(
                    score
                )
                for (
                    col,
                    score,
                )
                in zip(
                    row.indices,
                    row.data,
                )
            }

            name_results.append(
                name_candidates
            )


            row = address_sim.getrow(
                idx
            )

            address_candidates = {
                self.target_ids[
                    col
                ]:
                float(
                    score
                )
                for (
                    col,
                    score,
                )
                in zip(
                    row.indices,
                    row.data,
                )
            }

            address_results.append(
                address_candidates
            )


        return (
            name_results,
            address_results,
        )


    def generate(
        self,
        query_df,
    ):

        """
        Returns one candidate dictionary per query.

        candidate_id -> provenance dictionary
        """

        query_df = (
            prepare_blocking_frame(
                query_df
            )
            .reset_index(
                drop=True
            )
        )


        (
            name_results,
            address_results,
        ) = self._char_candidates(
            query_df
        )


        all_results = []


        for query_idx, query_row in query_df.iterrows():

            name_candidates = (
                name_results[
                    query_idx
                ]
            )

            address_candidates = (
                address_results[
                    query_idx
                ]
            )


            structured_pool = (
                self.structured_index.retrieve(
                    query_row[
                        "address_norm"
                    ],
                    pool_k=
                        self.structured_pool_k,
                )
            )


            structured_candidates = (
                structured_rerank(
                    query_row=
                        query_row,

                    target_df=
                        self.target_df,

                    structured_pool=
                        structured_pool,

                    keep_k=
                        self.structured_keep_k,
                )
            )


            candidate_ids = (
                set(
                    name_candidates
                )
                |
                set(
                    address_candidates
                )
                |
                set(
                    structured_candidates
                )
            )


            result = {}


            for candidate_id in candidate_ids:

                result[
                    candidate_id
                ] = {

                    "from_name_blocker":
                        int(
                            candidate_id
                            in
                            name_candidates
                        ),

                    "from_address_blocker":
                        int(
                            candidate_id
                            in
                            address_candidates
                        ),

                    "from_structured_blocker":
                        int(
                            candidate_id
                            in
                            structured_candidates
                        ),

                    "from_embedding_blocker":
                        0,

                    "name_retrieval_score":
                        float(
                            name_candidates.get(
                                candidate_id,
                                0.0,
                            )
                        ),

                    "address_retrieval_score":
                        float(
                            address_candidates.get(
                                candidate_id,
                                0.0,
                            )
                        ),

                    "structured_retrieval_score":
                        float(
                            structured_candidates.get(
                                candidate_id,
                                0.0,
                            )
                        ),

                    "embedding_rank":
                        0,
                }


            all_results.append(
                result
            )


        return all_results