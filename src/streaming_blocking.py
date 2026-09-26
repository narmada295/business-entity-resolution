import heapq
import math
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import HashingVectorizer
from sparse_dot_topn import sp_matmul_topn

from src.normalize import (
    normalize_name,
    normalize_address,
)


# ============================================================
# SETTINGS
# ============================================================

NAME_TOP_K = 20
ADDRESS_TOP_K = 20
STRUCTURED_TOP_K = 25

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
}


# ============================================================
# HELPERS
# ============================================================

def canonical_number(value):

    try:
        return str(int(value))
    except ValueError:
        return value


def extract_numbers(text):

    if not text:
        return set()

    return {
        canonical_number(x)
        for x in NUMBER_RE.findall(text)
    }


def tokenize_address(text):

    if not text:
        return set()

    return {
        token
        for token in text.split()
        if (
            token
            and token not in STRUCTURED_STOP_TOKENS
            and not token.isdigit()
        )
    }


def prepare_query_frame(df):

    result = df[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ]
    ].copy()

    result["name_norm"] = (
        result["business_name"]
        .map(normalize_name)
    )

    result["address_norm"] = (
        result["business_address"]
        .map(normalize_address)
    )

    result["address_tokens"] = (
        result["address_norm"]
        .map(tokenize_address)
    )

    result["address_numbers"] = (
        result["address_norm"]
        .map(extract_numbers)
    )

    return result.reset_index(
        drop=True
    )


def prepare_target_chunk(df):

    result = df[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ]
    ].copy()

    result["name_norm"] = (
        result["business_name"]
        .map(normalize_name)
    )

    result["address_norm"] = (
        result["business_address"]
        .map(normalize_address)
    )

    result["address_tokens"] = (
        result["address_norm"]
        .map(tokenize_address)
    )

    result["address_numbers"] = (
        result["address_norm"]
        .map(extract_numbers)
    )

    return result.reset_index(
        drop=True
    )


# ============================================================
# TOP-K STORAGE
# ============================================================

class TopKStore:

    """
    Keeps best score seen for each candidate ID,
    then returns top K.

    Designed for relatively small K.
    """

    def __init__(
        self,
        num_queries,
        k,
    ):

        self.k = k

        self.scores = [
            {}
            for _ in range(num_queries)
        ]


    def update(
        self,
        query_idx,
        candidate_id,
        score,
    ):

        current = self.scores[
            query_idx
        ].get(
            candidate_id
        )

        if (
            current is None
            or score > current
        ):

            self.scores[
                query_idx
            ][candidate_id] = float(
                score
            )


    def prune_query(
        self,
        query_idx,
    ):

        values = self.scores[
            query_idx
        ]

        if len(values) <= self.k:
            return

        best = heapq.nlargest(
            self.k,
            values.items(),
            key=lambda x: x[1],
        )

        self.scores[
            query_idx
        ] = dict(
            best
        )


    def prune_all(self):

        for idx in range(
            len(self.scores)
        ):
            self.prune_query(
                idx
            )


    def result(self):

        self.prune_all()

        return self.scores


# ============================================================
# VECTORISERS
# ============================================================

def make_name_vectorizer():

    return HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        n_features=HASH_FEATURES,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )


def make_address_vectorizer():

    return HashingVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        n_features=HASH_FEATURES,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )


# ============================================================
# STRUCTURED ADDRESS SCORE
# ============================================================

def structured_score(
    q_tokens,
    q_numbers,
    t_tokens,
    t_numbers,
):

    shared_tokens = (
        q_tokens
        & t_tokens
    )

    shared_numbers = (
        q_numbers
        & t_numbers
    )

    if (
        not shared_tokens
        and not shared_numbers
    ):
        return 0.0

    score = 0.0

    # Numbers get stronger weight.
    score += (
        2.5
        * len(shared_numbers)
    )

    score += (
        1.0
        * len(shared_tokens)
    )

    if (
        shared_tokens
        and shared_numbers
    ):
        score += 2.0

    if len(shared_numbers) >= 2:
        score += 2.0

    if len(shared_tokens) >= 2:
        score += 1.0

    return score


# ============================================================
# STREAMING BLOCKER
# ============================================================

class StreamingCountryBlocker:

    def __init__(
        self,
        query_df,
        name_top_k=NAME_TOP_K,
        address_top_k=ADDRESS_TOP_K,
        structured_top_k=STRUCTURED_TOP_K,
    ):

        self.query_df = (
            prepare_query_frame(
                query_df
            )
        )

        self.name_top_k = (
            name_top_k
        )

        self.address_top_k = (
            address_top_k
        )

        self.structured_top_k = (
            structured_top_k
        )

        self.name_vectorizer = (
            make_name_vectorizer()
        )

        self.address_vectorizer = (
            make_address_vectorizer()
        )


        self.query_name_matrix = (
            self.name_vectorizer.transform(
                self.query_df[
                    "name_norm"
                ].tolist()
            )
        )


        self.query_address_matrix = (
            self.address_vectorizer.transform(
                self.query_df[
                    "address_norm"
                ].tolist()
            )
        )


        n = len(
            self.query_df
        )

        self.name_store = TopKStore(
            n,
            self.name_top_k,
        )

        self.address_store = TopKStore(
            n,
            self.address_top_k,
        )

        self.structured_store = TopKStore(
            n,
            self.structured_top_k,
        )


    # ========================================================
    # ONE TARGET CHUNK
    # ========================================================

    def process_target_chunk(
        self,
        target_chunk,
    ):

        target_chunk = (
            prepare_target_chunk(
                target_chunk
            )
        )

        if len(target_chunk) == 0:
            return


        target_ids = (
            target_chunk[
                "entity_id"
            ]
            .tolist()
        )


        # ----------------------------------------------------
        # NAME CHAR RETRIEVAL
        # ----------------------------------------------------

        target_name_matrix = (
            self.name_vectorizer.transform(
                target_chunk[
                    "name_norm"
                ].tolist()
            )
        )


        name_sim = sp_matmul_topn(
            self.query_name_matrix,
            target_name_matrix.T,
            top_n=self.name_top_k,
            threshold=0.0,
            sort=True,
        )


        for q_idx in range(
            name_sim.shape[0]
        ):

            row = name_sim.getrow(
                q_idx
            )

            for (
                col_idx,
                score,
            ) in zip(
                row.indices,
                row.data,
            ):

                self.name_store.update(
                    q_idx,
                    target_ids[
                        col_idx
                    ],
                    score,
                )


        # ----------------------------------------------------
        # ADDRESS CHAR RETRIEVAL
        # ----------------------------------------------------

        target_address_matrix = (
            self.address_vectorizer.transform(
                target_chunk[
                    "address_norm"
                ].tolist()
            )
        )


        address_sim = sp_matmul_topn(
            self.query_address_matrix,
            target_address_matrix.T,
            top_n=self.address_top_k,
            threshold=0.0,
            sort=True,
        )


        for q_idx in range(
            address_sim.shape[0]
        ):

            row = address_sim.getrow(
                q_idx
            )

            for (
                col_idx,
                score,
            ) in zip(
                row.indices,
                row.data,
            ):

                self.address_store.update(
                    q_idx,
                    target_ids[
                        col_idx
                    ],
                    score,
                )


        # ----------------------------------------------------
        # STRUCTURED ADDRESS
        #
        # This simple streaming version compares only target
        # rows that share at least one token/number by building
        # a temporary chunk-local inverted index.
        # ----------------------------------------------------

        token_index = defaultdict(
            list
        )

        number_index = defaultdict(
            list
        )


        for t_idx, row in target_chunk.iterrows():

            for token in row[
                "address_tokens"
            ]:

                token_index[
                    token
                ].append(
                    t_idx
                )


            for number in row[
                "address_numbers"
            ]:

                number_index[
                    number
                ].append(
                    t_idx
                )


        for q_idx, qrow in self.query_df.iterrows():

            candidate_indices = set()


            for token in qrow[
                "address_tokens"
            ]:

                candidate_indices.update(
                    token_index.get(
                        token,
                        []
                    )
                )


            for number in qrow[
                "address_numbers"
            ]:

                candidate_indices.update(
                    number_index.get(
                        number,
                        []
                    )
                )


            for t_idx in candidate_indices:

                trow = target_chunk.iloc[
                    t_idx
                ]

                score = structured_score(
                    qrow[
                        "address_tokens"
                    ],
                    qrow[
                        "address_numbers"
                    ],
                    trow[
                        "address_tokens"
                    ],
                    trow[
                        "address_numbers"
                    ],
                )

                if score <= 0:
                    continue

                self.structured_store.update(
                    q_idx,
                    trow[
                        "entity_id"
                    ],
                    score,
                )


        # Keep stores bounded after every chunk.
        self.name_store.prune_all()
        self.address_store.prune_all()
        self.structured_store.prune_all()


    # ========================================================
    # FINAL UNION
    # ========================================================

    def finalize(self):

        name_results = (
            self.name_store.result()
        )

        address_results = (
            self.address_store.result()
        )

        structured_results = (
            self.structured_store.result()
        )


        all_results = []


        for q_idx in range(
            len(
                self.query_df
            )
        ):

            name_candidates = (
                name_results[
                    q_idx
                ]
            )

            address_candidates = (
                address_results[
                    q_idx
                ]
            )

            structured_candidates = (
                structured_results[
                    q_idx
                ]
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
                            in name_candidates
                        ),

                    "from_address_blocker":
                        int(
                            candidate_id
                            in address_candidates
                        ),

                    "from_structured_blocker":
                        int(
                            candidate_id
                            in structured_candidates
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