from pathlib import Path
import pickle

import numpy as np

from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sparse_dot_topn import sp_matmul_topn

from src.normalize import (
    normalize_name,
    normalize_address,
)

from src.indexed_blocking import (
    make_name_vectorizer,
    make_address_vectorizer,
)


MAX_ROUTE_KEYS = 3

NAME_TOP_K = 20
ADDRESS_TOP_K = 20


NAME_STOP_TOKENS = {
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


ADDRESS_STOP_TOKENS = {
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


def safe_country_name(country):
    return (
        country
        .replace("/", "_")
        .replace(" ", "_")
    )


def token_prefix(token):

    token = "".join(
        ch
        for ch in token
        if ch.isalnum()
    )

    if not token:
        return None

    if len(token) == 1:
        return f"p:{token}"

    return f"p:{token[:2]}"


def name_route_keys(text):

    text = normalize_name(text)

    keys = set()


    for token in text.split():

        if token in NAME_STOP_TOKENS:
            continue

        key = token_prefix(
            token
        )

        if key:
            keys.add(key)


    return keys


def address_route_keys(text):

    text = normalize_address(text)

    keys = set()


    for token in text.split():

        cleaned = "".join(
            ch
            for ch in token
            if ch.isalnum()
        )


        if not cleaned:
            continue


        if cleaned.isdigit():

            try:
                cleaned = str(
                    int(cleaned)
                )
            except ValueError:
                pass

            keys.add(
                f"n:{cleaned}"
            )

            continue


        if token in ADDRESS_STOP_TOKENS:
            continue


        key = token_prefix(
            token
        )

        if key:
            keys.add(key)


    return keys


def choose_rarest_keys(
    keys,
    df,
    max_keys,
):

    present = [
        key
        for key in keys
        if key in df
    ]


    present.sort(
        key=lambda key: df[key]
    )


    return present[
        :max_keys
    ]


class TopKCandidateStore:

    def __init__(
        self,
        n_queries,
        k,
    ):

        self.k = k

        self.values = [
            {}
            for _ in range(
                n_queries
            )
        ]


    def update(
        self,
        query_idx,
        candidate_idx,
        score,
    ):

        previous = self.values[
            query_idx
        ].get(
            candidate_idx
        )


        if (
            previous is None
            or score > previous
        ):

            self.values[
                query_idx
            ][candidate_idx] = (
                float(score)
            )


    def prune(self):

        for idx in range(
            len(self.values)
        ):

            values = self.values[
                idx
            ]


            if len(values) <= self.k:
                continue


            best = sorted(
                values.items(),
                key=lambda x: x[1],
                reverse=True,
            )[
                :self.k
            ]


            self.values[
                idx
            ] = dict(
                best
            )


class ShardedCharBlocker:

    def __init__(
        self,
        index_dir,
        source,
        country,
        name_top_k=NAME_TOP_K,
        address_top_k=ADDRESS_TOP_K,
        max_route_keys=MAX_ROUTE_KEYS,
        n_threads=1,
    ):

        self.index_dir = Path(
            index_dir
        )

        self.source = (
            source.upper()
        )

        self.country = country

        self.name_top_k = (
            name_top_k
        )

        self.address_top_k = (
            address_top_k
        )

        self.max_route_keys = (
            max_route_keys
        )

        self.n_threads = (
            n_threads
        )


        safe_country = (
            safe_country_name(
                country
            )
        )


        prefix = (
            self.index_dir
            / (
                f"{self.source.lower()}_"
                f"{safe_country}"
            )
        )


        print(
            f"Loading sparse matrices "
            f"for {self.source}/{country}..."
        )


        self.target_ids = np.load(
            Path(
                f"{prefix}_ids.npy"
            ),
            allow_pickle=False,
        )


        self.name_matrix = (
            sparse.load_npz(
                Path(
                    f"{prefix}_name.npz"
                )
            )
            .tocsr()
        )


        self.address_matrix = (
            sparse.load_npz(
                Path(
                    f"{prefix}_address.npz"
                )
            )
            .tocsr()
        )


        routing_path = Path(
            f"{prefix}_routing.pkl"
        )


        with open(
            routing_path,
            "rb",
        ) as f:

            routing = (
                pickle.load(f)
            )


        self.name_postings = (
            routing[
                "name_postings"
            ]
        )

        self.address_postings = (
            routing[
                "address_postings"
            ]
        )

        self.name_df = (
            routing[
                "name_df"
            ]
        )

        self.address_df = (
            routing[
                "address_df"
            ]
        )


        self.name_vectorizer = (
            make_name_vectorizer()
        )

        self.address_vectorizer = (
            make_address_vectorizer()
        )


        print(
            f"Loaded "
            f"{len(self.target_ids):,} "
            f"targets"
        )


    # ========================================================
    # GENERATE
    # ========================================================

    def generate(
        self,
        query_df,
    ):

        n_queries = len(
            query_df
        )


        names = (
            query_df[
                "business_name"
            ]
            .map(normalize_name)
            .tolist()
        )


        addresses = (
            query_df[
                "business_address"
            ]
            .map(normalize_address)
            .tolist()
        )


        query_name_matrix = (
            self.name_vectorizer
            .transform(names)
            .tocsr()
        )


        query_address_matrix = (
            self.address_vectorizer
            .transform(addresses)
            .tocsr()
        )


        name_store = (
            TopKCandidateStore(
                n_queries,
                self.name_top_k,
            )
        )


        address_store = (
            TopKCandidateStore(
                n_queries,
                self.address_top_k,
            )
        )


        # ====================================================
        # QUERY → ROUTING KEYS
        # ====================================================

        name_key_to_queries = {}

        address_key_to_queries = {}


        for q_idx in range(
            n_queries
        ):

            keys = (
                name_route_keys(
                    names[q_idx]
                )
            )


            keys = choose_rarest_keys(
                keys,
                self.name_df,
                self.max_route_keys,
            )


            for key in keys:

                name_key_to_queries.setdefault(
                    key,
                    [],
                ).append(
                    q_idx
                )


            keys = (
                address_route_keys(
                    addresses[
                        q_idx
                    ]
                )
            )


            keys = choose_rarest_keys(
                keys,
                self.address_df,
                self.max_route_keys,
            )


            for key in keys:

                address_key_to_queries.setdefault(
                    key,
                    [],
                ).append(
                    q_idx
                )


        # ====================================================
        # NAME SHARDS
        # ====================================================

        for key, query_indices in (
            name_key_to_queries.items()
        ):

            target_indices = (
                self.name_postings.get(
                    key
                )
            )


            if (
                target_indices is None
                or len(target_indices) == 0
            ):
                continue


            q_indices = np.asarray(
                query_indices,
                dtype=np.int32,
            )


            target_submatrix = (
                self.name_matrix[
                    target_indices
                ]
            )


            result = sp_matmul_topn(
                query_name_matrix[
                    q_indices
                ],
                target_submatrix.T,
                top_n=self.name_top_k,
                threshold=0.0,
                sort=True,
                n_threads=self.n_threads,
            )


            for local_q_idx in range(
                result.shape[0]
            ):

                global_q_idx = int(
                    q_indices[
                        local_q_idx
                    ]
                )


                row = result.getrow(
                    local_q_idx
                )


                for local_target_idx, score in zip(
                    row.indices,
                    row.data,
                ):

                    global_target_idx = int(
                        target_indices[
                            local_target_idx
                        ]
                    )


                    name_store.update(
                        global_q_idx,
                        global_target_idx,
                        score,
                    )


        name_store.prune()


        # ====================================================
        # ADDRESS SHARDS
        # ====================================================

        for key, query_indices in (
            address_key_to_queries.items()
        ):

            target_indices = (
                self.address_postings.get(
                    key
                )
            )


            if (
                target_indices is None
                or len(target_indices) == 0
            ):
                continue


            q_indices = np.asarray(
                query_indices,
                dtype=np.int32,
            )


            target_submatrix = (
                self.address_matrix[
                    target_indices
                ]
            )


            result = sp_matmul_topn(
                query_address_matrix[
                    q_indices
                ],
                target_submatrix.T,
                top_n=self.address_top_k,
                threshold=0.0,
                sort=True,
                n_threads=self.n_threads,
            )


            for local_q_idx in range(
                result.shape[0]
            ):

                global_q_idx = int(
                    q_indices[
                        local_q_idx
                    ]
                )


                row = result.getrow(
                    local_q_idx
                )


                for local_target_idx, score in zip(
                    row.indices,
                    row.data,
                ):

                    global_target_idx = int(
                        target_indices[
                            local_target_idx
                        ]
                    )


                    address_store.update(
                        global_q_idx,
                        global_target_idx,
                        score,
                    )


        address_store.prune()


        # ====================================================
        # UNION
        # ====================================================

        output = []


        for q_idx in range(
            n_queries
        ):

            candidates = {}


            for target_idx, score in (
                name_store.values[
                    q_idx
                ].items()
            ):

                candidate_id = str(
                    self.target_ids[
                        target_idx
                    ]
                )


                candidates[
                    candidate_id
                ] = {
                    "from_name_blocker": 1,
                    "from_address_blocker": 0,

                    "name_retrieval_score":
                        float(score),

                    "address_retrieval_score":
                        0.0,
                }


            for target_idx, score in (
                address_store.values[
                    q_idx
                ].items()
            ):

                candidate_id = str(
                    self.target_ids[
                        target_idx
                    ]
                )


                if candidate_id not in candidates:

                    candidates[
                        candidate_id
                    ] = {
                        "from_name_blocker": 0,
                        "from_address_blocker": 1,

                        "name_retrieval_score":
                            0.0,

                        "address_retrieval_score":
                            float(score),
                    }

                else:

                    candidates[
                        candidate_id
                    ][
                        "from_address_blocker"
                    ] = 1

                    candidates[
                        candidate_id
                    ][
                        "address_retrieval_score"
                    ] = float(score)


            output.append(
                candidates
            )


        return output