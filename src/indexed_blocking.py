from pathlib import Path
import json

import numpy as np
from scipy import sparse

from sklearn.feature_extraction.text import HashingVectorizer
from sparse_dot_topn import sp_matmul_topn

from src.normalize import (
    normalize_name,
    normalize_address,
)


HASH_FEATURES = 2 ** 18

NAME_TOP_K = 20
ADDRESS_TOP_K = 20


def safe_country_name(country):
    return (
        country
        .replace("/", "_")
        .replace(" ", "_")
    )


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


class IndexedCharBlocker:

    def __init__(
        self,
        index_dir,
        source,
        country,
        name_top_k=NAME_TOP_K,
        address_top_k=ADDRESS_TOP_K,
    ):

        self.index_dir = Path(index_dir)

        self.source = source.upper()
        self.country = country

        self.name_top_k = name_top_k
        self.address_top_k = address_top_k

        safe_country = safe_country_name(
            country
        )

        prefix = (
            self.index_dir
            / f"{self.source.lower()}_{safe_country}"
        )

        self.name_path = Path(
            f"{prefix}_name.npz"
        )

        self.address_path = Path(
            f"{prefix}_address.npz"
        )

        self.ids_path = Path(
            f"{prefix}_ids.npy"
        )

        self.meta_path = Path(
            f"{prefix}_meta.json"
        )


        for path in [
            self.name_path,
            self.address_path,
            self.ids_path,
            self.meta_path,
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing index file: {path}"
                )


        print(
            f"Loading {self.source}/{country} indexes..."
        )


        self.target_ids = np.load(
            self.ids_path,
            allow_pickle=False,
        )


        self.name_matrix = sparse.load_npz(
            self.name_path
        ).tocsr()


        self.address_matrix = sparse.load_npz(
            self.address_path
        ).tocsr()


        with open(
            self.meta_path,
            "r",
        ) as f:
            self.metadata = json.load(f)


        if (
            self.name_matrix.shape[0]
            != len(self.target_ids)
        ):
            raise RuntimeError(
                "Name matrix / target ID mismatch"
            )


        if (
            self.address_matrix.shape[0]
            != len(self.target_ids)
        ):
            raise RuntimeError(
                "Address matrix / target ID mismatch"
            )


        self.name_vectorizer = (
            make_name_vectorizer()
        )

        self.address_vectorizer = (
            make_address_vectorizer()
        )


        print(
            f"Loaded {len(self.target_ids):,} targets"
        )


    def generate(
        self,
        query_df,
    ):

        names = (
            query_df["business_name"]
            .map(normalize_name)
            .tolist()
        )

        addresses = (
            query_df["business_address"]
            .map(normalize_address)
            .tolist()
        )


        # ====================================================
        # QUERY MATRICES
        # ====================================================

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


        # ====================================================
        # NAME RETRIEVAL
        # ====================================================

        name_result = sp_matmul_topn(
            query_name_matrix,
            self.name_matrix.T,
            top_n=self.name_top_k,
            threshold=0.0,
            sort=True,
        )


        # ====================================================
        # ADDRESS RETRIEVAL
        # ====================================================

        address_result = sp_matmul_topn(
            query_address_matrix,
            self.address_matrix.T,
            top_n=self.address_top_k,
            threshold=0.0,
            sort=True,
        )


        # ====================================================
        # UNION
        # ====================================================

        results = []


        for q_idx in range(
            len(query_df)
        ):

            candidates = {}


            # -----------------------------------------------
            # NAME
            # -----------------------------------------------

            row = name_result.getrow(
                q_idx
            )


            for col_idx, score in zip(
                row.indices,
                row.data,
            ):

                candidate_id = str(
                    self.target_ids[
                        col_idx
                    ]
                )


                candidates[
                    candidate_id
                ] = {
                    "from_name_blocker": 1,
                    "from_address_blocker": 0,
                    "from_structured_blocker": 0,
                    "from_embedding_blocker": 0,

                    "name_retrieval_score":
                        float(score),

                    "address_retrieval_score":
                        0.0,

                    "structured_retrieval_score":
                        0.0,

                    "embedding_rank":
                        0,
                }


            # -----------------------------------------------
            # ADDRESS
            # -----------------------------------------------

            row = address_result.getrow(
                q_idx
            )


            for col_idx, score in zip(
                row.indices,
                row.data,
            ):

                candidate_id = str(
                    self.target_ids[
                        col_idx
                    ]
                )


                if candidate_id not in candidates:

                    candidates[
                        candidate_id
                    ] = {
                        "from_name_blocker": 0,
                        "from_address_blocker": 1,
                        "from_structured_blocker": 0,
                        "from_embedding_blocker": 0,

                        "name_retrieval_score":
                            0.0,

                        "address_retrieval_score":
                            float(score),

                        "structured_retrieval_score":
                            0.0,

                        "embedding_rank":
                            0,
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


            results.append(
                candidates
            )


        return results