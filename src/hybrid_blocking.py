from pathlib import Path
import math
import pickle
import re

import numpy as np
import pandas as pd

from rapidfuzz import fuzz
from unidecode import unidecode

from src.normalize import (
    normalize_name,
    normalize_address,
)

from src.sharded_blocking import (
    ShardedCharBlocker,
)


STRUCTURED_POOL_K = 100
STRUCTURED_KEEP_K = 25

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
        for x in NUMBER_RE.findall(
            text
        )
    }


def address_tokens(text):

    if not text:
        return set()

    output = set()


    for token in text.split():

        if token in STRUCTURED_STOP_TOKENS:
            continue

        if token.isdigit():
            continue

        if len(token) <= 1:
            continue

        output.add(token)


    return output


def transliterate(text):

    if not text:
        return ""

    return normalize_name(
        unidecode(text)
    )


class HybridBlocker:

    def __init__(
        self,
        index_dir,
        source,
        country,
        structured_pool_k=STRUCTURED_POOL_K,
        structured_keep_k=STRUCTURED_KEEP_K,
    ):

        self.index_dir = Path(
            index_dir
        )

        self.source = (
            source.upper()
        )

        self.country = country

        self.structured_pool_k = (
            structured_pool_k
        )

        self.structured_keep_k = (
            structured_keep_k
        )


        # ----------------------------------------------------
        # LEXICAL BLOCKER
        # ----------------------------------------------------

        self.lexical = ShardedCharBlocker(
            index_dir,
            source,
            country,
            max_route_keys=3,
            n_threads=1,
        )


        safe_country = safe_country_name(
            country
        )


        prefix = (
            self.index_dir
            / (
                f"{self.source.lower()}_"
                f"{safe_country}"
            )
        )


        structured_path = Path(
            f"{prefix}_structured.pkl"
        )

        records_path = Path(
            f"{prefix}_records.parquet"
        )


        print(
            "Loading structured index..."
        )


        with open(
            structured_path,
            "rb",
        ) as f:

            structured = (
                pickle.load(f)
            )


        self.token_postings = (
            structured[
                "token_postings"
            ]
        )

        self.number_postings = (
            structured[
                "number_postings"
            ]
        )

        self.token_df = (
            structured[
                "token_df"
            ]
        )

        self.number_df = (
            structured[
                "number_df"
            ]
        )

        self.target_count = (
            structured[
                "rows"
            ]
        )


        print(
            "Loading target record cache..."
        )


        self.records = pd.read_parquet(
            records_path,
            columns=[
                "entity_id",
                "name_norm",
                "address_norm",
            ],
        )


        self.target_names = (
            self.records[
                "name_norm"
            ].tolist()
        )

        self.target_addresses = (
            self.records[
                "address_norm"
            ].tolist()
        )


        self.target_ids = (
            self.records[
                "entity_id"
            ].tolist()
        )


        print(
            f"Structured records loaded: "
            f"{len(self.records):,}"
        )


    # ========================================================
    # STRUCTURED SCORE
    # ========================================================

    def raw_structured_score(
        self,
        q_tokens,
        q_numbers,
        t_tokens,
        t_numbers,
    ):

        shared_tokens = (
            q_tokens
            &
            t_tokens
        )

        shared_numbers = (
            q_numbers
            &
            t_numbers
        )


        if (
            not shared_tokens
            and
            not shared_numbers
        ):
            return 0.0


        score = 0.0


        # ----------------------------------------------------
        # TOKENS: IDF-LIKE WEIGHTING
        # ----------------------------------------------------

        for token in shared_tokens:

            df = self.token_df.get(
                token,
                self.target_count,
            )

            weight = (
                math.log(
                    (
                        self.target_count
                        + 1
                    )
                    /
                    (
                        df
                        + 1
                    )
                )
                + 1.0
            )

            score += weight


        # ----------------------------------------------------
        # NUMBERS: STRONGER
        # ----------------------------------------------------

        for number in shared_numbers:

            df = self.number_df.get(
                number,
                self.target_count,
            )

            weight = (
                math.log(
                    (
                        self.target_count
                        + 1
                    )
                    /
                    (
                        df
                        + 1
                    )
                )
                + 1.0
            )

            score += (
                2.5
                * weight
            )


        if (
            shared_tokens
            and
            shared_numbers
        ):
            score += 2.0


        if len(
            shared_numbers
        ) >= 2:
            score += 2.0


        if len(
            shared_tokens
        ) >= 2:
            score += 1.0


        return score


    # ========================================================
    # STRUCTURED RETRIEVAL FOR ONE QUERY
    # ========================================================

    def structured_candidates(
        self,
        business_name,
        business_address,
    ):

        q_name = normalize_name(
            business_name
        )

        q_address = normalize_address(
            business_address
        )


        q_tokens = address_tokens(
            q_address
        )

        q_numbers = extract_numbers(
            q_address
        )


        candidate_rows = set()


        # ----------------------------------------------------
        # EXACT SELECTIVE TOKENS
        # ----------------------------------------------------

        for token in q_tokens:

            postings = (
                self.token_postings.get(
                    token
                )
            )

            if postings is not None:

                candidate_rows.update(
                    postings.tolist()
                )


        # ----------------------------------------------------
        # EXACT NUMBERS
        # ----------------------------------------------------

        for number in q_numbers:

            postings = (
                self.number_postings.get(
                    number
                )
            )

            if postings is not None:

                candidate_rows.update(
                    postings.tolist()
                )


        if not candidate_rows:
            return {}


        # ----------------------------------------------------
        # RAW STRUCTURED POOL
        # ----------------------------------------------------

        scored = []


        for target_idx in candidate_rows:

            target_address = (
                self.target_addresses[
                    target_idx
                ]
            )


            t_tokens = address_tokens(
                target_address
            )

            t_numbers = extract_numbers(
                target_address
            )


            score = (
                self.raw_structured_score(
                    q_tokens,
                    q_numbers,
                    t_tokens,
                    t_numbers,
                )
            )


            if score > 0:

                scored.append(
                    (
                        target_idx,
                        score,
                    )
                )


        if not scored:
            return {}


        scored.sort(
            key=lambda x: x[1],
            reverse=True,
        )


        scored = scored[
            :self.structured_pool_k
        ]


        max_structured = (
            scored[0][1]
            if scored
            else 1.0
        )


        q_translit = transliterate(
            q_name
        )


        reranked = []


        for target_idx, raw_score in scored:

            target_name = (
                self.target_names[
                    target_idx
                ]
            )


            t_translit = transliterate(
                target_name
            )


            normal_name_score = (
                fuzz.ratio(
                    q_name,
                    target_name,
                )
                / 100.0
            )


            translit_name_score = (
                fuzz.ratio(
                    q_translit,
                    t_translit,
                )
                / 100.0
            )


            normalized_structured = (
                raw_score
                / max_structured
                if max_structured > 0
                else 0.0
            )


            final_score = (
                0.50
                * normalized_structured

                + 0.35
                * translit_name_score

                + 0.15
                * normal_name_score
            )


            reranked.append(
                (
                    target_idx,
                    final_score,
                )
            )


        reranked.sort(
            key=lambda x: x[1],
            reverse=True,
        )


        reranked = reranked[
            :self.structured_keep_k
        ]


        output = {}


        for target_idx, score in reranked:

            candidate_id = (
                self.target_ids[
                    target_idx
                ]
            )


            output[
                candidate_id
            ] = float(score)


        return output


    # ========================================================
    # COMPLETE HYBRID
    # ========================================================

    def generate(
        self,
        query_df,
    ):

        lexical_results = (
            self.lexical.generate(
                query_df
            )
        )


        final_results = []


        query_rows = list(
            query_df.itertuples(
                index=False
            )
        )


        for q_idx, row in enumerate(
            query_rows
        ):

            candidates = {
                candidate_id:
                    dict(values)

                for candidate_id, values
                in lexical_results[
                    q_idx
                ].items()
            }


            structured = (
                self.structured_candidates(
                    row.business_name,
                    row.business_address,
                )
            )


            for candidate_id, score in (
                structured.items()
            ):

                if candidate_id not in candidates:

                    candidates[
                        candidate_id
                    ] = {
                        "from_name_blocker":
                            0,

                        "from_address_blocker":
                            0,

                        "from_structured_blocker":
                            1,

                        "from_embedding_blocker":
                            0,

                        "name_retrieval_score":
                            0.0,

                        "address_retrieval_score":
                            0.0,

                        "structured_retrieval_score":
                            float(score),

                        "embedding_rank":
                            0,
                    }

                else:

                    candidates[
                        candidate_id
                    ][
                        "from_structured_blocker"
                    ] = 1

                    candidates[
                        candidate_id
                    ][
                        "structured_retrieval_score"
                    ] = float(score)


                    # Ensure common fields exist.
                    candidates[
                        candidate_id
                    ].setdefault(
                        "from_embedding_blocker",
                        0,
                    )

                    candidates[
                        candidate_id
                    ].setdefault(
                        "structured_retrieval_score",
                        float(score),
                    )

                    candidates[
                        candidate_id
                    ].setdefault(
                        "embedding_rank",
                        0,
                    )


            # Normalize missing structured fields
            # for lexical-only candidates.
            for values in candidates.values():

                values.setdefault(
                    "from_structured_blocker",
                    0,
                )

                values.setdefault(
                    "from_embedding_blocker",
                    0,
                )

                values.setdefault(
                    "structured_retrieval_score",
                    0.0,
                )

                values.setdefault(
                    "embedding_rank",
                    0,
                )


            final_results.append(
                candidates
            )


        return final_results