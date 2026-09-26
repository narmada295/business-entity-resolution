import sys
from pathlib import Path

import pandas as pd
from rapidfuzz.fuzz import ratio, token_sort_ratio, token_set_ratio
from unidecode import unidecode


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.normalize import normalize_name


INPUT_PATH = (
    ROOT
    / "outputs"
    / "s2_india_blocking_misses.csv"
)

OUTPUT_PATH = (
    ROOT
    / "outputs"
    / "transliteration_miss_analysis.csv"
)


THRESHOLDS = [60, 70, 80, 90]


def contains_non_ascii(text):
    return any(
        ord(ch) > 127
        for ch in str(text)
    )


def transliterate_name(text):
    """
    Convert Unicode scripts to approximate Latin text,
    then run our normal name normalization.
    """

    if not text:
        return ""

    return normalize_name(
        unidecode(text)
    )


def similarity(a, b):
    """
    Take several cheap lexical similarities.

    token_set_ratio is useful when:
        Private Limited / Pvt Ltd
        token order
        extra legal suffixes

    token_sort_ratio is useful for reordered names.
    """

    if not a or not b:
        return 0.0

    return max(
        ratio(a, b),
        token_sort_ratio(a, b),
        token_set_ratio(a, b),
    )


def main():

    print("Loading previous blocking misses...")

    df = pd.read_csv(
        INPUT_PATH,
        dtype=str,
        keep_default_na=False,
    )

    print(
        f"Missed true links loaded: "
        f"{len(df):,}"
    )

    # -------------------------------------------------
    # Original normalized names
    # -------------------------------------------------

    df["s1_name_norm"] = (
        df["s1_name"]
        .map(normalize_name)
    )

    df["s2_name_norm"] = (
        df["s2_name"]
        .map(normalize_name)
    )

    # -------------------------------------------------
    # Transliteration
    # -------------------------------------------------

    df["s1_name_translit"] = (
        df["s1_name"]
        .map(transliterate_name)
    )

    df["s2_name_translit"] = (
        df["s2_name"]
        .map(transliterate_name)
    )

    # -------------------------------------------------
    # Detect likely script mismatch
    # -------------------------------------------------

    df["s1_non_ascii"] = (
        df["s1_name"]
        .map(contains_non_ascii)
    )

    df["s2_non_ascii"] = (
        df["s2_name"]
        .map(contains_non_ascii)
    )

    df["cross_script_like"] = (
        df["s1_non_ascii"]
        != df["s2_non_ascii"]
    )

    # -------------------------------------------------
    # Similarities
    # -------------------------------------------------

    original_scores = []
    translit_scores = []

    for row in df.itertuples(
        index=False
    ):

        original_scores.append(
            similarity(
                row.s1_name_norm,
                row.s2_name_norm,
            )
        )

        translit_scores.append(
            similarity(
                row.s1_name_translit,
                row.s2_name_translit,
            )
        )

    df["original_name_score"] = (
        original_scores
    )

    df["translit_name_score"] = (
        translit_scores
    )

    df["translit_gain"] = (
        df["translit_name_score"]
        - df["original_name_score"]
    )

    # -------------------------------------------------
    # Overall statistics
    # -------------------------------------------------

    print("\n" + "=" * 80)
    print("TRANSLITERATION SIGNAL — ALL PREVIOUS MISSES")
    print("=" * 80)

    print(
        f"Cross-script-like pairs: "
        f"{df['cross_script_like'].sum():,}"
        f" / {len(df):,}"
        f" "
        f"({df['cross_script_like'].mean():.2%})"
    )

    print(
        "\nOriginal score:"
    )

    print(
        df[
            "original_name_score"
        ].describe(
            percentiles=[
                .25,
                .5,
                .75,
                .9,
            ]
        )
    )

    print(
        "\nTransliterated score:"
    )

    print(
        df[
            "translit_name_score"
        ].describe(
            percentiles=[
                .25,
                .5,
                .75,
                .9,
            ]
        )
    )

    print(
        "\nTransliteration gain:"
    )

    print(
        df[
            "translit_gain"
        ].describe(
            percentiles=[
                .25,
                .5,
                .75,
                .9,
            ]
        )
    )

    # -------------------------------------------------
    # Threshold analysis
    # -------------------------------------------------

    print("\n" + "=" * 80)
    print("THRESHOLD COVERAGE")
    print("=" * 80)

    for threshold in THRESHOLDS:

        original = (
            df["original_name_score"]
            >= threshold
        ).mean()

        translit = (
            df["translit_name_score"]
            >= threshold
        ).mean()

        print(
            f"score >= {threshold}: "
            f"original={original:.2%} "
            f"translit={translit:.2%} "
            f"gain={translit-original:+.2%}"
        )

    # -------------------------------------------------
    # Cross-script subset
    # -------------------------------------------------

    cross = df[
        df["cross_script_like"]
    ].copy()

    print("\n" + "=" * 80)
    print("CROSS-SCRIPT SUBSET")
    print("=" * 80)

    print(
        f"Pairs: {len(cross):,}"
    )

    if len(cross):

        for threshold in THRESHOLDS:

            original = (
                cross[
                    "original_name_score"
                ]
                >= threshold
            ).mean()

            translit = (
                cross[
                    "translit_name_score"
                ]
                >= threshold
            ).mean()

            print(
                f"score >= {threshold}: "
                f"original={original:.2%} "
                f"translit={translit:.2%} "
                f"gain={translit-original:+.2%}"
            )

    # -------------------------------------------------
    # Large improvements
    # -------------------------------------------------

    improved = df[
        df["translit_gain"] >= 30
    ].copy()

    print("\n" + "=" * 80)
    print("LARGE TRANSLITERATION IMPROVEMENTS")
    print("=" * 80)

    print(
        f"Gain >= 30 points: "
        f"{len(improved):,}"
        f" ({len(improved)/len(df):.2%})"
    )

    examples = (
        improved
        .sort_values(
            "translit_gain",
            ascending=False,
        )
        .head(30)
    )

    for i, row in enumerate(
        examples.itertuples(
            index=False
        ),
        start=1,
    ):

        print(
            f"\n--- {i} ---"
        )

        print(
            "S1:",
            row.s1_name,
        )

        print(
            "S2:",
            row.s2_name,
        )

        print(
            "S1 translit:",
            row.s1_name_translit,
        )

        print(
            "S2 translit:",
            row.s2_name_translit,
        )

        print(
            "Original score:",
            f"{row.original_name_score:.1f}",
        )

        print(
            "Translit score:",
            f"{row.translit_name_score:.1f}",
        )

        print(
            "Gain:",
            f"{row.translit_gain:+.1f}",
        )

    # -------------------------------------------------
    # Save
    # -------------------------------------------------

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nSaved detailed results to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()