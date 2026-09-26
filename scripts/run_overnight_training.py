from pathlib import Path
import argparse
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = (
    ROOT
    / "data"
    / "train"
)

S1_PATH = (
    TRAIN_DIR
    / "train_source1.tsv"
)


def run(command):

    print()
    print("=" * 80)
    print(
        "RUNNING:"
    )

    print(
        " ".join(command)
    )

    print("=" * 80)
    print()

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--max-entities-per-country",
        type=int,
        default=25_000,
    )


    parser.add_argument(
        "--batch-size",
        type=int,
        default=1_000,
    )


    parser.add_argument(
        "--max-train-negatives",
        type=int,
        default=20,
    )


    args = parser.parse_args()


    # ========================================================
    # DISCOVER COUNTRIES
    # ========================================================

    countries = set()


    reader = pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "country",
        ],
        chunksize=200_000,
    )


    for chunk in reader:

        countries.update(
            x
            for x
            in chunk[
                "country"
            ].unique()
            if x
        )


    countries = sorted(
        countries
    )


    print(
        "Countries discovered:",
        countries,
    )


    # ========================================================
    # GENERATE PAIRS
    # ========================================================

    for country in countries:

        for source in [
            "S2",
            "S3",
        ]:

            run(
                [
                    sys.executable,

                    "scripts/"
                    "build_training_pairs.py",

                    "--source",
                    source,

                    "--country",
                    country,

                    "--max-entities",
                    str(
                        args.max_entities_per_country
                    ),

                    "--batch-size",
                    str(
                        args.batch_size
                    ),

                    "--max-train-negatives",
                    str(
                        args.max_train_negatives
                    ),
                ]
            )


    # ========================================================
    # TRAIN MODEL
    # ========================================================

    run(
        [
            sys.executable,
            "scripts/train_final_matcher.py",
        ]
    )


    print()
    print("=" * 80)
    print(
        "OVERNIGHT PIPELINE COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()