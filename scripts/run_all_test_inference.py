from pathlib import Path
import argparse
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TEST_DIR = (
    ROOT
    / "data"
    / "test"
)

S1_PATH = (
    TEST_DIR
    / "test_source1.tsv"
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

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
    )


    parser.add_argument(
        "--max-entities",
        type=int,
        default=0,
    )


    args = parser.parse_args()


    countries = set()


    for chunk in pd.read_csv(
        S1_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "country",
        ],
        chunksize=200_000,
    ):

        countries.update(
            x
            for x in chunk[
                "country"
            ].unique()
            if x
        )


    countries = sorted(
        countries
    )


    print(
        "Countries:",
        countries,
    )


    for country in countries:

        for source in [
            "S2",
            "S3",
        ]:

            command = [
                sys.executable,

                "scripts/"
                "run_test_inference.py",

                "--source",
                source,

                "--country",
                country,

                "--batch-size",
                str(
                    args.batch_size
                ),
            ]


            if (
                args.max_entities
                > 0
            ):

                command.extend(
                    [
                        "--max-entities",
                        str(
                            args.max_entities
                        ),
                    ]
                )


            run(
                command
            )


    # Only finalize complete inference.
    if args.max_entities == 0:

        run(
            [
                sys.executable,
                "scripts/"
                "finalize_submission.py",
            ]
        )


if __name__ == "__main__":
    main()