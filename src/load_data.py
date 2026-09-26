from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "train"
TEST_DIR = ROOT / "data" / "test"


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )


def load_train_data():
    s1 = read_tsv(TRAIN_DIR / "train_source1.tsv")
    s2 = read_tsv(TRAIN_DIR / "train_source2.tsv")
    s3 = read_tsv(TRAIN_DIR / "train_source3.tsv")
    gt = read_tsv(TRAIN_DIR / "train_ground_truth.tsv")

    return s1, s2, s3, gt


def load_test_data():
    s1 = read_tsv(TEST_DIR / "test_source1.tsv")
    s2 = read_tsv(TEST_DIR / "test_source2.tsv")
    s3 = read_tsv(TEST_DIR / "test_source3.tsv")

    return s1, s2, s3