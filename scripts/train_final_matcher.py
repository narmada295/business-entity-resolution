from pathlib import Path
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from src.features import (
    MODEL_FEATURE_COLUMNS,
)


PAIR_ROOT = (
    ROOT
    / "outputs"
    / "training_pairs"
)

MODEL_DIR = (
    ROOT
    / "outputs"
    / "models"
)

TRAIN_DIR = (
    ROOT
    / "data"
    / "train"
)

GT_PATH = (
    TRAIN_DIR
    / "train_ground_truth.tsv"
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HELPERS
# ============================================================

def parse_matches(value):

    if not value:
        return set()

    return {
        x.strip()
        for x in value.split(",")
        if x.strip()
    }


def load_validation_entities():

    paths = list(
        PAIR_ROOT.glob(
            "*/*/queries.parquet"
        )
    )


    if not paths:

        raise RuntimeError(
            "No query manifests found."
        )


    ids = set()


    for path in paths:

        df = pd.read_parquet(
            path
        )


        ids.update(
            df.loc[
                df["split"] == "val",
                "entity_id",
            ].tolist()
        )


    return sorted(
        ids
    )


def load_truth_counts(
    entity_ids,
):

    entity_ids = set(
        entity_ids
    )


    result = {
        qid: 0
        for qid in entity_ids
    }


    reader = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
        chunksize=200_000,
    )


    for chunk in reader:

        chunk = chunk[
            chunk[
                "source1_entity_id"
            ].isin(
                entity_ids
            )
        ]


        for row in chunk.itertuples(
            index=False
        ):

            result[
                row.source1_entity_id
            ] = len(
                parse_matches(
                    row.matched_entity_ids
                )
            )


    return result


# ============================================================
# ENTITY F0.5
# ============================================================

def entity_macro_f05(
    probabilities,
    labels,
    entity_codes,
    truth_counts,
    threshold,
):

    prediction_mask = (
        probabilities
        >= threshold
    )


    n_entities = len(
        truth_counts
    )


    predicted_counts = np.bincount(
        entity_codes[
            prediction_mask
        ],
        minlength=n_entities,
    )


    tp_counts = np.bincount(
        entity_codes[
            prediction_mask
            &
            (labels == 1)
        ],
        minlength=n_entities,
    )


    fp_counts = (
        predicted_counts
        - tp_counts
    )


    fn_counts = (
        truth_counts
        - tp_counts
    )


    beta2 = 0.25


    denominator = (
        (1.0 + beta2)
        * tp_counts

        + beta2
        * fn_counts

        + fp_counts
    )


    scores = np.zeros(
        n_entities,
        dtype=np.float64,
    )


    both_empty = (
        (truth_counts == 0)
        &
        (predicted_counts == 0)
    )


    scores[
        both_empty
    ] = 1.0


    valid = (
        ~both_empty
        &
        (denominator > 0)
    )


    scores[
        valid
    ] = (
        (1.0 + beta2)
        * tp_counts[
            valid
        ]
        /
        denominator[
            valid
        ]
    )


    return float(
        scores.mean()
    )


# ============================================================
# LOAD PAIRS
# ============================================================

print(
    "Finding pair partitions..."
)


paths = sorted(
    PAIR_ROOT.glob(
        "*/*/part_*.parquet"
    )
)


if not paths:

    raise RuntimeError(
        "No training pair files found."
    )


print(
    "Partitions:",
    len(paths),
)


frames = []


columns = [
    "source1_entity_id",
    "candidate_entity_id",
    "split",
    "label",
] + list(
    MODEL_FEATURE_COLUMNS
)


started = time.perf_counter()


for idx, path in enumerate(
    paths,
    start=1,
):

    frame = pd.read_parquet(
        path,
        columns=columns,
    )


    frames.append(
        frame
    )


    print(
        f"\r"
        f"loaded "
        f"{idx}/{len(paths)}",
        end="",
        flush=True,
    )


print()


data = pd.concat(
    frames,
    ignore_index=True,
)


del frames


print(
    "Rows:",
    f"{len(data):,}",
)


print(
    "Positives:",
    f"{int(data['label'].sum()):,}",
)


print(
    "Load time:",
    f"{time.perf_counter() - started:.2f}s",
)


# ============================================================
# MATRICES
# ============================================================

train_mask = (
    data["split"]
    == "train"
)


val_mask = (
    data["split"]
    == "val"
)


X_train = (
    data.loc[
        train_mask,
        MODEL_FEATURE_COLUMNS,
    ]
    .astype(
        np.float32
    )
)


y_train = (
    data.loc[
        train_mask,
        "label",
    ]
    .astype(
        np.uint8
    )
    .to_numpy()
)


X_val = (
    data.loc[
        val_mask,
        MODEL_FEATURE_COLUMNS,
    ]
    .astype(
        np.float32
    )
)


y_val = (
    data.loc[
        val_mask,
        "label",
    ]
    .astype(
        np.uint8
    )
    .to_numpy()
)


val_qids = (
    data.loc[
        val_mask,
        "source1_entity_id",
    ]
    .astype(str)
    .to_numpy()
)


print()
print(
    "Train pairs:",
    f"{len(X_train):,}",
)

print(
    "Validation pairs:",
    f"{len(X_val):,}",
)


# ============================================================
# TRAIN
# ============================================================

print()
print(
    "=" * 70
)

print(
    "TRAINING LIGHTGBM"
)

print(
    "=" * 70
)


model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=1500,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=100,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=16,
    verbosity=-1,
)


train_started = (
    time.perf_counter()
)


model.fit(
    X_train,
    y_train,

    eval_set=[
        (
            X_val,
            y_val,
        )
    ],

    eval_metric="binary_logloss",

    callbacks=[
        lgb.early_stopping(
            100,
            verbose=True,
        ),
    ],
)


train_time = (
    time.perf_counter()
    - train_started
)


print(
    "Training time:",
    f"{train_time / 60:.2f} min",
)


# ============================================================
# PREDICTIONS
# ============================================================

probabilities = (
    model.predict_proba(
        X_val
    )[
        :,
        1
    ]
)


roc_auc = (
    roc_auc_score(
        y_val,
        probabilities,
    )
)


pr_auc = (
    average_precision_score(
        y_val,
        probabilities,
    )
)


print()
print(
    "ROC AUC:",
    f"{roc_auc:.6f}",
)

print(
    "PR AUC:",
    f"{pr_auc:.6f}",
)


# ============================================================
# ENTITY SET
# ============================================================

validation_entities = (
    load_validation_entities()
)


entity_to_idx = {
    qid: idx
    for idx, qid
    in enumerate(
        validation_entities
    )
}


entity_codes = np.asarray(
    [
        entity_to_idx[
            qid
        ]
        for qid
        in val_qids
    ],
    dtype=np.int32,
)


truth_map = load_truth_counts(
    validation_entities
)


truth_counts = np.asarray(
    [
        truth_map[
            qid
        ]
        for qid
        in validation_entities
    ],
    dtype=np.int32,
)


# ============================================================
# CANDIDATE ORACLE
# ============================================================

candidate_tp = np.bincount(
    entity_codes[
        y_val == 1
    ],
    minlength=len(
        validation_entities
    ),
)


candidate_fn = (
    truth_counts
    - candidate_tp
)


beta2 = 0.25


oracle_denominator = (
    (1.0 + beta2)
    * candidate_tp

    +
    beta2
    * candidate_fn
)


oracle_scores = np.zeros(
    len(
        validation_entities
    ),
    dtype=float,
)


both_empty = (
    truth_counts == 0
)


oracle_scores[
    both_empty
] = 1.0


valid = (
    ~both_empty
    &
    (oracle_denominator > 0)
)


oracle_scores[
    valid
] = (
    (1.0 + beta2)
    * candidate_tp[
        valid
    ]
    /
    oracle_denominator[
        valid
    ]
)


oracle_f05 = float(
    oracle_scores.mean()
)


print()
print(
    "Candidate oracle F0.5:",
    f"{oracle_f05:.6f}",
)


# ============================================================
# THRESHOLD SEARCH
# ============================================================

print()
print(
    "Searching threshold..."
)


coarse_thresholds = np.arange(
    0.05,
    0.951,
    0.025,
)


best_threshold = None
best_score = -1.0


for threshold in (
    coarse_thresholds
):

    score = entity_macro_f05(
        probabilities,
        y_val,
        entity_codes,
        truth_counts,
        threshold,
    )


    if score > best_score:

        best_score = score
        best_threshold = float(
            threshold
        )


refine_low = max(
    0.001,
    best_threshold
    - 0.04,
)


refine_high = min(
    0.999,
    best_threshold
    + 0.04,
)


refined_thresholds = np.arange(
    refine_low,
    refine_high
    + 1e-9,
    0.002,
)


for threshold in (
    refined_thresholds
):

    score = entity_macro_f05(
        probabilities,
        y_val,
        entity_codes,
        truth_counts,
        threshold,
    )


    if score > best_score:

        best_score = score
        best_threshold = float(
            threshold
        )


print()
print(
    "Best threshold:",
    f"{best_threshold:.4f}",
)

print(
    "Best macro F0.5:",
    f"{best_score:.6f}",
)


print(
    "Fraction of oracle:",
    f"{best_score / oracle_f05:.2%}"
    if oracle_f05 > 0
    else "N/A",
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = (
    MODEL_DIR
    / "lightgbm_v2.txt"
)


model.booster_.save_model(
    model_path
)


importance = pd.DataFrame(
    {
        "feature":
            MODEL_FEATURE_COLUMNS,

        "importance":
            model.feature_importances_,
    }
)


importance = (
    importance
    .sort_values(
        "importance",
        ascending=False,
    )
)


importance.to_csv(
    MODEL_DIR
    / "feature_importance.csv",
    index=False,
)


metrics = {
    "roc_auc":
        roc_auc,

    "pr_auc":
        pr_auc,

    "candidate_oracle_f05":
        oracle_f05,

    "best_threshold":
        best_threshold,

    "best_macro_f05":
        best_score,

    "fraction_of_oracle":
        (
            best_score / oracle_f05
            if oracle_f05 > 0
            else None
        ),

    "train_pairs":
        len(X_train),

    "validation_pairs":
        len(X_val),

    "validation_entities":
        len(
            validation_entities
        ),

    "training_seconds":
        train_time,
}


with open(
    MODEL_DIR
    / "metrics.json",
    "w",
) as f:

    json.dump(
        metrics,
        f,
        indent=2,
    )


print()
print(
    "Saved model:",
    model_path,
)

print(
    "Saved metrics:",
    MODEL_DIR
    / "metrics.json",
)