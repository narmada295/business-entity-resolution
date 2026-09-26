import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "data" / "train"
OUTPUT_DIR = ROOT / "outputs"

FEATURE_PATH = (
    OUTPUT_DIR
    / "matcher_features_s2_india_sample.parquet"
)

GT_PATH = (
    TRAIN_DIR
    / "train_ground_truth.tsv"
)

MODEL_PATH = (
    OUTPUT_DIR
    / "lightgbm_matcher_sample.txt"
)

PREDICTION_PATH = (
    OUTPUT_DIR
    / "lightgbm_validation_predictions.parquet"
)

IMPORTANCE_PATH = (
    OUTPUT_DIR
    / "lightgbm_feature_importance.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_FRACTION = 0.70

SEED = 42

BETA = 0.5


# ============================================================
# GROUND TRUTH
# ============================================================

def parse_matches(value):

    if not value:
        return []

    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


def load_full_truth(query_ids):

    print(
        "\nLoading COMPLETE S2 ground truth..."
    )

    query_set = set(
        query_ids
    )

    # Include entities with zero S2 matches too.
    truth = {
        entity_id: set()
        for entity_id in query_ids
    }

    gt = pd.read_csv(
        GT_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    for row in gt.itertuples(
        index=False
    ):

        s1_id = (
            row.source1_entity_id
        )

        if s1_id not in query_set:
            continue

        for target_id in parse_matches(
            row.matched_entity_ids
        ):

            if target_id.startswith(
                "S2-"
            ):

                truth[
                    s1_id
                ].add(
                    target_id
                )

    total_links = sum(
        len(x)
        for x in truth.values()
    )

    zero_entities = sum(
        len(x) == 0
        for x in truth.values()
    )

    print(
        f"Full S2 truth links: "
        f"{total_links:,}"
    )

    print(
        f"Entities with zero S2 matches: "
        f"{zero_entities:,}"
    )

    return truth


# ============================================================
# ENTITY SPLIT
# ============================================================

def split_entities(entity_ids):

    """
    Entity-level split.

    Every candidate pair for one S1 entity stays entirely
    in train or entirely in validation.
    """

    entity_ids = np.array(
        sorted(entity_ids),
        dtype=object,
    )

    rng = np.random.default_rng(
        SEED
    )

    rng.shuffle(
        entity_ids
    )

    split_idx = int(
        len(entity_ids)
        * TRAIN_FRACTION
    )

    train_ids = set(
        entity_ids[
            :split_idx
        ]
    )

    val_ids = set(
        entity_ids[
            split_idx:
        ]
    )

    print(
        f"Matcher train entities: "
        f"{len(train_ids):,}"
    )

    print(
        f"Matcher validation entities: "
        f"{len(val_ids):,}"
    )

    assert len(
        train_ids & val_ids
    ) == 0

    assert (
        len(train_ids)
        + len(val_ids)
        == len(entity_ids)
    )

    return (
        train_ids,
        val_ids,
    )


# ============================================================
# ENTITY-LEVEL F-BETA
# ============================================================

def entity_fbeta(
    true_set,
    pred_set,
    beta=0.5,
):

    """
    Entity-level F-beta.

    Singleton behavior:

    true empty, pred empty
        -> 1

    true empty, pred non-empty
        -> 0
    """

    if not true_set:

        if not pred_set:
            return 1.0

        return 0.0

    if not pred_set:
        return 0.0

    tp = len(
        true_set
        & pred_set
    )

    fp = len(
        pred_set
        - true_set
    )

    fn = len(
        true_set
        - pred_set
    )

    beta2 = (
        beta
        * beta
    )

    numerator = (
        (1.0 + beta2)
        * tp
    )

    denominator = (
        (1.0 + beta2)
        * tp
        +
        beta2
        * fn
        +
        fp
    )

    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
    )


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

def evaluate_threshold(
    pred_df,
    truth,
    entity_ids,
    threshold,
):

    predicted = defaultdict(
        set
    )

    selected = pred_df[
        pred_df[
            "probability"
        ]
        >= threshold
    ]

    for row in selected.itertuples(
        index=False
    ):

        predicted[
            row.source1_entity_id
        ].add(
            row.candidate_entity_id
        )

    scores = []

    total_predictions = 0

    for s1_id in entity_ids:

        pred_set = (
            predicted.get(
                s1_id,
                set(),
            )
        )

        true_set = (
            truth.get(
                s1_id,
                set(),
            )
        )

        total_predictions += len(
            pred_set
        )

        scores.append(
            entity_fbeta(
                true_set,
                pred_set,
                beta=BETA,
            )
        )

    return (
        float(
            np.mean(
                scores
            )
        ),
        total_predictions,
    )


# ============================================================
# CANDIDATE-SET ORACLE
# ============================================================

def evaluate_candidate_oracle(
    val_df,
    truth,
    val_ids,
):

    """
    Assume a perfect classifier.

    It predicts every true candidate pair and no false pair.

    Any true link missing from candidate generation remains
    impossible to recover.

    This measures the maximum F0.5 achievable by the current
    candidate set.
    """

    candidate_sets = defaultdict(
        set
    )

    for row in val_df.itertuples(
        index=False
    ):

        candidate_sets[
            row.source1_entity_id
        ].add(
            row.candidate_entity_id
        )

    scores = []

    recovered_links = 0
    total_links = 0

    full_entities = 0
    partial_entities = 0
    zero_recovery_entities = 0

    entities_with_truth = 0

    for s1_id in val_ids:

        true_set = (
            truth.get(
                s1_id,
                set(),
            )
        )

        candidate_set = (
            candidate_sets.get(
                s1_id,
                set(),
            )
        )

        oracle_pred = (
            true_set
            & candidate_set
        )

        recovered_links += len(
            oracle_pred
        )

        total_links += len(
            true_set
        )

        if true_set:

            entities_with_truth += 1

            if len(
                oracle_pred
            ) == len(
                true_set
            ):

                full_entities += 1

            elif len(
                oracle_pred
            ) == 0:

                zero_recovery_entities += 1

            else:

                partial_entities += 1

        scores.append(
            entity_fbeta(
                true_set,
                oracle_pred,
                beta=BETA,
            )
        )

    pair_recall = (
        recovered_links
        / total_links
        if total_links
        else 0.0
    )

    macro_f05 = float(
        np.mean(
            scores
        )
    )

    full_pct = (
        full_entities
        / entities_with_truth
        if entities_with_truth
        else 0.0
    )

    zero_pct = (
        zero_recovery_entities
        / entities_with_truth
        if entities_with_truth
        else 0.0
    )

    print()

    print(
        "=" * 65
    )

    print(
        "CANDIDATE-SET ORACLE"
    )

    print(
        "=" * 65
    )

    print(
        f"Validation truth links: "
        f"{total_links:,}"
    )

    print(
        f"True links present in candidates: "
        f"{recovered_links:,}"
    )

    print(
        f"Pair blocking recall: "
        f"{pair_recall:.4%}"
    )

    print(
        f"Oracle macro F0.5: "
        f"{macro_f05:.6f}"
    )

    print(
        f"Full entity recovery: "
        f"{full_entities:,}"
        f" ({full_pct:.2%})"
    )

    print(
        f"Partial entity recovery: "
        f"{partial_entities:,}"
    )

    print(
        f"Zero entity recovery: "
        f"{zero_recovery_entities:,}"
        f" ({zero_pct:.2%})"
    )

    return {
        "pair_recall":
            pair_recall,

        "macro_f05":
            macro_f05,

        "full_entity_pct":
            full_pct,

        "zero_entity_pct":
            zero_pct,

        "recovered_links":
            recovered_links,

        "total_links":
            total_links,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load features
    # --------------------------------------------------------

    print(
        "Loading matcher feature dataset..."
    )

    df = pd.read_parquet(
        FEATURE_PATH
    )

    print(
        f"Pairs: "
        f"{len(df):,}"
    )

    query_ids = (
        df[
            "source1_entity_id"
        ]
        .unique()
        .tolist()
    )

    print(
        f"Entities: "
        f"{len(query_ids):,}"
    )

    # --------------------------------------------------------
    # Full GT
    # --------------------------------------------------------

    truth = load_full_truth(
        query_ids
    )

    # --------------------------------------------------------
    # Entity split
    # --------------------------------------------------------

    (
        train_ids,
        val_ids,
    ) = split_entities(
        query_ids
    )

    train_mask = (
        df[
            "source1_entity_id"
        ]
        .isin(
            train_ids
        )
    )

    val_mask = (
        df[
            "source1_entity_id"
        ]
        .isin(
            val_ids
        )
    )

    train_df = (
        df[
            train_mask
        ]
        .copy()
    )

    val_df = (
        df[
            val_mask
        ]
        .copy()
    )

    print(
        f"\nTrain candidate pairs: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation candidate pairs: "
        f"{len(val_df):,}"
    )

    print(
        f"Train positives: "
        f"{int(train_df['label'].sum()):,}"
    )

    print(
        f"Validation positives: "
        f"{int(val_df['label'].sum()):,}"
    )

    print(
        f"Train positive rate: "
        f"{train_df['label'].mean():.4%}"
    )

    print(
        f"Validation positive rate: "
        f"{val_df['label'].mean():.4%}"
    )

    # --------------------------------------------------------
    # Oracle BEFORE training
    # --------------------------------------------------------

    oracle = (
        evaluate_candidate_oracle(
            val_df,
            truth,
            val_ids,
        )
    )

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    NON_FEATURE_COLUMNS = [
        "source1_entity_id",
        "candidate_entity_id",
        "label",
    ]

    feature_cols = [
        col
        for col in df.columns
        if col not in NON_FEATURE_COLUMNS
    ]

    print()

    print(
        f"Model features: "
        f"{len(feature_cols)}"
    )

    for feature in feature_cols:

        print(
            " ",
            feature,
        )

    X_train = (
        train_df[
            feature_cols
        ]
    )

    y_train = (
        train_df[
            "label"
        ]
    )

    X_val = (
        val_df[
            feature_cols
        ]
    )

    y_val = (
        val_df[
            "label"
        ]
    )

    # --------------------------------------------------------
    # LightGBM
    # --------------------------------------------------------

    print(
        "\nTraining LightGBM..."
    )

    model = (
        lgb.LGBMClassifier(

            objective="binary",

            n_estimators=1500,

            learning_rate=0.05,

            num_leaves=63,

            min_child_samples=100,

            subsample=0.8,

            colsample_bytree=0.8,

            reg_lambda=1.0,

            random_state=SEED,

            n_jobs=-1,

            force_row_wise=True,
        )
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

        eval_metric=
            "binary_logloss",

        callbacks=[
            lgb.early_stopping(
                stopping_rounds=100
            ),

            lgb.log_evaluation(
                period=50
            ),
        ],
    )

    # --------------------------------------------------------
    # Validation probabilities
    # --------------------------------------------------------

    val_prob = (
        model.predict_proba(
            X_val
        )[:, 1]
    )

    print(
        "\nPairwise diagnostics:"
    )

    roc_auc = (
        roc_auc_score(
            y_val,
            val_prob,
        )
    )

    pr_auc = (
        average_precision_score(
            y_val,
            val_prob,
        )
    )

    print(
        f"ROC-AUC: "
        f"{roc_auc:.6f}"
    )

    print(
        f"PR-AUC: "
        f"{pr_auc:.6f}"
    )

    # --------------------------------------------------------
    # Prediction table
    # --------------------------------------------------------

    pred_df = (
        val_df[
            [
                "source1_entity_id",
                "candidate_entity_id",
                "label",
            ]
        ]
        .copy()
    )

    pred_df[
        "probability"
    ] = (
        val_prob
    )

    pred_df.to_parquet(
        PREDICTION_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Coarse threshold tuning
    # --------------------------------------------------------

    print(
        "\nTuning threshold on "
        "ENTITY-LEVEL macro F0.5..."
    )

    thresholds = np.arange(
        0.01,
        1.00,
        0.01,
    )

    results = []

    for threshold in thresholds:

        (
            score,
            predicted_links,
        ) = evaluate_threshold(
            pred_df,
            truth,
            val_ids,
            threshold,
        )

        results.append(
            (
                float(
                    threshold
                ),
                score,
                predicted_links,
            )
        )

    results.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    print()

    print(
        f"{'Threshold':>10}"
        f"{'Macro F0.5':>15}"
        f"{'Predicted links':>18}"
    )

    print(
        "-" * 45
    )

    for (
        threshold,
        score,
        predicted_links,
    ) in results[
        :15
    ]:

        print(
            f"{threshold:>10.2f}"
            f"{score:>15.6f}"
            f"{predicted_links:>18,}"
        )

    best_threshold = (
        results[0][0]
    )

    best_score = (
        results[0][1]
    )

    best_predictions = (
        results[0][2]
    )

    print()

    print(
        "Best threshold:",
        f"{best_threshold:.2f}",
    )

    print(
        "Best validation macro F0.5:",
        f"{best_score:.6f}",
    )

    print(
        "Predicted links at best threshold:",
        f"{best_predictions:,}",
    )

    # --------------------------------------------------------
    # Fine threshold search
    # --------------------------------------------------------

    fine_low = max(
        0.001,
        best_threshold
        - 0.02,
    )

    fine_high = min(
        0.999,
        best_threshold
        + 0.02,
    )

    fine_thresholds = np.arange(
        fine_low,
        fine_high
        + 0.0001,
        0.001,
    )

    fine_results = []

    for threshold in fine_thresholds:

        (
            score,
            predicted_links,
        ) = evaluate_threshold(
            pred_df,
            truth,
            val_ids,
            threshold,
        )

        fine_results.append(
            (
                float(
                    threshold
                ),
                score,
                predicted_links,
            )
        )

    fine_results.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    fine_best = (
        fine_results[0]
    )

    print()

    print(
        "Fine-tuned threshold:",
        f"{fine_best[0]:.3f}",
    )

    print(
        "Fine-tuned validation macro F0.5:",
        f"{fine_best[1]:.6f}",
    )

    print(
        "Predicted links:",
        f"{fine_best[2]:,}",
    )

    # --------------------------------------------------------
    # Compare model vs oracle
    # --------------------------------------------------------

    print()

    print(
        "=" * 65
    )

    print(
        "MODEL VS CANDIDATE ORACLE"
    )

    print(
        "=" * 65
    )

    print(
        f"Candidate oracle F0.5: "
        f"{oracle['macro_f05']:.6f}"
    )

    print(
        f"LightGBM best F0.5:    "
        f"{fine_best[1]:.6f}"
    )

    gap = (
        oracle[
            "macro_f05"
        ]
        - fine_best[1]
    )

    print(
        f"Matcher gap to oracle: "
        f"{gap:.6f}"
    )

    if oracle[
        "macro_f05"
    ] > 0:

        recovered_fraction = (
            fine_best[1]
            / oracle[
                "macro_f05"
            ]
        )

        print(
            "Fraction of oracle score achieved:",
            f"{recovered_fraction:.2%}",
        )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = (
        pd.DataFrame(
            {
                "feature":
                    feature_cols,

                "importance":
                    model.feature_importances_,
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    print()

    print(
        "=" * 60
    )

    print(
        "FEATURE IMPORTANCE"
    )

    print(
        "=" * 60
    )

    print(
        importance
        .to_string(
            index=False
        )
    )

    importance.to_csv(
        IMPORTANCE_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model.booster_.save_model(
        MODEL_PATH
    )

    print()

    print(
        f"Saved model:\n"
        f"{MODEL_PATH}"
    )

    print()

    print(
        f"Saved validation predictions:\n"
        f"{PREDICTION_PATH}"
    )

    print()

    print(
        f"Saved feature importance:\n"
        f"{IMPORTANCE_PATH}"
    )


if __name__ == "__main__":
    main()