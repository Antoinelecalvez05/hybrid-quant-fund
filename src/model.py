"""
model.py
--------
Random Forest classifier — the sole decision-making layer.

V2 changes:
  - class_weight added to reduce upward prediction bias
  - confusion_matrix_df added for Streamlit display
  - evaluate_model returns accuracy, ROC-AUC, classification report, and confusion matrix
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from config import (
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF,
    RF_RANDOM_STATE,
    RF_CLASS_WEIGHT,
)

logger = logging.getLogger(__name__)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """
    Fit a Random Forest on the training data.
    """

    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=RF_RANDOM_STATE,
        class_weight=RF_CLASS_WEIGHT,
        n_jobs=-1,
    )

    logger.info(
        "Training RandomForest on %d samples, %d features",
        len(X_train),
        X_train.shape[1],
    )

    clf.fit(X_train, y_train)

    logger.info("Training complete.")

    return clf


def predict_signals(
    clf: RandomForestClassifier,
    X: pd.DataFrame,
) -> pd.Series:
    """
    Return the predicted probability of the 'up' class for each row.
    """

    probabilities = clf.predict_proba(X)

    up_class_index = list(clf.classes_).index(1)

    return pd.Series(
        probabilities[:, up_class_index],
        index=X.index,
        name="signal",
    )


def evaluate_model(
    clf: RandomForestClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    period_name: str = "",
) -> dict:
    """
    Compute diagnostic classification metrics.
    """

    y_pred = clf.predict(X)
    y_proba = predict_signals(clf, X)

    accuracy = accuracy_score(y, y_pred)

    try:
        auc = roc_auc_score(y, y_proba)
    except ValueError:
        auc = 0.0

    report = classification_report(
        y,
        y_pred,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(
        y,
        y_pred,
        labels=[0, 1],
    )

    logger.info("[%s] accuracy=%.4f  AUC=%.4f", period_name, accuracy, auc)

    return {
        "accuracy": accuracy,
        "auc": auc,
        "report": report,
        "confusion": cm,
        "y_pred": pd.Series(y_pred, index=y.index),
    }


def confusion_matrix_df(cm: np.ndarray) -> pd.DataFrame:
    """
    Convert a confusion matrix ndarray into a labelled DataFrame.

    Rows = actual class.
    Columns = predicted class.
    """

    return pd.DataFrame(
        cm,
        index=["Actual Down (0)", "Actual Up (1)"],
        columns=["Predicted Down (0)", "Predicted Up (1)"],
    )


def feature_importance_df(
    clf: RandomForestClassifier,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Return feature importances sorted from most to least important.
    """

    return (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": clf.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
