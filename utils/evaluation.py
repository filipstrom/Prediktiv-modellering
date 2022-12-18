# -*- coding: utf-8 -*-

"""Module with model evaluation functions"""

from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from constants import FIG_SIZE_16_9, METRICS, MIN_RANK, STAGE_CHEMICAL_COLS
from functions import calculate_shap_values


def pct_obs_within_pm_1(
    y: Union[pd.Series, np.ndarray],
    y_hat: Union[pd.Series, np.ndarray],
) -> float:
    """Calculate share of predictions within one unit of the actual.

    :param y: Actual observations.
    :type y: Union[pd.Series, np.ndarray]
    :param y_hat: Predictions.
    :type y_hat: Union[pd.Series, np.ndarray]
    :return: Share of observations within one unit of actual.
    :rtype: float
    """
    distance = np.abs(y - y_hat)

    return (distance <= 1).sum() / len(y)


def mean_bias(
    y: Union[pd.Series, np.ndarray],
    y_hat: Union[pd.Series, np.ndarray],
) -> float:
    """Calculate mean bias of the predictions.

    Bias is defined as:
    prediction - actual

    :param y: Actual observations.
    :type y: Union[pd.Series, np.ndarray]
    :param y_hat: Predictions.
    :type y_hat: Union[pd.Series,  np.ndarray]
    :return: Mean bias.
    :rtype: float
    """
    return (y_hat - y).mean()


def evaluate_metrics(
    y: np.ndarray,
    y_hat: np.ndarray,
) -> Dict[str, float]:
    """Evaluate all model metrics.

    :param y: Actual values.
    :param y_hat: Predicted values.
    :return: Model metrics.
    """
    # Supported metrics
    metric_funcs: Dict = {
        "explained_variance_score": explained_variance_score,
        "mean_absolute_error": mean_absolute_error,
        "mean_absolute_percentage_error": mean_absolute_percentage_error,
        "mean_bias": mean_bias,
        "mean_squared_error": mean_squared_error,
        "median_absolute_error": median_absolute_error,
        "pct_obs_within_pm_1": pct_obs_within_pm_1,
        "r2_score": r2_score,
    }
    # Evaluate all metrics
    model_metrics = {metric: metric_funcs[metric](y, y_hat) for metric in METRICS}

    return model_metrics


def check_feature_rank(
    shap_values: pd.DataFrame,
    feature: str,
    min_rank: int,
) -> Dict[str, int]:
    """Check that the ranking of the impact of a feature.

    Impact is calculated as the mean of absolute SHAP values.

    :param shap_values: SHAP values.
    :param feature: Feature to check.
    :param min_rank: Required min rank to pass test.
    :return: Binary check status (0/1).
    """
    # Mean of absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    # Sort array twice to get the rank within array
    mean_abs_shap_rank = mean_abs_shap.argsort().argsort()
    # Sorting ascending - reverse it
    mean_abs_shap_rank = np.abs(mean_abs_shap_rank - mean_abs_shap_rank.max())
    # Get chemical position in features
    feature_index = list(shap_values.columns).index(feature)
    # Get chemical rank
    feature_rank = mean_abs_shap_rank[feature_index]
    # Perform check (highest rank = 0)
    check_status = 1 if feature_rank < min_rank else 0

    return {
        "status": check_status,
        "feature_rank": feature_rank,
    }


def check_feature_effect(
    shap_values: pd.DataFrame,
    feature_values: pd.DataFrame,
    feature: str,
) -> Dict[str, Union[int, List[float], float]]:
    """Check that the modeled effect of a feature is positive and monotonic.

    This is done by first calculating the median SHAP and feature values for bins
    percentiles 0-25, 25-50, 50-75 and 75-100 in terms of feature values. The median
    SHAP values within a bin are then compared to the median in the previous bin, and
    required to always be greater or equal.

    :param shap_values: SHAP values.
    :param feature_values: Feature values.
    :param feature: Feature whose effect to check.
    :return: Binary check status (0/1).
    """
    # Construct dataframe with feature // SHAP values
    df = pd.DataFrame(
        data={
            "shap_value": shap_values[feature],
            "feature_value": feature_values[feature],
        },
        index=feature_values.index,
    )
    # Group by bins and calculate median feature // SHAP values
    medians_df = df.groupby(
        by=pd.cut(
            x=df["feature_value"],
            bins=np.percentile(
                a=df["feature_value"],
                q=[0, 25, 50, 75, 100],
            ),
            include_lowest=True,
        ),
        as_index=False,
    ).median()
    # Sort by median feature value
    median_shap_values = medians_df.sort_values(
        by="feature_value",
        ascending=True,
    )["shap_value"]
    # Neighboring bin diffs
    neighbor_diffs = (median_shap_values - median_shap_values.shift(1)).dropna()
    total_span = median_shap_values.iloc[-1] - median_shap_values.iloc[0]
    # Perform check
    check_status = 1 if (neighbor_diffs >= 0).all() else 0

    return {
        "status": check_status,
        "shap_diffs": neighbor_diffs,
        "shap_span": total_span,
    }


def check_model(
    stage: str,
    model: Any,
    model_type: str,
    feature_values: pd.DataFrame,
) -> Dict[str, Dict]:
    """Perform validity checks on model.

    :param stage: Bleaching stage being modeled.
    :param model: Trained model.
    :param model_type: Type of trained model.
    :param feature_values: Feature values to perform checks for.
    :return: Check results.
    """
    # Calculate SHAP values
    shap_values = calculate_shap_values(
        model=model,
        model_type=model_type,
        X_calibrate=feature_values,
        X_explain=feature_values,
    )
    # Name of chemical feature variable to check
    chemical_col = STAGE_CHEMICAL_COLS[stage]
    # Perform checks
    check_results = {
        "stage": stage,
        "chemical": chemical_col.split("_")[-1],
        "rank_check": check_feature_rank(
            shap_values=shap_values,
            feature=chemical_col,
            min_rank=MIN_RANK,
        ),
        "effect_check": check_feature_effect(
            shap_values=shap_values,
            feature_values=feature_values,
            feature=chemical_col,
        ),
    }

    return check_results


def create_check_summary(model_check: Dict) -> str:
    stage = model_check["stage"]
    chemical = model_check["chemical"]
    rank_status = model_check["rank_check"]["status"]
    rank = model_check["rank_check"]["feature_rank"] + 1
    effect_status = model_check["effect_check"]["status"]
    shap_diffs = list(model_check["effect_check"]["shap_diffs"].round(2).to_numpy())
    shap_span = model_check["effect_check"]["shap_span"].round(2)
    summary_string = [
        f"|-----------------------------------------------|",
        f"| Check summary for stage {stage.capitalize()} with chemical {chemical} |",
        f"|-----------------------------------------------|",
        f"|",
        f"| Overall check status: {'OK' if rank_status*effect_status else 'not OK'}",
        f"|",
        f"| Rank check: {'OK' if rank_status else 'not OK'}",
        f"|   - Chemical feature rank: {rank}",
        f"|",
        f"| Effect check: {'OK' if effect_status else 'not OK'}",
        f"|   - Neighbor SHAP diffs: {shap_diffs}",
        f"|   - Overall SHAP span: {shap_span}",
        f"|",
    ]

    return "\n".join(summary_string)


def evaluate_model(
    model: Any,
    feature_values: pd.DataFrame,
    actuals: pd.Series,
) -> None:
    """Evaluate model against testing set.

    :param model: Model to be evaluated.
    :param feature_values: Predictors.
    :param actuals: Actual observations.
    :return: Evaluation metrics .
    """
    # Predictions & errors
    predictions = model.predict(feature_values)
    # Calculate metrics
    model_metrics = evaluate_metrics(actuals, predictions)
    print("Model metrics")
    for k, v in model_metrics.items():
        print(f"  - {k}: {round(v,4)}")
    print()

    # Time series plot
    _, ax = plt.subplots(figsize=FIG_SIZE_16_9)
    ax.plot(feature_values.index, actuals, color="darkgray")
    ax.plot(feature_values.index, predictions, color="steelblue")
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Brightness [ISO]", fontsize=14)
    ax.legend(["Actuals", "Predictions"], fontsize=14)

    plt.show()
    return None
