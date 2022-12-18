# -*- coding: utf-8 -*-

"""Module with handy functions for ISO prediction"""

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from constants import FIG_SIZE_16_9, SHAP_ALGORITHMS


def hello_mill(mill: str) -> str:
    """Print hello mill msg.

    Test function for package imports etc.

    :param mill: Name of mill to greet.
    :return: Greeting.
    """
    return (
        f"Hello {mill.capitalize()}, "
        f"we are about to optimize your bleaching sequence!"
    )


def apply_time_offset(
    data: pd.DataFrame,
    col_to_offset: str,
    offset_cols: List[str],
    time_granularity: Optional[str] = "1min",
) -> pd.DataFrame:
    """Apply time offset to data.

    :param data: Data with un-offset data.
    :param col_to_offset: Name of column with data to offset.
    :param offset_cols: Name of columns with offset times.
    :param time_granularity: Granularity of timestamps.
    :return: Offset data.
    """
    # Copy in data
    df = data.copy()
    # Calculate offset
    offset = pd.to_timedelta(
        df[offset_cols].sum(axis=1),
        unit="minutes",
    )
    # Separate target from features
    data_to_offset = df[col_to_offset]
    df = df.drop(columns=[col_to_offset])
    # Calculate target timestamp and merge
    df["offset_ts"] = (df.index + offset).dt.round(time_granularity)
    df = pd.merge(
        df,
        data_to_offset,
        left_on="offset_ts",
        right_index=True,
        how="inner",
    )
    # Select original columns & sort
    df = df[data.columns].sort_index()

    return df


def aggregate_data(
    data: pd.DataFrame,
    aggregation_freq: str,
    target_col: str = None,
    feature_cols: List[str] = None,
) -> pd.DataFrame:
    """Aggregate data on chosen time freuqency.

    Two methods available:
      - Target measurement cadence based ("target_driven")
      - Specified frequency (any frequency such as "1H", "5min", etc)

    :param data: Data to aggregate.
    :param aggregation_freq: Frequency for time series aggregation.
    :param target_col: Name of target column.
    :return: Aggregated data.
    """
    # Copy input data
    df = data.copy()

    if aggregation_freq == "target_driven":
        # Sort data
        df = df.sort_index()

        # Find rows where target value changes
        df["shifted_target"] = df[target_col].shift(1, fill_value=0)
        df["target_change_indicator"] = 0
        df.loc[df["shifted_target"] != df[target_col], "target_change_indicator"] = 1

        # Create aggregation groups: Cumulative sum from bottom and up
        df["aggregation_group"] = np.flip(
            np.cumsum(np.flip(df["target_change_indicator"].to_numpy())),
        )

        # Reset index in order to aggregate timestamp
        timestamp_col = df.index.name
        df = df.reset_index()

        # Define aggregations
        aggregations = {timestamp_col: "last"}
        aggregations.update({feature_col: "mean" for feature_col in feature_cols})
        aggregations.update({target_col: "last"})

        # Group by and aggregate
        df = (
            df.groupby("aggregation_group", as_index=False)
            .agg(aggregations)
            .drop(columns=["aggregation_group"])
        )

        # Reset to timestamp index
        df = df.set_index(timestamp_col).sort_index()

    else:
        # Standard resampling using specified frequency
        df = df.resample(aggregation_freq).mean().dropna()

    return df


def calculate_shap_values(
    model: Any,
    model_type: str,
    X_calibrate: pd.DataFrame,
    X_explain: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Calculate SHAP values for a pre-trained model given feature values.

    :param model: Trained model.
    :param model_type: Model type.
    :param X_calibrate: Feature values to use for calibrating SHAP.
    :param X_explain: Feature values to explain with SHAP values.
    :return: SHAP values.
    """
    # Calculate SHAP values
    shap_algorithm = SHAP_ALGORITHMS.get(model_type, None)
    if shap_algorithm is None:
        raise ValueError(f"SHAP algorithm {model_type} not supported.")
    explainer = {"linear": shap.explainers.Linear, "tree": shap.explainers.Tree,}[
        shap_algorithm
    ](model, X_calibrate)
    X_explain = X_calibrate if X_explain is None else X_explain
    shap_values = explainer.shap_values(X_explain)

    # Turn into pandas dataframe
    shap_values_df = pd.DataFrame(
        data=shap_values,
        columns=X_calibrate.columns,
        index=X_calibrate.index,
    )

    return shap_values_df


def create_shap_plots(
    shap_values: pd.DataFrame,
    feature_values: pd.DataFrame,
) -> Dict[str, plt.figure]:
    """Create SHAP summary plot and SHAP partial dependence plots for all
    feature variables from SHAP values.

    :param shap_values: SHAP values for all feature variables.
    :param feature_values: Values for all feature variables.
    :return: SHAP plots.
    """
    # Initiate dict with plots
    shap_plots: Dict = {}

    # Create new fig object
    fig, ax = plt.subplots(figsize=FIG_SIZE_16_9)
    # Summary plot

    shap.summary_plot(
        shap_values=shap_values.to_numpy(),
        feature_names=feature_values.columns,
    )
    # Add summary plot to dict with plots
    shap_plots.update({"summary_plot": fig})

    # Partial dependency plots
    for feature in feature_values.columns:
        # Create figure & axes
        fig, ax = plt.subplots(figsize=FIG_SIZE_16_9)
        # Draw partial dependence plot

        shap.dependence_plot(
            ind=feature,
            shap_values=shap_values.to_numpy(),
            features=feature_values,
            show=False,
            ax=ax,
            alpha=0.25,
        )
        # Add figure to dict with plots
        shap_plots.update({feature: fig})
    plt.show()
    return shap_plots
