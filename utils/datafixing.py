# -*- coding: utf-8 -*-

"""Module with data fixing functions"""

import pandas as pd
from functions import apply_time_offset
from typing import List, Dict

def get_data(file_path: str, cols: List[str], extra_offset: float = None) -> pd.DataFrame:
    """Get the right columns of data with timeoffset.

    This is done by loading the data from the file path,
    and filter out the columns/features to save and do timeoffset on. (s3_production, s3_residence_time_reactor and s3_brightness_out
    are entered by deafult)

    :param file_path: File path.
    :param cols: Features to get.
    :param extra_offset: Extra offset addition to the "s3_residence_time_reactor" data.
    :return: Data.
    """
    # load data
    df = pd.read_parquet(file_path).copy()
    
    # Make datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")

    # Hämtar alla columns som startar med s3 och s2_brightness_out och döper om den kolimnen.
    filter_col = [col for col in df if col in cols]

    # Sets default columns if not included
    if "s3_production" not in filter_col:
        filter_col.append("s3_production")
    if "s3_residence_time_reactor" not in filter_col:
        filter_col.append("s3_residence_time_reactor")

    s3 = df[filter_col]

    s3_answer_not_offset = df.get("s3_brightness_out")

    return time_offset(s3, s3_answer_not_offset, extra_offset = extra_offset)


def time_offset(
    s3: pd.DataFrame, 
    s3_answer_not_offset: pd.Series,
    extra_offset: float = None
) -> pd.DataFrame:
    """Adding a time offset to the data.

    :param s3_raw: Data to be time offset and the data for the time offset.
    :param s3_answer_not_offset: Answers to be time offset.
    :param extra_offset: Extra offset addition to the "s3_residence_time_reactor" data.
    :return: Data that have been time offset with answer and drop time offset data.
    """
    # Setting default value for extra_offset
    if extra_offset is None:
        extra_offset = 2.6

    # Makes dataframe with the s3_brightness_out variable and the time in reactor
    to_be_offset = pd.DataFrame()
    to_be_offset["ans"] = s3_answer_not_offset
    to_be_offset["time"] = s3["s3_residence_time_reactor"] + extra_offset

    # Offset the s3 answer with time in reactor
    complete_offset = apply_time_offset(to_be_offset, "ans", ["time"])
    s3["s3_brightness_out"] = complete_offset.get("ans")
    return s3


def clean_up(s3: pd.DataFrame, n_std: float):
    """Clean up nan values and get rid of outlairs

    :param s3_raw: Data to clean.
    :param n_std: How many standard diviations from the mean value is ok to not be classified as an outlier.
    :return: Data that have been cleand from outliers and nan values.
    """
    # Drops nan values
    s3 = s3.dropna()
    # Drops row when production is lower than 1000
    s3 = s3[(s3["s3_production"] > 1000)]
    s3 = s3.drop("s3_production", axis=1)

    # Drops every row with any value outside of mean +- n_std * std

    c = 0
    s3n = s3.copy()
    for col in s3.columns.to_list():

        std = s3[col].std()
        mean = s3[col].mean()
        s3n = s3n[
            (s3[col] <= mean + (n_std * std)) & (s3[col] >= mean - (n_std * std))
        ]
        c += 1
    return s3n


def dividing_data(
    s3: pd.DataFrame,
    training2: float,
    validation2: float,
    testing2: float,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Dividing data into three diffrent parts.

    :param s3: Data with both answer and input data.
    :param training2: Procent of trainging data for the division.
    :param validation2: Procent of validation data for the division.
    :param testing2: Procent of testing data for the division.
    :return: Dict with training, testing and validation data with both answers and input data.
    """
    if training2 + testing2 + validation2 != 1:
        print("training + testing + validation is not 1")
        return None

    s3_answer = s3.get("s3_brightness_out")
    A = s3.drop("s3_brightness_out", axis=1)

    # Fördelning utav datan
    training = training2
    validation = validation2 + training
    testing = testing2 + validation

    # Delar upp datan i olika mägnder
    size = A.shape[0]
    train = A[0 : round(size * training)]
    valid = A[round(size * training) : round(size * validation)]
    test = A[round(size * validation) : round(size * 1)]

    # Matchar datan med svaren.
    train_answer = s3_answer[0 : round(size * training)]
    valid_answer = s3_answer[round(size * training) : round(size * validation)]
    test_answer = s3_answer[round(size * validation) : round(size * 1)]

    s3new = {
        "training": {"data": train, "answer": train_answer},
        "validation": {"data": valid, "answer": valid_answer},
        "testing": {"data": test, "answer": test_answer},
    }

    return s3new
