"""
features.py
-----------
Lag feature engineering for electricity price forecasting.

Creates autoregressive price lags and time-based features used by ARX and NARX models.
"""

import pandas as pd
import numpy as np


def add_lag_features(df: pd.DataFrame, price_col: str = "price") -> pd.DataFrame:
    """
    Add lagged price features to capture daily and weekly seasonality.

    Lags used:
      - 24h  : same hour yesterday
      - 48h  : same hour two days ago
      - 168h : same hour last week (strongest seasonal signal)

    Parameters
    ----------
    df : DataFrame with a datetime index and a price column
    price_col : name of the price column

    Returns
    -------
    DataFrame with new lag columns appended (rows with NaN lags are dropped)
    """
    df = df.copy()
    df[f"{price_col}_lag24"]  = df[price_col].shift(24)
    df[f"{price_col}_lag48"]  = df[price_col].shift(48)
    df[f"{price_col}_lag168"] = df[price_col].shift(168)
    return df.dropna()


def add_time_features(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    """
    Add calendar/time features useful for capturing intraday and weekly patterns.

    Added columns:
      - hour          : hour of day (0-23)
      - day_of_week   : 0=Monday … 6=Sunday
      - is_weekend    : binary flag
      - month         : month of year (1-12)
    """
    df = df.copy()
    dt = pd.to_datetime(df[datetime_col])
    df["hour"]        = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["month"]       = dt.dt.month
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    price_col: str = "price",
    load_col: str = "load",
    datetime_col: str = "datetime",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Full feature pipeline: lags + time features.

    Returns
    -------
    X : feature DataFrame
    y : target price series
    """
    df = add_time_features(df, datetime_col)
    df = add_lag_features(df, price_col)

    feature_cols = [
        f"{price_col}_lag24",
        f"{price_col}_lag48",
        f"{price_col}_lag168",
        load_col,
        "hour",
        "day_of_week",
        "is_weekend",
    ]
    X = df[feature_cols]
    y = df[price_col]
    return X, y


def train_test_split_temporal(
    df: pd.DataFrame, test_days: int = 7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/test preserving temporal order.
    Test set = last `test_days` days (default: 7, matching the paper).
    """
    test_hours = test_days * 24
    train = df.iloc[:-test_hours]
    test  = df.iloc[-test_hours:]
    return train, test
