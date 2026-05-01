"""
evaluate.py
-----------
Main evaluation pipeline for electricity price forecasting.

Runs all models, computes MAE/RMSE, and prints a results summary table.
Reproduces the RMSE evaluation from notebook Cell [126].

Usage:
    python src/evaluate.py --data data/ercot_north_2024.csv [--plot]
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import build_feature_matrix, train_test_split_temporal
from models import NaiveModel, ARXModel, NARXModel, HoltWintersModel
from combinations import combine_all


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def print_metrics(label: str, y_true, y_pred):
    print(f"  {label:<22} MAE: {mae(y_true, y_pred):7.4f}   RMSE: {rmse(y_true, y_pred):7.4f}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(data_path: str, plot: bool = False):
    # ── Load data ──────────────────────────────────────────────────────────
    df = pd.read_csv(data_path, parse_dates=["datetime"])
    df = df.sort_values("datetime").set_index("datetime")

    train_df, test_df = train_test_split_temporal(df, test_days=7)
    print(f"Train: {len(train_df)} hours | Test: {len(test_df)} hours\n")

    y_test = test_df["price"].values

    # ── Feature matrices ───────────────────────────────────────────────────
    X_train, y_train = build_feature_matrix(
        train_df.reset_index(), price_col="price", load_col="load", datetime_col="datetime"
    )
    X_test, _ = build_feature_matrix(
        test_df.reset_index(), price_col="price", load_col="load", datetime_col="datetime"
    )
    # Align test index to actual prices after lag creation
    y_test_aligned = test_df["price"].iloc[-len(X_test):].values

    # ── 1. Naïve ───────────────────────────────────────────────────────────
    naive = NaiveModel().fit(train_df["price"])
    naive_preds = naive.predict(test_df.index[-len(X_test):])

    # ── 2. ARX (fixed window) ──────────────────────────────────────────────
    arx_fixed = ARXModel(window=None).fit(X_train, y_train)
    arx_fixed_preds = arx_fixed.predict(X_test)

    # ── 3. ARX (rolling window) ────────────────────────────────────────────
    arx_rolling = ARXModel(window=168)
    arx_rolling_preds = arx_rolling.rolling_predict(X_train, y_train, X_test, pd.Series(y_test_aligned))

    # ── 4. NARX ────────────────────────────────────────────────────────────
    narx = NARXModel(epochs=200).fit(X_train.values, y_train.values)
    narx_preds = narx.predict(X_test.values)

    # ── 5. Holt-Winters ────────────────────────────────────────────────────
    hw = HoltWintersModel(seasonal_periods=168).fit(train_df["price"])
    hw_preds = hw.predict(len(y_test_aligned))

    # ── Forecast combinations ──────────────────────────────────────────────
    forecasts = {
        "naive": naive_preds,
        "arx":   arx_fixed_preds,
        "narx":  narx_preds,
        "hw":    hw_preds,
    }
    combos = combine_all(forecasts, actuals=y_test_aligned)

    # ── Print results ──────────────────────────────────────────────────────
    print("-" * 55)
    print("Root Mean Squared Error & MAE — Individual Models")
    print("-" * 55)
    print_metrics("Naïve",             y_test_aligned, naive_preds)
    print_metrics("ARX (fixed)",       y_test_aligned, arx_fixed_preds)
    print_metrics("ARX (rolling)",     y_test_aligned, arx_rolling_preds)
    print_metrics("NARX",              y_test_aligned, narx_preds)
    print_metrics("Holt-Winters",      y_test_aligned, hw_preds)
    print("-" * 55)
    print("Forecast Combinations")
    print("-" * 55)
    print_metrics("Simple Average",    y_test_aligned, combos["simple_avg"])
    print_metrics("Trimmed Mean",      y_test_aligned, combos["trimmed_mean"])
    print_metrics("Weighted Average",  y_test_aligned, combos["weighted_avg"])
    print("-" * 55)

    # ── Optional plot ──────────────────────────────────────────────────────
    if plot:
        _plot_forecasts(y_test_aligned, forecasts, combos)


def _plot_forecasts(actuals, forecasts, combos):
    import matplotlib.pyplot as plt

    hours = np.arange(len(actuals))
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(hours, actuals, "k-", lw=2, label="Actual", zorder=5)
    colors = {"naive": "blue", "arx": "orange", "narx": "green", "hw": "purple"}
    for name, preds in forecasts.items():
        ax.plot(hours, preds, "--o", color=colors[name], markersize=3, label=name.upper(), alpha=0.7)
    ax.plot(hours, combos["weighted_avg"], "-s", color="red", markersize=3,
            label="Weighted Avg", lw=1.5, alpha=0.9)

    ax.set_xlabel("Hour")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.set_title("ERCOT Price Forecasts vs Actual — Test Week")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("results/forecast_plot.png", dpi=150)
    print("\nPlot saved to results/forecast_plot.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Electricity Price Forecasting Evaluation")
    parser.add_argument("--data", default="data/ercot_north_2024.csv")
    parser.add_argument("--plot", action="store_true", help="Generate forecast plot")
    args = parser.parse_args()
    run_pipeline(args.data, plot=args.plot)
