"""
models.py
---------
Electricity price forecasting models:
  1. NaiveModel        — same-hour-last-week
  2. ARXModel          — linear regression on price lags + load
  3. NARXModel         — neural network on price lags + load
  4. HoltWintersModel  — triple exponential smoothing
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# 1. Naïve Model
# ---------------------------------------------------------------------------

class NaiveModel:
    """
    Naïve seasonal forecast: predict using the price from 168 hours (1 week) ago.

    Despite its simplicity, this is a strong baseline for electricity prices
    due to strong weekly demand patterns.
    """

    def fit(self, train: pd.Series) -> "NaiveModel":
        self._train = train
        return self

    def predict(self, test_index: pd.Index) -> np.ndarray:
        preds = []
        for ts in test_index:
            lag_ts = ts - pd.Timedelta(hours=168)
            if lag_ts in self._train.index:
                preds.append(self._train[lag_ts])
            else:
                preds.append(np.nan)
        return np.array(preds)


# ---------------------------------------------------------------------------
# 2. ARX Model (AutoRegressive with eXogenous variables)
# ---------------------------------------------------------------------------

class ARXModel:
    """
    Linear regression on lagged prices (24h, 48h, 168h) and load.

    Parameters
    ----------
    window : int or None
        Rolling training window in hours. None = fixed (full history).
    """

    def __init__(self, window: int | None = None):
        self.window = window
        self._model = LinearRegression()

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "ARXModel":
        if self.window:
            X_train = X_train.iloc[-self.window:]
            y_train = y_train.iloc[-self.window:]
        self._model.fit(X_train, y_train)
        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X_test)

    def rolling_predict(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> np.ndarray:
        """
        One-step-ahead rolling forecast: retrain on each new observation.
        """
        preds = []
        X_all = pd.concat([X_train, X_test])
        y_all = pd.concat([y_train, y_test])

        for i in range(len(X_test)):
            end_idx = len(X_train) + i
            start_idx = max(0, end_idx - self.window) if self.window else 0
            self._model.fit(X_all.iloc[start_idx:end_idx], y_all.iloc[start_idx:end_idx])
            preds.append(self._model.predict(X_test.iloc[[i]])[0])

        return np.array(preds)


# ---------------------------------------------------------------------------
# 3. NARX Model (Neural AutoRegressive with eXogenous variables)
# ---------------------------------------------------------------------------

class _NARXNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NARXModel:
    """
    Feedforward neural network with price lags and load as inputs.

    Parameters
    ----------
    hidden_dim : int   — neurons in first hidden layer
    epochs     : int   — training epochs
    lr         : float — learning rate
    """

    def __init__(self, hidden_dim: int = 64, epochs: int = 200, lr: float = 1e-3):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self._net = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "NARXModel":
        # Normalize
        self._x_mean, self._x_std = X_train.mean(0), X_train.std(0) + 1e-8
        self._y_mean, self._y_std = y_train.mean(), y_train.std() + 1e-8

        X_norm = (X_train - self._x_mean) / self._x_std
        y_norm = (y_train - self._y_mean) / self._y_std

        X_t = torch.tensor(X_norm, dtype=torch.float32)
        y_t = torch.tensor(y_norm, dtype=torch.float32)

        self._net = _NARXNet(X_train.shape[1], self.hidden_dim)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        dataset = TensorDataset(X_t, y_t)
        loader  = DataLoader(dataset, batch_size=64, shuffle=True)

        self._net.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self._net(xb), yb)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_norm = (X_test - self._x_mean) / self._x_std
        X_t = torch.tensor(X_norm, dtype=torch.float32)
        self._net.eval()
        with torch.no_grad():
            y_norm = self._net(X_t).numpy()
        return y_norm * self._y_std + self._y_mean


# ---------------------------------------------------------------------------
# 4. Holt-Winters Model
# ---------------------------------------------------------------------------

class HoltWintersModel:
    """
    Triple Exponential Smoothing with additive trend and multiplicative seasonality.

    Seasonal period = 168 hours (weekly cycle).
    Does NOT use exogenous variables — which limits accuracy during load spikes.
    """

    def __init__(self, seasonal_periods: int = 168):
        self.seasonal_periods = seasonal_periods
        self._model = None

    def fit(self, y_train: pd.Series) -> "HoltWintersModel":
        self._model = ExponentialSmoothing(
            y_train,
            trend="add",
            seasonal="mul",
            seasonal_periods=self.seasonal_periods,
            initialization_method="estimated",
        ).fit(optimized=True)
        return self

    def predict(self, steps: int) -> np.ndarray:
        return self._model.forecast(steps).values
