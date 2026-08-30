# -*- coding: utf-8 -*-
"""ADA_forecast_Vvv.ipynb
"""

import math
import os
import random
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.arima.model import ARIMA

import lightgbm as lgb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)

# ================================================================
# ADAUSDT MULTISCALE HYBRID MULTI-HORIZON FORECASTING SYSTEM
# ================================================================
#
# INPUT
# -----
# ADAUSDT_multiscale_long.csv
#
# PERIOD
# ------
# 2022-08-27 -> 2026-08-27
#
# PRIMARY FORECAST HORIZONS
# -------------------------
# 1H   = short horizon
# 1W   = medium horizon
# 1M   = long horizon
#
# MODEL ARCHITECTURE
# ------------------
#
# 1. Leakage-safe multiscale feature builder
# 2. ARIMA expert
# 3. LightGBM expert
# 4. 1D-CNN AutoEncoder -> latent representation
# 5. ResNet18 image expert
# 6. Leakage-safe validation predictions
# 7. LightGBM stacking fusion
#
# TARGET
# ------
# future log return:
#
#   y_h(t) = log(P(t+h) / P(t))
#
# final price:
#
#   P_hat(t+h) = P(t) * exp(y_hat_h(t))
#
# ================================================================

from __future__ import annotations

import json
import math
import os
import random
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.arima.model import ARIMA

import lightgbm as lgb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)


# =====================
# GLOBAL CONFIGURATION

@dataclass
class Config:

    # Input

    data_path: str = ("C:\/Users\/Ali\/Documents\/github_alibavarchiee\/ADAXXX\/ADAUSDT_multiscale_long.csv")

    output_dir: str = (
        "C:\/Users\/Ali\/Documents\/github_alibavarchiee\/ADAXXX\/ada_hybrid_forecasting"
    )

    # time/date

    start_date: str = "2022-08-27"
    end_date: str = "2026-08-27"

    timezone: str = "UTC"

    # Prediction grid

    prediction_frequency: str = "1H"

    horizons: tuple = (
        "1H",
        "1W",
        "1M",
    )

    # Lookback

    sequence_length: int = 168

    # Technical features

    rolling_windows: tuple = (
        6,
        12,
        24,
        48,
        72,
        168,
    )

    # Dataset split
    train_end: str = "2025-06-30"

    validation_end: str = "2025-12-31"

    test_end: str = "2026-08-27"

    # AutoEncoder
    ae_latent_dim: int = 64

    ae_epochs: int = 20

    ae_batch_size: int = 128

    ae_learning_rate: float = 1e-3

    ae_weight_decay: float = 1e-5

    # ResNet
    resnet_epochs: int = 5

    resnet_batch_size: int = 32

    resnet_learning_rate: float = 2e-4

    resnet_weight_decay: float = 1e-4

    resnet_dropout: float = 0.25

    # To keep initial development practical.
    resnet_pretrained: bool = True

    image_size: int = 224

    # Image representation

    image_lookback: int = 168

    image_points: int = 128

    # LightGBM
    lgbm_estimators: int = 1500

    lgbm_learning_rate: float = 0.03

    lgbm_num_leaves: int = 63

    lgbm_max_depth: int = -1

    lgbm_min_child_samples: int = 40

    lgbm_subsample: float = 0.85

    lgbm_colsample_bytree: float = 0.85

    lgbm_reg_alpha: float = 0.1

    lgbm_reg_lambda: float = 0.5

    lgbm_early_stopping: int = 100

    # ARIMA

    arima_order: tuple = (
        2,
        1,
        2,
    )

    arima_window: int = 2000

    # Refit every N hours.
    # Smaller = more rigorous, but much slower.
    arima_refit_every: int = 24

    # Image/model training limits
    max_ae_train_samples: int = 50000

    max_resnet_train_samples: int = 15000

    max_resnet_validation_samples: int = 4000

    # Randomness
    random_seed: int = 313


CFG = Config()


# ===================
# DIRECTORIES

OUTPUT_DIR = Path(
    CFG.output_dir
)

MODELS_DIR = (
    OUTPUT_DIR
    / "models"
)

PREDICTIONS_DIR = (
    OUTPUT_DIR
    / "predictions"
)

REPORTS_DIR = (
    OUTPUT_DIR
    / "reports"
)

FEATURES_DIR = (
    OUTPUT_DIR
    / "features"
)

for directory in [
    OUTPUT_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    FEATURES_DIR,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# RANDOM SEEDS

def set_seed(
    seed: int
):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


set_seed(
    CFG.random_seed
)


# DEVICE: CPU Vs. GPU

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"PyTorch device: {DEVICE}"
)

if DEVICE.type == "cuda":

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


warnings.filterwarnings(
    "ignore"
)

# DATA LOADING

def load_long_dataset(
    path: str
) -> pd.DataFrame:

    print(
        "\n"
        "===========================================\n"
        "LOADING ADAUSDT MULTISCALE LONG DATASET\n"
        "==========================================="
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    df = pd.read_csv(
        path
    )

    if "timestamp" not in df.columns:

        raise ValueError(
            "Input dataset must contain 'timestamp'."
        )

    if "interval" not in df.columns:

        raise ValueError(
            "Input dataset must contain 'interval'."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "timestamp"
        ]
    )

    df = (
        df
        .sort_values(
            [
                "timestamp",
                "interval",
            ]
        )
        .drop_duplicates(
            [
                "timestamp",
                "interval",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    start = pd.Timestamp(
        CFG.start_date,
        tz="UTC",
    )

    end = pd.Timestamp(
        CFG.end_date,
        tz="UTC",
    ) + pd.Timedelta(
        days=1
    ) - pd.Timedelta(
        seconds=1
    )

    df = df[
        (df["timestamp"] >= start)
        &
        (df["timestamp"] <= end)
    ].copy()

    print(
        f"Shape: {df.shape}"
    )

    print(
        "\nIntervals:"
    )

    print(
        df["interval"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nDate range:"
    )

    print(
        df["timestamp"].min(),
        "->",
        df["timestamp"].max()
    )

    return df


# INTERVAL DURATIONS
INTERVAL_DURATION = {
    "1H": pd.Timedelta(hours=1),
    "4H": pd.Timedelta(hours=4),
    "12H": pd.Timedelta(hours=12),
    "1D": pd.Timedelta(days=1),
    "3D": pd.Timedelta(days=3),
    "1W": pd.Timedelta(days=7),
}


def interval_availability_delay(
    interval: str
):

    if interval == "1M":

        return pd.DateOffset(
            months=1
        )

    return INTERVAL_DURATION[
        interval
    ]


# CREATE CANONICAL HOURLY DATA
def build_hourly_base(
    long_df: pd.DataFrame
) -> pd.DataFrame:

    hourly = (
        long_df[
            long_df["interval"] == "1H"
        ]
        .copy()
        .sort_values(
            "timestamp"
        )
        .drop_duplicates(
            "timestamp"
    ))

    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "return",
        "log_return",
        "price_range",
        "price_range_pct",
        "direction",
    ]

    available = [
        c
        for c in required
        if c in hourly.columns
    ]

    hourly = hourly[
        available
    ].copy()

    hourly = (
        hourly
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    return hourly


# TECHNICAL FEATURE FUNCTIONS
def add_rsi(
    close: pd.Series,
    period: int = 14
) -> pd.Series:

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = (
        -delta.clip(
            upper=0
        )
    )

    avg_gain = (
        gain
        .rolling(
            period
        )
        .mean()
    )

    avg_loss = (
        loss
        .rolling(
            period
        )
        .mean()
    )

    rs = (
        avg_gain
        /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    return (
        100
        -
        (
            100
            /
            (1 + rs)
        )
    )


def add_macd(
    close: pd.Series
):

    ema12 = (
        close
        .ewm(
            span=12,
            adjust=False,
        )
        .mean()
    )

    ema26 = (
        close
        .ewm(
            span=26,
            adjust=False,
        )
        .mean()
    )

    macd = (
        ema12 - ema26
    )

    signal = (
        macd
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    histogram = (
        macd - signal
    )

    return (
        macd,
        signal,
        histogram
    )


def add_atr(
    df: pd.DataFrame,
    period: int = 14
):

    high = df["high"]

    low = df["low"]

    close = df["close"]

    tr1 = (
        high - low
    )

    tr2 = (
        high
        - close.shift(1)
    ).abs()

    tr3 = (
        low
        - close.shift(1)
    ).abs()

    tr = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1,
    ).max(
        axis=1
    )

    return (
        tr
        .rolling(
            period
        )
        .mean()
    )


# BASE FEATURE ENGINEERING
def engineer_hourly_features(
    hourly: pd.DataFrame
) -> pd.DataFrame:

    df = hourly.copy()

    close = df["close"]

    volume = df["volume"]

    # Lagged returns

    lag_hours = [
        1,
        2,
        3,
        6,
        12,
        24,
        48,
        72,
        168,
    ]

    for lag in lag_hours:

        df[
            f"return_lag_{lag}h"
        ] = (
            df["close"]
            .pct_change(
                lag
            )
        )

        df[
            f"log_return_lag_{lag}h"
        ] = (
            np.log(
                df["close"]
                /
                df["close"].shift(lag)
            )
        )

    # Rolling statistics

    for window in CFG.rolling_windows:

        roll = (
            close
            .rolling(
                window
            )
        )

        df[
            f"close_mean_{window}h"
        ] = roll.mean()

        df[
            f"close_std_{window}h"
        ] = roll.std()

        df[
            f"close_min_{window}h"
        ] = roll.min()

        df[
            f"close_max_{window}h"
        ] = roll.max()

        df[
            f"return_mean_{window}h"
        ] = (
            df["log_return"]
            .rolling(
                window
            )
            .mean()
        )

        df[
            f"return_std_{window}h"
        ] = (
            df["log_return"]
            .rolling(
                window
            )
            .std()
        )

        df[
            f"volume_mean_{window}h"
        ] = (
            volume
            .rolling(
                window
            )
            .mean()
        )

        df[
            f"volume_std_{window}h"
        ] = (
            volume
            .rolling(
                window
            )
            .std()
        )

        df[
            f"volume_z_{window}h"
        ] = (
            (
                volume
                -
                volume.rolling(
                    window
                ).mean()
            )
            /
            volume.rolling(
                window
            ).std()
        )

    # Moving averages

    for window in [
        6,
        12,
        24,
        48,
        72,
        168,
    ]:

        sma = (
            close
            .rolling(
                window
            )
            .mean()
        )

        ema = (
            close
            .ewm(
                span=window,
                adjust=False,
            )
            .mean()
        )

        df[
            f"sma_{window}"
        ] = sma

        df[
            f"ema_{window}"
        ] = ema

        df[
            f"close_sma_ratio_{window}"
        ] = (
            close / sma
        ) - 1.0

        df[
            f"close_ema_ratio_{window}"
        ] = (
            close / ema
        ) - 1.0

    # RSI

    df["rsi_14"] = add_rsi(
        close,
        14
    )

    # MACD

    (
        df["macd"],
        df["macd_signal"],
        df["macd_hist"],
    ) = add_macd(
        close
    )

    # ATR

    df["atr_14"] = add_atr(
        df,
        14
    )

    df["atr_pct"] = (
        df["atr_14"]
        / df["close"]
    )

    # Volume pressure

    if (
        "taker_buy_base_volume"
        in df.columns
    ):

        df[
            "taker_buy_ratio"
        ] = (
            df[
                "taker_buy_base_volume"
            ]
            /
            df["volume"].replace(
                0,
                np.nan
            )
        )

    if (
        "taker_buy_quote_volume"
        in df.columns
    ):

        df[
            "taker_buy_quote_ratio"
        ] = (
            df[
                "taker_buy_quote_volume"
            ]
            /
            df["quote_volume"].replace(
                0,
                np.nan
            )
        )

    # Calendar features

    df["hour"] = (
        df["timestamp"].dt.hour
    )

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["day_of_month"] = (
        df["timestamp"].dt.day
    )

    df["month"] = (
        df["timestamp"].dt.month
    )

    df["hour_sin"] = np.sin(
        2
        * np.pi
        * df["hour"]
        / 24.0
    )

    df["hour_cos"] = np.cos(
        2
        * np.pi
        * df["hour"]
        / 24.0
    )

    df["dow_sin"] = np.sin(
        2
        * np.pi
        * df["day_of_week"]
        / 7.0
    )

    df["dow_cos"] = np.cos(
        2
        * np.pi
        * df["day_of_week"]
        / 7.0
    )

    return df


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# | LEAKAGE-SAFE MULTISCALE FEATURE JOIN  |
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def get_completed_interval_features(
    long_df: pd.DataFrame,
    prediction_times: pd.Series,
    interval: str,
) -> pd.DataFrame:
    """
    For every prediction timestamp t, we use the MOST RECENT
    COMPLETED interval observation.

    e.g.:
        At 2025-01-10 14:00,
        a daily candle that closes at 23:00 must NOT be used.

    If interval starts at T and lasts 1D:
        available_time = T + 1D

    We therefore join using available_time <= prediction_time.
    """

    source = (
        long_df[
            long_df["interval"] == interval
        ]
        .copy()
        .sort_values(
            "timestamp"
        )
        .drop_duplicates(
            "timestamp"
    ))

    if source.empty:

        return pd.DataFrame(
            {
                "timestamp":
                    prediction_times
            }
        )

    if interval == "1M":

        source[
            "available_time"
        ] = (
            source["timestamp"]
            + pd.offsets.MonthBegin(
                1
            )
        )

    else:

        source[
            "available_time"
        ] = (
            source["timestamp"]
            +
            INTERVAL_DURATION[
                interval
            ]
        )

    # Rename feature columns
    numeric_features = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "return",
        "log_return",
        "price_range",
        "price_range_pct",
        "direction",
    ]

    numeric_features = [
        c
        for c in numeric_features
        if c in source.columns
    ]

    source = source[
        [
            "available_time"
        ]
        + numeric_features
    ].copy()

    source = source.rename(
        columns={
            c:
            f"{interval}_{c}"
            for c in numeric_features
        }
    )

    target = pd.DataFrame(
        {
            "timestamp":
                pd.to_datetime(
                    prediction_times,
                    utc=True,
                )
        }
    )

    source = source.sort_values(
        "available_time"
    )

    target = target.sort_values(
        "timestamp"
    )

    merged = pd.merge_asof(
        target,
        source,
        left_on="timestamp",
        right_on="available_time",
        direction="backward",
    )

    merged.drop(
        columns=[
            "available_time"
        ],
        inplace=True,
        errors="ignore",
    )

    return merged


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#     BUILD LEAKAGE-SAFE FEATURE MATRIX
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def build_multiscale_feature_matrix(
    long_df: pd.DataFrame,
    hourly_features: pd.DataFrame,
) -> pd.DataFrame:

    anchors = hourly_features[
        ["timestamp"]
    ].copy()

    result = anchors.copy()

    intervals = [
        "1H",
        "4H",
        "12H",
        "1D",
        "3D",
        "1W",
        "1M",
    ]

    # Multiscale completed-window joins

    for interval in intervals:

        print(
            f"Leakage-safe join: {interval}"
        )

        joined = (
            get_completed_interval_features(
                long_df,
                anchors["timestamp"],
                interval,
            )
        )

        result = result.merge(
            joined,
            on="timestamp",
            how="left",
        )

    # Hourly engineered features
    engineered_cols = [
        c
        for c in hourly_features.columns
        if c != "timestamp"
    ]

    result = result.merge(
        hourly_features[
            [
                "timestamp"
            ]
            + engineered_cols
        ],
        on="timestamp",
        how="left",
        suffixes=(
            "",
            "_hourly",
        ),
    )

    # Remove duplicate columns resulting from merge
    result = (
        result.loc[
            :,
            ~result.columns.duplicated()
        ]
    )

    return (
        result
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )


# TARGET CONSTRUCTION

def add_targets(
    feature_df: pd.DataFrame,
    hourly: pd.DataFrame,
) -> pd.DataFrame:

    result = feature_df.copy()

    prices = hourly[
        [
            "timestamp",
            "close",
        ]
    ].copy()

    prices = (
        prices
        .sort_values(
            "timestamp"
        )
        .drop_duplicates(
            "timestamp"
        )
    )

    # Current price

    result = result.merge(
        prices.rename(
            columns={
                "close":
                    "current_price"
            }
        ),
        on="timestamp",
        how="left",
    )

    # 1H target

    future_1h = prices.rename(
        columns={
            "timestamp":
                "target_time_1H",
            "close":
                "future_price_1H",
        }
    )

    future_1h[
        "target_time_1H"
    ] = (
        future_1h[
            "target_time_1H"
        ]
        - pd.Timedelta(
            hours=1
        )
    )

    result = result.merge(
        future_1h[
            [
                "target_time_1H",
                "future_price_1H",
            ]
        ],
        left_on="timestamp",
        right_on="target_time_1H",
        how="left",
    )

    result[
        "target_1H"
    ] = np.log(
        result[
            "future_price_1H"
        ]
        /
        result[
            "current_price"
        ]
    )

    # 1W target

    target_times_1w = (
        result["timestamp"]
        + pd.Timedelta(
            days=7
        )
    )

    target_1w = pd.DataFrame(
        {
            "target_timestamp":
                target_times_1w,
        }
    )

    target_1w = pd.merge_asof(
        target_1w.sort_values(
            "target_timestamp"
        ),
        prices.rename(
            columns={
                "timestamp":
                    "future_timestamp",
                "close":
                    "future_price_1W",
            }
        ).sort_values(
            "future_timestamp"
        ),
        left_on="target_timestamp",
        right_on="future_timestamp",
        direction="forward",
    )

    # Return to original row order
    target_1w.index = (
        result[
            "timestamp"
        ].index
    )

    result[
        "future_price_1W"
    ] = target_1w[
        "future_price_1W"
    ].values

    result[
        "target_1W"
    ] = np.log(
        result[
            "future_price_1W"
        ]
        /
        result[
            "current_price"
        ]
    )

    # 1M target

    target_times_1m = (
        result["timestamp"]
        + pd.DateOffset(
            months=1
        )
    )

    target_1m = pd.DataFrame(
        {
            "target_timestamp":
                target_times_1m,
        }
    )

    target_1m = pd.merge_asof(
        target_1m.sort_values(
            "target_timestamp"
        ),
        prices.rename(
            columns={
                "timestamp":
                    "future_timestamp",
                "close":
                    "future_price_1M",
            }
        ).sort_values(
            "future_timestamp"
        ),
        left_on="target_timestamp",
        right_on="future_timestamp",
        direction="forward",
    )

    target_1m.index = (
        result[
            "timestamp"
        ].index
    )

    result[
        "future_price_1M"
    ] = target_1m[
        "future_price_1M"
    ].values

    result[
        "target_1M"
    ] = np.log(
        result[
            "future_price_1M"
        ]
        /
        result[
            "current_price"
        ]
    )

    return result


# SPLIT DATA

def create_temporal_splits(
    df: pd.DataFrame
):

    train_end = pd.Timestamp(
        CFG.train_end,
        tz="UTC",
    ) + pd.Timedelta(
        days=1
    )

    valid_end = pd.Timestamp(
        CFG.validation_end,
        tz="UTC",
    ) + pd.Timedelta(
        days=1
    )

    test_end = pd.Timestamp(
        CFG.test_end,
        tz="UTC",
    ) + pd.Timedelta(
        days=1
    )

    train = df[
        df["timestamp"]
        < train_end
    ].copy()

    validation = df[
        (
            df["timestamp"]
            >= train_end
        )
        &
        (
            df["timestamp"]
            < valid_end
        )
    ].copy()

    test = df[
        (
            df["timestamp"]
            >= valid_end
        )
        &
        (
            df["timestamp"]
            < test_end
        )
    ].copy()

    print(
        "\n"
        "========================================================\n"
        "TEMPORAL SPLIT\n"
        "========================================================"
    )

    print(
        f"Train      : "
        f"{len(train):,}"
    )

    print(
        f"Validation : "
        f"{len(validation):,}"
    )

    print(
        f"Test       : "
        f"{len(test):,}"
    )

    print(
        "\nTrain:"
    )

    print(
        train[
            "timestamp"
        ].min(),
        "->",
        train[
            "timestamp"
        ].max(),
    )

    print(
        "\nValidation:"
    )

    print(
        validation[
            "timestamp"
        ].min(),
        "->",
        validation[
            "timestamp"
        ].max(),
    )

    print(
        "\nTest:"
    )

    print(
        test[
            "timestamp"
        ].min(),
        "->",
        test[
            "timestamp"
        ].max(),
    )

    return (
        train,
        validation,
        test,
    )


# FEATURE COLUMN SELECTION

TARGET_COLUMNS = [
    "target_1H",
    "target_1W",
    "target_1M",
    "future_price_1H",
    "future_price_1W",
    "future_price_1M",
]


META_COLUMNS = [
    "timestamp",
    "current_price",
]


def select_feature_columns(
    df: pd.DataFrame
):

    excluded = set(
        TARGET_COLUMNS
        + META_COLUMNS
        + [
            "target_time_1H",
        ]
    )

    columns = []

    for col in df.columns:

        if col in excluded:
            continue

        if (
            pd.api.types
            .is_numeric_dtype(
                df[col]
            )
        ):

            columns.append(
                col
            )

    return columns


# PREPARE TABULAR DATA

def prepare_tabular_data(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
):

    feature_columns = (
        select_feature_columns(
            train
        )
    )

    print(
        f"\nTabular features: "
        f"{len(feature_columns)}"
    )

    X_train = train[
        feature_columns
    ].copy()

    X_valid = validation[
        feature_columns
    ].copy()

    X_test = test[
        feature_columns
    ].copy()

    # Replace infinities
    for X in [
        X_train,
        X_valid,
        X_test,
    ]:

        X.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
            inplace=True,
        )

    # Median imputation *******
    #
    # Fit ONLY on training data.
    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imp = imputer.fit_transform(
        X_train
    )

    X_valid_imp = imputer.transform(
        X_valid
    )

    X_test_imp = imputer.transform(
        X_test
    )

    return (
        feature_columns,
        X_train_imp,
        X_valid_imp,
        X_test_imp,
        imputer,
    )


# LGBM FACTORY

def create_lgbm_model():

    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=CFG.lgbm_estimators,
        learning_rate=CFG.lgbm_learning_rate,
        num_leaves=CFG.lgbm_num_leaves,
        max_depth=CFG.lgbm_max_depth,
        min_child_samples=CFG.lgbm_min_child_samples,
        subsample=CFG.lgbm_subsample,
        colsample_bytree=CFG.lgbm_colsample_bytree,
        reg_alpha=CFG.lgbm_reg_alpha,
        reg_lambda=CFG.lgbm_reg_lambda,
        random_state=CFG.random_seed,
        n_jobs=-1,
    )


# TRAIN LGBM
def train_lgbm_models(
    train,
    validation,
    X_train,
    X_valid,
    feature_columns,
):

    models = {}

    predictions = pd.DataFrame(
        {
            "timestamp":
                validation[
                    "timestamp"
                ].values
        }
    )

    for horizon in [
        "1H",
        "1W",
        "1M",
    ]:

        target_column = (
            f"target_{horizon}"
        )

        print(
            "\n"
            f"Training LightGBM "
            f"{horizon}"
        )

        model = create_lgbm_model()

        y_train = (
            train[
                target_column
            ]
            .values
        )

        y_valid = (
            validation[
                target_column
            ]
            .values
        )

        valid_train = (
            np.isfinite(
                y_train
            )
        )

        valid_valid = (
            np.isfinite(
                y_valid
            )
        )

        model.fit(
            X_train[
                valid_train
            ],
            y_train[
                valid_train
            ],
            eval_set=[
                (
                    X_valid[
                        valid_valid
                    ],
                    y_valid[
                        valid_valid
                    ],
                )
            ],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(
                    CFG.lgbm_early_stopping,
                    verbose=False,
                ),
                lgb.log_evaluation(
                    100
                ),
            ],
        )

        pred = model.predict(
            X_valid
        )

        predictions[
            f"lgbm_{horizon}"
        ] = pred

        models[
            horizon
        ] = model

        joblib.dump(
            model,
            MODELS_DIR
            / f"lgbm_{horizon}.joblib",
        )

        print(
            f"Best iteration: "
            f"{model.best_iteration_}"
        )

    return (
        models,
        predictions,
    )


# AUTOENCODER Dataset
class SequenceDataset(
    Dataset
):

    def __init__(
        self,
        sequences
    ):

        self.x = torch.tensor(
            sequences,
            dtype=torch.float32,
        )

    def __len__(self):

        return len(
            self.x
        )

    def __getitem__(
        self,
        index
    ):

        return self.x[
            index
        ]


# AUTOENCODER (ANet)

class ConvAutoEncoder(
    nn.Module
):

    def __init__(
        self,
        n_features: int,
        latent_dim: int,
    ):

        super().__init__()

        self.encoder_conv = nn.Sequential(

            nn.Conv1d(
                n_features,
                64,
                kernel_size=5,
                padding=2,
            ),

            nn.BatchNorm1d(
                64
            ),

            nn.ReLU(),

            nn.Conv1d(
                64,
                128,
                kernel_size=5,
                padding=2,
            ),

            nn.BatchNorm1d(
                128
            ),

            nn.ReLU(),

            nn.Conv1d(
                128,
                128,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(
            1
        )

        self.encoder_fc = nn.Sequential(

            nn.Linear(
                128,
                128,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.1
            ),

            nn.Linear(
                128,
                latent_dim,
            ),
        )

        self.decoder_fc = nn.Sequential(

            nn.Linear(
                latent_dim,
                128,
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                128,
            ),

            nn.ReLU(),
        )

        self.decoder_conv = nn.Sequential(

            nn.Conv1d(
                128,
                64,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.Conv1d(
                64,
                n_features,
                kernel_size=5,
                padding=2,
            )
        )

    def encode(
        self,
        x
    ):

        # x:
        # batch x time x features

        x = x.transpose(
            1,
            2
        )

        x = self.encoder_conv(
            x
        )

        x = self.pool(
            x
        ).squeeze(
            -1
        )

        z = self.encoder_fc(
            x
        )

        return z

    def decode(
        self,
        z,
        sequence_length
    ):

        x = self.decoder_fc(
            z
        )

        x = x.unsqueeze(
            -1
        ).repeat(
            1,
            1,
            sequence_length
        )

        x = self.decoder_conv(
            x
        )

        return x.transpose(
            1,
            2
        )

    def forward(
        self,
        x
    ):

        z = self.encode(
            x
        )

        reconstruction = self.decode(
            z,
            x.shape[1],
        )

        return (
            reconstruction,
            z,
        )


# BUILD SEQUENCES
def build_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    scaler: StandardScaler | None = None,
):

    values = (
        df[
            feature_columns
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .ffill()
        .bfill()
        .fillna(0.0)
        .values
    )

    if scaler is None:

        scaler = StandardScaler()

        values = scaler.fit_transform(
            values
        )

    else:

        values = scaler.transform(
            values
        )

    sequences = []

    timestamps = []

    indices = []

    for i in range(
        sequence_length,
        len(df),
    ):

        window = values[
            i - sequence_length:
            i
        ]

        sequences.append(
            window
        )

        timestamps.append(
            df.iloc[
                i
            ]["timestamp"]
        )

        indices.append(
            i
        )

    sequences = np.asarray(
        sequences,
        dtype=np.float32,
    )

    return (
        sequences,
        timestamps,
        indices,
        scaler,
    )


# TRAIN autoencoder

def train_autoencoder(
    train_df: pd.DataFrame,
    feature_columns: list[str],
):

    print(
        "\n"
        "========================================================\n"
        "TRAINING AUTOENCODER\n"
        "========================================================"
    )

    scaler = StandardScaler()

    raw_values = (
        train_df[
            feature_columns
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .ffill()
        .bfill()
        .fillna(0.0)
        .values
    )

    scaled_values = scaler.fit_transform(
        raw_values
    )

    sequences = []

    for i in range(
        CFG.sequence_length,
        len(
            scaled_values
        ),
    ):

        sequences.append(
            scaled_values[
                i-CFG.sequence_length:
                i
            ]
        )

    sequences = np.asarray(
        sequences,
        dtype=np.float32,
    )

    if (
        len(sequences)
        > CFG.max_ae_train_samples
    ):

        rng = np.random.default_rng(
            CFG.random_seed
        )

        idx = rng.choice(
            len(sequences),
            size=CFG.max_ae_train_samples,
            replace=False,
        )

        sequences = sequences[
            np.sort(idx)
        ]

    dataset = SequenceDataset(
        sequences
    )

    loader = DataLoader(
        dataset,
        batch_size=CFG.ae_batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = ConvAutoEncoder(
        n_features=len(
            feature_columns
        ),
        latent_dim=CFG.ae_latent_dim,
    ).to(
        DEVICE
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.ae_learning_rate,
        weight_decay=CFG.ae_weight_decay,
    )

    loss_fn = nn.MSELoss()

    model.train()

    for epoch in range(
        1,
        CFG.ae_epochs + 1,
    ):

        running_loss = 0.0

        count = 0

        for batch in loader:

            batch = batch.to(
                DEVICE
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            reconstruction, _ = model(
                batch
            )

            loss = loss_fn(
                reconstruction,
                batch
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            running_loss += (
                loss.item()
                * len(batch)
            )

            count += len(batch)

        epoch_loss = (
            running_loss
            /
            max(
                1,
                count
            )
        )

        print(
            f"AE Epoch "
            f"{epoch:02d}/{CFG.ae_epochs} "
            f"Loss={epoch_loss:.8f}"
        )

    checkpoint = {
        "state_dict":
            model.state_dict(),
        "feature_columns":
            feature_columns,
        "scaler":
            scaler,
        "config":
            asdict(CFG),
    }

    torch.save(
        checkpoint,
        MODELS_DIR
        / "autoencoder.pt",
    )

    joblib.dump(
        scaler,
        MODELS_DIR
        / "autoencoder_scaler.joblib",
    )

    return (
        model,
        scaler,
    )


# EXTRACT AE LATENT FEATURES
def extract_ae_latents(
    model,
    scaler,
    df: pd.DataFrame,
    feature_columns: list[str],
):

    values = (
        df[
            feature_columns
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .ffill()
        .bfill()
        .fillna(0.0)
        .values
    )

    values = scaler.transform(
        values
    )

    latent_rows = []

    timestamps = []

    model.eval()

    with torch.no_grad():

        for i in range(
            CFG.sequence_length,
            len(df),
        ):

            window = values[
                i-CFG.sequence_length:
                i
            ]

            tensor = torch.tensor(
                window,
                dtype=torch.float32,
            ).unsqueeze(
                0
            ).to(
                DEVICE
            )

            z = model.encode(
                tensor
            )

            z = (
                z
                .squeeze(0)
                .cpu()
                .numpy()
            )

            latent_rows.append(
                z
            )

            timestamps.append(
                df.iloc[
                    i
                ]["timestamp"]
            )

    if not latent_rows:

        return pd.DataFrame()

    columns = [
        f"AE_z_{i+1:03d}"
        for i in range(
            CFG.ae_latent_dim
        )
    ]

    result = pd.DataFrame(
        latent_rows,
        columns=columns,
    )

    result.insert(
        0,
        "timestamp",
        timestamps,
    )

    return result


# GRAMIAN ANGULAR FIELD
def scale_minus_one_to_one(
    x: np.ndarray
) -> np.ndarray:

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x_min = np.nanmin(
        x
    )

    x_max = np.nanmax(
        x
    )

    if (
        not np.isfinite(
            x_min
        )
        or
        not np.isfinite(
            x_max
        )
    ):

        return np.zeros_like(
            x
        )

    if (
        x_max - x_min
        < 1e-12
    ):

        return np.zeros_like(
            x
        )

    scaled = (
        2
        *
        (
            x - x_min
        )
        /
        (
            x_max - x_min
        )
        - 1
    )

    return np.clip(
        scaled,
        -1,
        1,
    )


def gasf(
    series: np.ndarray
) -> np.ndarray:

    x = scale_minus_one_to_one(
        series
    )

    phi = np.arccos(
        np.clip(
            x,
            -1,
            1,
        )
    )

    return np.cos(
        phi[:, None]
        +
        phi[None, :]
    )


def gadf(
    series: np.ndarray
) -> np.ndarray:

    x = scale_minus_one_to_one(
        series
    )

    phi = np.arccos(
        np.clip(
            x,
            -1,
            1,
        )
    )

    return np.sin(
        phi[:, None]
        -
        phi[None, :]
    )


# RECURRENCE PLOT

def recurrence_plot(
    series: np.ndarray
) -> np.ndarray:

    x = scale_minus_one_to_one(
        series
    )

    distance = np.abs(
        x[:, None]
        -
        x[None, :]
    )

    # Normalize
    if np.max(
        distance
    ) > 0:

        distance = (
            distance
            /
            np.max(
                distance
            )
        )

    rp = np.exp(
        -5.0 * distance
    )

    return rp


# CREATE 3-CHANNEL MARKET IMAGE
def make_market_image(
    sequence: np.ndarray
):

    # sequence:
    # [time, features]
    #
    # channel source:
    #   0 -> close
    #   1 -> log_return
    #   2 -> volume

    if sequence.ndim != 2:

        raise ValueError(
            "Sequence must be 2D."
        )

    # ***Try named positions when possible***
    # The caller constructs the sequence in a known order.

    close = sequence[
        :, 0
    ]

    log_return = sequence[
        :, 1
    ]

    volume = sequence[
        :, 2
    ]

    # resize sequence to manageable dimension
    n = CFG.image_points

    def downsample(
        x
    ):

        if len(x) == n:

            return x

        positions = np.linspace(
            0,
            len(x) - 1,
            n
        )

        base = np.arange(
            len(x)
        )

        return np.interp(
            positions,
            base,
            x,
        )

    close = downsample(
        close
    )

    log_return = downsample(
        log_return
    )

    volume = downsample(
        volume
    )

    image = np.stack(
        [
            gasf(
                close
            ),
            gadf(
                log_return
            ),
            recurrence_plot(
                volume
            ),
        ],
        axis=0,
    )

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return image.astype(
        np.float32
    )


# RESNET dataset

class ResNetMarketDataset(
    Dataset
):

    def __init__(
        self,
        df: pd.DataFrame,
        sequence_length: int,
        feature_columns: list[str],
        max_samples: int | None = None,
    ):

        self.df = df.reset_index(
            drop=True
        )

        self.sequence_length = (
            sequence_length
        )

        self.feature_columns = (
            feature_columns
        )

        values = (
            self.df[
                feature_columns
            ]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .ffill()
            .bfill()
            .fillna(0.0)
            .values
        )

        self.values = values.astype(
            np.float32
        )

        self.indices = np.arange(
            sequence_length,
            len(self.df)
        )

        if (
            max_samples is not None
            and
            len(self.indices)
            > max_samples
        ):

            rng = np.random.default_rng(
                CFG.random_seed
            )

            chosen = rng.choice(
                self.indices,
                size=max_samples,
                replace=False,
            )

            self.indices = np.sort(
                chosen
            )

    def __len__(
        self
    ):

        return len(
            self.indices
        )

    def __getitem__(
        self,
        index
    ):

        i = int(
            self.indices[
                index
            ]
        )

        sequence = self.values[
            i-self.sequence_length:
            i
        ]

        image = make_market_image(
            sequence
        )

        targets = np.array(
            [
                self.df.iloc[
                    i
                ]["target_1H"],
                self.df.iloc[
                    i
                ]["target_1W"],
                self.df.iloc[
                    i
                ]["target_1M"],
            ],
            dtype=np.float32,
        )

        return (
            torch.tensor(
                image,
                dtype=torch.float32,
            ),
            torch.tensor(
                targets,
                dtype=torch.float32,
            ),
        )


# RESNET18 MULTI-HEAD REGRESSOR
class ResNet18MultiHorizon(
    nn.Module
):

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.25,
    ):

        super().__init__()

        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = resnet18(
            weights=weights
        )

        in_features = (
            self.backbone.fc.in_features
        )

        self.backbone.fc = nn.Identity()

        self.head = nn.Sequential(

            nn.Linear(
                in_features,
                256,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                256,
                3,
            ),
        )

    def forward(
        self,
        x
    ):
        # Normalize each image channel independently:
        mean = x.mean(
            dim=(2, 3),
            keepdim=True
        )

        std = (
            x.std(
                dim=(2, 3),
                keepdim=True
            )
            + 1e-6
        )

        x = (
            x - mean
        ) / std

        # ResNet pretrained weights expect 224 or larger.
        x = F.interpolate(
            x,
            size=(
                224,
                224
            ),
            mode="bilinear",
            align_corners=False,
        )

        features = self.backbone(
            x
        )

        return self.head(
            features
        )


# TRAIN RESNET

def train_resnet(
    train_df,
    validation_df,
    feature_columns,
):

    print(
        "\n"
        "========================================================\n"
        "TRAINING RESNET18\n"
        "========================================================"
    )

    train_dataset = (
        ResNetMarketDataset(
            train_df,
            CFG.image_lookback,
            feature_columns,
            max_samples=CFG.max_resnet_train_samples,
        )
    )

    valid_dataset = (
        ResNetMarketDataset(
            validation_df,
            CFG.image_lookback,
            feature_columns,
            max_samples=CFG.max_resnet_validation_samples,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.resnet_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.resnet_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    model = (
        ResNet18MultiHorizon(
            pretrained=CFG.resnet_pretrained,
            dropout=CFG.resnet_dropout,
        )
        .to(DEVICE)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.resnet_learning_rate,
        weight_decay=CFG.resnet_weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CFG.resnet_epochs,
    )

    criterion = nn.HuberLoss()

    best_val = np.inf

    best_state = None

    for epoch in range(
        1,
        CFG.resnet_epochs + 1,
    ):
        # Training is going to be

        model.train()

        train_loss = 0.0

        train_count = 0

        for images, targets in train_loader:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            targets = targets.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                targets
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            train_loss += (
                loss.item()
                * len(images)
            )

            train_count += len(
                images
            )

        train_loss /= max(
            1,
            train_count
        )

        # Validation

        model.eval()

        val_loss = 0.0

        val_count = 0

        with torch.no_grad():

            for images, targets in valid_loader:

                images = images.to(
                    DEVICE,
                    non_blocking=True,
                )

                targets = targets.to(
                    DEVICE,
                    non_blocking=True,
                )

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    targets
                )

                val_loss += (
                    loss.item()
                    * len(images)
                )

                val_count += len(
                    images
                )

        val_loss /= max(
            1,
            val_count
        )

        scheduler.step()

        print(
            f"Epoch "
            f"{epoch:02d}/{CFG.resnet_epochs} "
            f"train={train_loss:.8f} "
            f"val={val_loss:.8f}"
        )

        if val_loss < best_val:

            best_val = val_loss

            best_state = {
                k: v.detach().cpu().clone()
                for k, v
                in model.state_dict().items()
            }

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    torch.save(
        {
            "state_dict":
                model.state_dict(),
            "feature_columns":
                feature_columns,
            "config":
                asdict(CFG),
        },
        MODELS_DIR
        / "resnet18_multihorizon.pt",
    )

    return model

# RESNET PREDICTIONS
def predict_resnet(
    model,
    df,
    feature_columns,
):

    dataset = (
        ResNetMarketDataset(
            df,
            CFG.image_lookback,
            feature_columns,
            max_samples=None,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=CFG.resnet_batch_size,
        shuffle=False,
        num_workers=0,
    )

    predictions = []

    model.eval()

    with torch.no_grad():

        for images, _ in loader:

            images = images.to(
                DEVICE
            )

            output = model(
                images
            )

            predictions.append(
                output.cpu().numpy()
            )

    if not predictions:

        return pd.DataFrame(
            columns=[
                "timestamp",
                "resnet_1H",
                "resnet_1W",
                "resnet_1M",
            ]
        )

    predictions = np.vstack(
        predictions
    )

    timestamps = (
        df
        .iloc[
            dataset.indices
        ]["timestamp"]
        .values
    )

    result = pd.DataFrame(
        {
            "timestamp":
                timestamps,
            "resnet_1H":
                predictions[:, 0],
            "resnet_1W":
                predictions[:, 1],
            "resnet_1M":
                predictions[:, 2],
        }
    )

    return result


# ARIMA ROLLING EXPERT
def rolling_arima_predictions(
    train_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
):

    print(
        "\n"
        "========================================================\n"
        "ARIMA EXPERT\n"
        "========================================================"
    )

    hourly_train = train_df[
        [
            "timestamp",
            "close",
        ]
    ].copy()

    evaluation_times = (
        evaluation_df[
            "timestamp"
        ]
        .sort_values()
        .reset_index(
            drop=True
        )
    )

    full_prices = pd.concat(
        [
            hourly_train,
            evaluation_df[
                [
                    "timestamp",
                    "current_price",
                ]
            ].rename(
                columns={
                    "current_price":
                        "close"
                }
            ),
        ],
        ignore_index=True,
    )

    full_prices = (
        full_prices
        .drop_duplicates(
            "timestamp"
        )
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    log_price = np.log(
        full_prices[
            "close"
        ].astype(float)
    )

    timestamp_to_position = {
        ts: i
        for i, ts
        in enumerate(
            full_prices[
                "timestamp"
            ]
        )
    }

    output = {
        "timestamp": [],
        "arima_1H": [],
        "arima_1W": [],
        "arima_1M": [],
    }

    last_predictions = {
        "1H": np.nan,
        "1W": np.nan,
        "1M": np.nan,
    }

    counter = 0

    for timestamp in evaluation_times:

        if (
            counter
#             %
            CFG.arima_refit_every
            == 0
        ):

            position = timestamp_to_position.get(
                timestamp
            )

            if position is None:

                counter += 1
                continue

            start = max(
                0,
                position
                - CFG.arima_window
                + 1,
            )

            series = (
                log_price
                .iloc[
                    start:
                    position + 1
                ]
                .replace(
                    [
                        np.inf,
                        -np.inf,
                    ],
                    np.nan,
                )
                .dropna()
            )

            try:

                model = ARIMA(
                    series,
                    order=CFG.arima_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )

                fitted = model.fit()

                max_steps = 24 * 31

                forecast = (
                    fitted
                    .forecast(
                        steps=max_steps
                    )
                    .values
                )

                current_price = (
                    full_prices
                    .iloc[
                        position
                    ]["close"]
                )

                # 1H
                last_predictions[
                    "1H"
                ] = (
                    forecast[0]
                    - series.iloc[-1]
                )

                # 1W
                if len(
                    forecast
                ) >= 168:

                    last_predictions[
                        "1W"
                    ] = (
                        forecast[167]
                        - series.iloc[-1]
                    )

                # 1M
                if len(
                    forecast
                ) >= 720:

                    last_predictions[
                        "1M"
                    ] = (
                        forecast[719]
                        - series.iloc[-1]
                    )

            except Exception as exc:

                print(
                    f"ARIMA warning at "
                    f"{timestamp}: {exc}"
                )

        output[
            "timestamp"
        ].append(
            timestamp
        )

        output[
            "arima_1H"
        ].append(
            last_predictions["1H"]
        )

        output[
            "arima_1W"
        ].append(
            last_predictions["1W"]
        )

        output[
            "arima_1M"
        ].append(
            last_predictions["1M"]
        )

        counter += 1

    return pd.DataFrame(
        output
    )


# FUSION FEATURES
def build_fusion_training_data(
    validation_df,
    lgbm_validation,
    resnet_validation,
    arima_validation,
):

    result = validation_df[
        [
            "timestamp",
            "current_price",
            "target_1H",
            "target_1W",
            "target_1M",
        ]
    ].copy()

    for frame in [
        lgbm_validation,
        resnet_validation,
        arima_validation,
    ]:

        result = result.merge(
            frame,
            on="timestamp",
            how="left",
        )

    return result


# ********************
# TRAIN FUSION MODELS
# ********************

def train_fusion_models(
    fusion_train: pd.DataFrame
):

    print(
        "\n"
        "========================================================\n"
        "TRAINING FUSION MODELS\n"
        "========================================================"
    )

    models = {}

    predictions = (
        fusion_train[
            [
                "timestamp"
            ]
        ]
        .copy()
    )

    for horizon in [
        "1H",
        "1W",
        "1M",
    ]:

        target = (
            f"target_{horizon}"
        )

        prediction_features = [
            f"arima_{horizon}",
            f"lgbm_{horizon}",
            f"resnet_{horizon}",
        ]

        # AE information can be added if desired.
        available = [
            c
            for c in prediction_features
            if c in fusion_train.columns
        ]

        X = (
            fusion_train[
                available
            ]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
        )

        y = (
            fusion_train[
                target
            ]
        )

        valid = (
            y.notna()
        )

        X = X.loc[
            valid
        ]

        y = y.loc[
            valid
        ]

        imputer = SimpleImputer(
            strategy="median"
        )

        X_imp = imputer.fit_transform(
            X
        )

        # Fusion is trained on historical out-of-sample
        # predictions.
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=30,
            reg_alpha=0.2,
            reg_lambda=1.0,
            random_state=CFG.random_seed,
            n_jobs=-1,
        )

        model.fit(
            X_imp,
            y.values,
        )

        models[
            horizon
        ] = {
            "model":
                model,
            "imputer":
                imputer,
            "features":
                available,
        }

        model_path = (
            MODELS_DIR
            / f"fusion_{horizon}.joblib"
        )

        joblib.dump(
            models[horizon],
            model_path,
        )

        # In-sample meta prediction is only used for
        # inspecting the fusion training set.
        predictions[
            f"fusion_{horizon}"
        ] = model.predict(
            X_imp
        )

    return (
        models,
        predictions,
    )


# Apply fusion

def apply_fusion_models(
    models,
    prediction_df,
):

    result = prediction_df[
        [
            "timestamp"
        ]
    ].copy()

    for horizon in [
        "1H",
        "1W",
        "1M",
    ]:

        bundle = models[
            horizon
        ]

        model = bundle[
            "model"
        ]

        imputer = bundle[
            "imputer"
        ]

        features = bundle[
            "features"
        ]

        X = prediction_df[
            features
        ].copy()

        X = X.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        X = imputer.transform(
            X
        )

        result[
            f"fusion_{horizon}"
        ] = model.predict(
            X
        )

    return result


# METRICS

def calculate_metrics(
    actual,
    predicted,
):

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    mask = (
        np.isfinite(
            actual
        )
        &
        np.isfinite(
            predicted
        )
    )

    actual = actual[
        mask
    ]

    predicted = predicted[
        mask
    ]

    if len(actual) == 0:

        return {
            "n": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "Directional_Accuracy": np.nan,
            "R2": np.nan,
        }

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = math.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    denominator = np.maximum(
        np.abs(actual),
        1e-8,
    )

    mape = np.mean(
        np.abs(
            (
                actual
                - predicted
            )
            /
            denominator
        )
    ) * 100

    direction = np.mean(
        np.sign(
            actual
        )
        ==
        np.sign(
            predicted
        )
    ) * 100

    r2 = r2_score(
        actual,
        predicted,
    )

    return {
        "n":
            int(len(actual)),
        "MAE":
            float(mae),
        "RMSE":
            float(rmse),
        "MAPE":
            float(mape),
        "Directional_Accuracy":
            float(direction),
        "R2":
            float(r2),
    }


# EVALUATION of MODELS

def evaluate_predictions(
    actual_df,
    prediction_df,
):

    merged = actual_df[
        [
            "timestamp",
            "current_price",
            "target_1H",
            "target_1W",
            "target_1M",
        ]
    ].merge(
        prediction_df,
        on="timestamp",
        how="inner",
    )

    rows = []

    model_prefixes = [
        "arima",
        "lgbm",
        "resnet",
        "fusion",
    ]

    for model_name in model_prefixes:

        for horizon in [
            "1H",
            "1W",
            "1M",
        ]:

            pred_col = (
                f"{model_name}_{horizon}"
            )

            target_col = (
                f"target_{horizon}"
            )

            if (
                pred_col
                not in merged.columns
            ):

                continue

            metrics = calculate_metrics(
                merged[
                    target_col
                ],
                merged[
                    pred_col
                ],
            )

            metrics.update(
                {
                    "model":
                        model_name,
                    "horizon":
                        horizon,
                }
            )

            rows.append(
                metrics
            )

    return pd.DataFrame(
        rows
    )


# PRICE RECONSTRUCTION

def reconstruct_prices(
    prediction_df: pd.DataFrame,
    base_df: pd.DataFrame,
):

    result = base_df[
        [
            "timestamp",
            "current_price",
        ]
    ].merge(
        prediction_df,
        on="timestamp",
        how="left",
    )

    for horizon in [
        "1H",
        "1W",
        "1M",
    ]:

        fusion_col = (
            f"fusion_{horizon}"
        )

        if fusion_col not in result.columns:

            continue

        result[
            f"predicted_price_{horizon}"
        ] = (
            result["current_price"]
            *
            np.exp(
                result[
                    fusion_col
                ]
            )
        )

        result[
            f"predicted_direction_{horizon}"
        ] = np.where(
            result[
                fusion_col
            ] >= 0,
            1,
            -1,
        )

    return result


# BASELINE MODELS

def naive_predictions(
    df: pd.DataFrame
):

    result = df[
        [
            "timestamp"
        ]
    ].copy()

    for horizon in [
        "1H",
        "1W",
        "1M",
    ]:

        result[
            f"naive_{horizon}"
        ] = 0.0

    return result


# CREATE REPORT

def save_reports(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    final_predictions: pd.DataFrame,
):

    metrics_file = (
        REPORTS_DIR
        / "model_metrics.csv"
    )

    predictions_file = (
        PREDICTIONS_DIR
        / "test_predictions.csv"
    )

    final_file = (
        PREDICTIONS_DIR
        / "final_fused_forecasts.csv"
    )

    metrics.to_csv(
        metrics_file,
        index=False,
    )

    predictions.to_csv(
        predictions_file,
        index=False,
    )

    final_predictions.to_csv(
        final_file,
        index=False,
    )

    print(
        "\nReports saved:"
    )

    print(
        metrics_file
    )

    print(
        predictions_file
    )

    print(
        final_file
    )


# SAVE CONFIG.

def save_configuration():

    config_file = (
        OUTPUT_DIR
        / "config.json"
    )

    with open(
        config_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            asdict(CFG),
            f,
            indent=4,
            default=str,
        )

    return config_file


# MAIN PIPELINE

def run_pipeline():

    print(
        "\n"
        "================================================================\n"
        " ADAUSDT MULTISCALE HYBRID MULTI-HORIZON FORECASTING SYSTEM\n"
        "================================================================"
    )

    print(
        f"\nInput:\n{CFG.data_path}"
    )

    print(
        f"\nOutput:\n{OUTPUT_DIR}"
    )

    print(
        f"\nDevice:\n{DEVICE}"
    )

    # STEP 1

    long_df = load_long_dataset(
        CFG.data_path
    )

    # STEP 2
    # Canonical 1H

    hourly = build_hourly_base(
        long_df
    )

    print(
        f"\nCanonical hourly rows: "
        f"{len(hourly):,}"
    )

    # STEP 3
    # Engineering

    hourly_features = (
        engineer_hourly_features(
            hourly
        )
    )

    # STEP 4
    # Leakage-safe multiscale features

    feature_df = (
        build_multiscale_feature_matrix(
            long_df,
            hourly_features,
        )
    )

    # STEP 5
    # Targets

    modeling_df = add_targets(
        feature_df,
        hourly,
    )

    # STEP 6
    # Remove rows without current price

    modeling_df = modeling_df[
        modeling_df[
            "current_price"
        ].notna()
    ].copy()

    modeling_df = (
        modeling_df
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    # Save feature dataframe
    modeling_df.to_parquet(
        FEATURES_DIR
        / "ada_multiscale_modeling.parquet",
        index=False,
    )

    modeling_df.to_csv(
        FEATURES_DIR
        / "ada_multiscale_modeling.csv",
        index=False,
    )

    # STEP 7
    # Temporal split

    train_df, validation_df, test_df = (
        create_temporal_splits(
            modeling_df
        )
    )

    # STEP 8
    # Tabular features

    (
        feature_columns,
        X_train,
        X_valid,
        X_test,
        tabular_imputer,
    ) = prepare_tabular_data(
        train_df,
        validation_df,
        test_df,
    )

    joblib.dump(
        tabular_imputer,
        MODELS_DIR
        / "tabular_imputer.joblib",
    )

    with open(
        FEATURES_DIR
        / "tabular_feature_columns.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            feature_columns,
            f,
            indent=2,
        )

    # STEP 9
    # LIGHTGBM

    (
        lgbm_models,
        lgbm_validation,
    ) = train_lgbm_models(
        train_df,
        validation_df,
        X_train,
        X_valid,
        feature_columns,
    )

    # STEP 10
    # AUTOENCODER

    ae_model, ae_scaler = (
        train_autoencoder(
            train_df,
            feature_columns,
        )
    )

    # Extract latent features for validation/test.
    #
    # NOTE:
    # AE was trained only on train data.
    #

    ae_train_latent = (
        extract_ae_latents(
            ae_model,
            ae_scaler,
            train_df,
            feature_columns,
        )
    )

    ae_valid_latent = (
        extract_ae_latents(
            ae_model,
            ae_scaler,
            validation_df,
            feature_columns,
        )
    )

    ae_test_latent = (
        extract_ae_latents(
            ae_model,
            ae_scaler,
            test_df,
            feature_columns,
        )
    )

    # Save latent features.
    ae_train_latent.to_csv(
        FEATURES_DIR
        / "AE_latent_train.csv",
        index=False,
    )

    ae_valid_latent.to_csv(
        FEATURES_DIR
        / "AE_latent_validation.csv",
        index=False,
    )

    ae_test_latent.to_csv(
        FEATURES_DIR
        / "AE_latent_test.csv",
        index=False,
    )

    # STEP 11
    # Add AE latent variables to LightGBM
    #
    # Second-stage LightGBM with AE representations

    print(
        "\n"
        "========================================================\n"
        "LIGHTGBM + AUTOENCODER LATENT FEATURES\n"
        "========================================================"
    )

    # Merge AE latents
    train_ae = train_df.merge(
        ae_train_latent,
        on="timestamp",
        how="left",
    )

    valid_ae = validation_df.merge(
        ae_valid_latent,
        on="timestamp",
        how="left",
    )

    test_ae = test_df.merge(
        ae_test_latent,
        on="timestamp",
        how="left",
    )

    ae_columns = [
        c
        for c in ae_train_latent.columns
        if c != "timestamp"
    ]

    ae_lgbm_features = (
        feature_columns
        + ae_columns
    )

    # Remove possible duplicates
    ae_lgbm_features = list(
        dict.fromkeys(
            ae_lgbm_features
        )
    )

    X_train_ae = (
        train_ae[
            ae_lgbm_features
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    X_valid_ae = (
        valid_ae[
            ae_lgbm_features
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    ae_imputer = SimpleImputer(
        strategy="median"
    )

    X_train_ae = (
        ae_imputer
        .fit_transform(
            X_train_ae
        )
    )

    X_valid_ae = (
        ae_imputer
        .transform(
            X_valid_ae
        )
    )

    joblib.dump(
        ae_imputer,
        MODELS_DIR
        / "ae_lgbm_imputer.joblib",
    )

    ae_lgbm_models = {}

    for horizon in [
        "1H",
        "1W",
        "1M",
    ]:

        model = (
            create_lgbm_model()
        )

        y_train = (
            train_ae[
                f"target_{horizon}"
            ].values
        )

        y_valid = (
            valid_ae[
                f"target_{horizon}"
            ].values
        )

        mask_train = np.isfinite(
            y_train
        )

        mask_valid = np.isfinite(
            y_valid
        )

        model.fit(
            X_train_ae[
                mask_train
            ],
            y_train[
                mask_train
            ],
            eval_set=[
                (
                    X_valid_ae[
                        mask_valid
                    ],
                    y_valid[
                        mask_valid
                    ],
                )
            ],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(
                    CFG.lgbm_early_stopping,
                    verbose=False,
                ),
            ],
        )

        ae_lgbm_models[
            horizon
        ] = model

        valid_ae[
            f"ae_lgbm_{horizon}"
        ] = model.predict(
            X_valid_ae
        )

        joblib.dump(
            model,
            MODELS_DIR
            / f"ae_lgbm_{horizon}.joblib",
        )

    ae_validation_predictions = (
        valid_ae[
            [
                "timestamp",
                "ae_lgbm_1H",
                "ae_lgbm_1W",
                "ae_lgbm_1M",
            ]
        ].copy()
    )

    # STEP 12
    # RESNET18

    resnet_model = train_resnet(
        train_df,
        validation_df,
        feature_columns=[
            # Must put close, log_return, volume first because
            # make_market_image expects these positions.
            "1H_close",
            "1H_log_return",
            "1H_volume",
        ],
    )

    resnet_validation = (
        predict_resnet(
            resnet_model,
            validation_df,
            [
                "1H_close",
                "1H_log_return",
                "1H_volume",
            ],
        )
    )

    # STEP 13
    # ARIMA on validation

    arima_validation = (
        rolling_arima_predictions(
            train_df,
            validation_df,
        )
    )

    # STEP 14
    # FUSION TRAINING

    fusion_validation = (
        build_fusion_training_data(
            validation_df,
            lgbm_validation,
            resnet_validation,
            arima_validation,
        )
    )

    # Add AE-LightGBM as another expert
    fusion_validation = (
        fusion_validation.merge(
            ae_validation_predictions,
            on="timestamp",
            how="left",
        )
    )

    # The fusion function originally expects lgbm/resnet/arima.
    # Add AE expert and train custom extended fusion below.

    fusion_models = {}

    for horizon in [
        "1H",
        "1W",
        "1M",
    ]:

        feature_candidates = [
            f"arima_{horizon}",
            f"lgbm_{horizon}",
            f"resnet_{horizon}",
            f"ae_lgbm_{horizon}",
        ]

        fusion_features = [
            c
            for c in feature_candidates
            if c in fusion_validation.columns
        ]

        X = (
            fusion_validation[
                fusion_features
            ]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
        )

        y = (
            fusion_validation[
                f"target_{horizon}"
            ]
        )

        valid = (
            y.notna()
        )

        imputer = SimpleImputer(
            strategy="median"
        )

        X_imp = imputer.fit_transform(
            X
        )

        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=800,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=30,
            reg_alpha=0.2,
            reg_lambda=1.0,
            random_state=CFG.random_seed,
            n_jobs=-1,
        )

        model.fit(
            X_imp[
                valid.values
            ],
            y.loc[
                valid
            ].values,
        )

        fusion_models[
            horizon
        ] = {
            "model":
                model,
            "imputer":
                imputer,
            "features":
                fusion_features,
        }

        joblib.dump(
            fusion_models[
                horizon
            ],
            MODELS_DIR
            / f"extended_fusion_{horizon}.joblib",
        )

    # STEP 15
    # TEST BASE MODELS
    #
    # Retraining base models on TRAIN + VALIDATION is required
    # before final TEST prediction.

    development_df = pd.concat(
        [
            train_df,
            validation_df,
        ],
        ignore_index=True,
    )

    development_df = (
        development_df
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    X_dev = (
        development_df[
            feature_columns
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    X_test = (
        test_df[
            feature_columns
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    dev_imputer = SimpleImputer(
        strategy="median"
    )

    X_dev = (
        dev_imputer
        .fit_transform(
            X_dev
        )
    )

    X_test = (
        dev_imputer
        .transform(
            X_test
        )
    )

    development_lgbm = {}

    lgbm_test_predictions = (
        test_df[
            [
                "timestamp"
            ]
        ].copy()
    )

    for horizon in [
        "1H",
        "1W",
        "1M",
    ]:

        model = create_lgbm_model()

        y_dev = (
            development_df[
                f"target_{horizon}"
            ].values
        )

        valid_dev = np.isfinite(
            y_dev
        )

        model.fit(
            X_dev[
                valid_dev
            ],
            y_dev[
                valid_dev
            ],
        )

        development_lgbm[
            horizon
        ] = model

        lgbm_test_predictions[
            f"lgbm_{horizon}"
        ] = model.predict(
            X_test
        )

        joblib.dump(
            model,
            MODELS_DIR
            / f"final_lgbm_{horizon}.joblib",
        )

    # STEP 16
    # AE + LightGBM TEST

    development_ae_latent = (
        extract_ae_latents(
            ae_model,
            ae_scaler,
            development_df,
            feature_columns,
        )
    )

    # IMPORTANT:
    # This AE encoder was trained on train only.
    # No refitting on validation/test.
    #

    test_ae_latent = ae_test_latent

    development_ae = development_df.merge(
        development_ae_latent,
        on="timestamp",
        how="left",
    )

    test_ae = test_df.merge(
        test_ae_latent,
        on="timestamp",
        how="left",
    )

    X_dev_ae = (
        development_ae[
            ae_lgbm_features
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    X_test_ae = (
        test_ae[
            ae_lgbm_features
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    final_ae_imputer = SimpleImputer(
        strategy="median"
    )

    X_dev_ae = (
        final_ae_imputer
        .fit_transform(
            X_dev_ae
        )
    )

    X_test_ae = (
        final_ae_imputer
        .transform(
            X_test_ae
        )
    )

    final_ae_lgbm = {}

    ae_test_predictions = (
        test_df[
            [
                "timestamp"
            ]
        ].copy()
    )

    for horizon in [
        "1H",
        "1W",
        "1M",
    ]:

        model = create_lgbm_model()

        y = (
            development_ae[
                f"target_{horizon}"
            ].values
        )

        valid = np.isfinite(
            y
        )

        model.fit(
            X_dev_ae[
                valid
            ],
            y[
                valid
            ],
        )

        final_ae_lgbm[
            horizon
        ] = model

        ae_test_predictions[
            f"ae_lgbm_{horizon}"
        ] = model.predict(
            X_test_ae
        )

    # STEP 17
    # RESNET TEST

    # Train final ResNet on development set.
    #
    final_resnet = train_resnet(
        development_df,
        test_df,
        [
            "1H_close",
            "1H_log_return",
            "1H_volume",
        ],
    )

    resnet_test = (
        predict_resnet(
            final_resnet,
            test_df,
            [
                "1H_close",
                "1H_log_return",
                "1H_volume",
            ],
        )
    )

    # STEP 18
    # ARIMA TEST

    arima_test = (
        rolling_arima_predictions(
            development_df,
            test_df,
        )
    )

    # STEP 19
    # BUILD FINAL TEST META-FEATURES

    test_meta = test_df[
        [
            "timestamp",
            "current_price",
            "target_1H",
            "target_1W",
            "target_1M",
        ]
    ].copy()

    for frame in [
        lgbm_test_predictions,
        ae_test_predictions,
        resnet_test,
        arima_test,
    ]:

        test_meta = test_meta.merge(
            frame,
            on="timestamp",
            how="left",
        )

    # STEP 20
    # FINAL FUSION

    fusion_test_predictions = (
        test_meta[
            [
                "timestamp"
            ]
        ].copy()
    )

    for horizon in [
        "1H",
        "1W",
        "1M",
    ]:

        bundle = fusion_models[
            horizon
        ]

        model = bundle[
            "model"
        ]

        imputer = bundle[
            "imputer"
        ]

        features = bundle[
            "features"
        ]

        X = (
            test_meta[
                features
            ]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
        )

        X = imputer.transform(
            X
        )

        fusion_test_predictions[
            f"fusion_{horizon}"
        ] = model.predict(
            X
        )

    # STEP 21
    # Naive baseline

    naive = naive_predictions(
        test_df
    )

    for horizon in [
        "1H",
        "1W",
        "1M",
    ]:

        test_meta[
            f"naive_{horizon}"
        ] = naive[
            f"naive_{horizon}"
        ].values

    # STEP 22
    # Assemble predictions

    prediction_table = (
        test_meta[
            [
                "timestamp",
                "current_price",
                "target_1H",
                "target_1W",
                "target_1M",
            ]
        ]
        .merge(
            lgbm_test_predictions,
            on="timestamp",
            how="left",
        )
        .merge(
            ae_test_predictions,
            on="timestamp",
            how="left",
        )
        .merge(
            resnet_test,
            on="timestamp",
            how="left",
        )
        .merge(
            arima_test,
            on="timestamp",
            how="left",
        )
        .merge(
            fusion_test_predictions,
            on="timestamp",
            how="left",
        )
    )

    # Add naive
    for horizon in [
        "1H",
        "1W",
        "1M",
    ]:

        prediction_table[
            f"naive_{horizon}"
        ] = 0.0

    # STEP 23
    # Reconstruct future prices

    final_forecasts = reconstruct_prices(
        fusion_test_predictions,
        test_df,
    )

    # STEP 24
    # Evaluation

    all_predictions_for_eval = (
        prediction_table.copy()
    )

    metrics_frames = []

    # Base models
    for model_name in [
        "arima",
        "lgbm",
        "resnet",
        "fusion",
        "ae_lgbm",
        "naive",
    ]:

        for horizon in [
            "1H",
            "1W",
            "1M",
        ]:

            pred_column = (
                f"{model_name}_{horizon}"
            )

            target_column = (
                f"target_{horizon}"
            )

            if (
                pred_column
                not in all_predictions_for_eval.columns
            ):

                continue

            m = calculate_metrics(
                all_predictions_for_eval[
                    target_column
                ],
                all_predictions_for_eval[
                    pred_column
                ],
            )

            m[
                "model"
            ] = model_name

            m[
                "horizon"
            ] = horizon

            metrics_frames.append(
                m
            )

    metrics = pd.DataFrame(
        metrics_frames
    )

    # STEP 25
    # Save everything

    prediction_file = (
        PREDICTIONS_DIR
        / "ADAUSDT_test_predictions.csv"
    )

    forecast_file = (
        PREDICTIONS_DIR
        / "ADAUSDT_final_fused_forecasts.csv"
    )

    metrics_file = (
        REPORTS_DIR
        / "ADAUSDT_model_metrics.csv"
    )

    prediction_table.to_csv(
        prediction_file,
        index=False,
    )

    final_forecasts.to_csv(
        forecast_file,
        index=False,
    )

    metrics.to_csv(
        metrics_file,
        index=False,
    )

    # STEP 26
    # Save metadata

    metadata = {
        "dataset":
            CFG.data_path,
        "asset":
            "ADA",
        "market":
            "ADAUSDT",
        "source":
            "Binance Spot",
        "start":
            CFG.start_date,
        "end":
            CFG.end_date,
        "prediction_horizons":
            list(
                CFG.horizons
            ),
        "canonical_frequency":
            "1H",
        "models": [
            "ARIMA",
            "LightGBM",
            "AutoEncoder",
            "AE+LightGBM",
            "ResNet18",
            "Fusion LightGBM",
        ],
        "device":
            str(DEVICE),
        "rows_modeling_dataset":
            int(len(modeling_df)),
        "rows_train":
            int(len(train_df)),
        "rows_validation":
            int(len(validation_df)),
        "rows_test":
            int(len(test_df)),
    }

    with open(
        REPORTS_DIR
        / "pipeline_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )

    save_configuration()

    # FINAL SUMMARY

    print(
        "\n"
        "================================================================\n"
        " PIPELINE COMPLETE\n"
        "================================================================"
    )

    print(
        "\nModel metrics:"
    )

    print(
        metrics
        .sort_values(
            [
                "horizon",
                "RMSE",
            ]
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nFinal forecast file:"
    )

    print(
        forecast_file
    )

    print(
        "\nPrediction file:"
    )

    print(
        prediction_file
    )

    print(
        "\nMetrics:"
    )

    print(
        metrics_file
    )

    print(
        "\nModels:"
    )

    print(
        MODELS_DIR
    )

    return {
        "modeling_df":
            modeling_df,
        "train":
            train_df,
        "validation":
            validation_df,
        "test":
            test_df,
        "predictions":
            prediction_table,
        "forecasts":
            final_forecasts,
        "metrics":
            metrics,
    }


# بدوبدوبدوبدوبدوبدو
# RUN
# بدوبدوبدوبدوبدوبدو

if __name__ == "__main__":

    results = run_pipeline()



# !tar -cvf archive.tar /content/ada_hybrid_forecasting/