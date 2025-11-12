import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import pandas_ta as ta
import sqlalchemy
from sqlalchemy import bindparam, text
import joblib

# ===================== CONFIG =====================
FUTURE_MODE = True
USE_DATABASE = True
DB_CONNECTION = "mysql+mysqlconnector://GPTFOREX:GPtushechkaForexUshechka@localhost/GPTFOREX"

PRICE_START_DATE = "2022-01-01"
START_DATE = "2005-01-01"

IMP_TOTAL_THRESHOLD = 0.10
FUTURE_DIRECTION_PROB_THRESHOLD = 0.40
FUTURE_MAGNITUDE_THRESHOLD = 2.50
HISTORY_DIRECTION_PROB_THRESHOLD = 0.65
HISTORY_MAGNITUDE_THRESHOLD = 7.00

GROUP_WINDOW_MINUTES = 30

CURRENCY_PAIR = "EUR/USD"
MODEL_TAG = "V19F_15m"
MAGNITUDE_PRIORITY_THRESHOLD = 6.6227
LOG_PATH = "/home/ilyamus/GPTGROKWORK/AITrainer_V5/logs/predict_V19F_to_db.log"

MODEL_DIR = "/home/ilyamus/GPTGROKWORK/AITrainer_V5"
FALLBACK_MODEL_DIR = "/home/ilyamus/GPTGROKWORK/AITrainer_V5"

FEATURES: List[str] = [
    "actual_minus_forecast",
    "imp_calculated",
    "imp_trend",
    "imp_total",
    "dependence_encoded",
    "hour",
    "day_of_week",
    "actual",
    "news_impact",
    "volatility_pre",
    "event_key_encoded",
    "prev_magnitude_1",
    "prev_direction_1",
    "prev_magnitude_2",
    "prev_direction_2",
    "prev_magnitude_3",
    "prev_direction_3",
    "imp_total_category",
    "RSI_14",
    "SMA_20",
    "time_since_last_event",
    "ATR_14",
    "correlation",
    "price_change",
    "corr_direction",
    "probability",
    "observations",
    "SMA_365",
    "trend_365",
    "SMA_1M",
    "SMA_3M",
    "SMA_6M",
    "SMA_12M",
    "trend_1M",
    "trend_3M",
    "trend_6M",
    "trend_12M",
    "SMA_3Q",
    "SMA_6Q",
    "SMA_12Q",
    "trend_3Q",
    "trend_6Q",
    "trend_12Q",
]

# ===================== LOGGING =====================
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)sZ - %(levelname)s - %(message)s",
)

# ===================== HELPERS =====================

def _create_engine() -> sqlalchemy.Engine:
    if not USE_DATABASE:
        raise RuntimeError("USE_DATABASE=False is not supported for this script")
    engine = sqlalchemy.create_engine(DB_CONNECTION, pool_pre_ping=True, pool_recycle=3600)
    with engine.begin() as conn:
        conn.execute(text("SET time_zone = '+00:00'"))
        conn.execute(text("SET NAMES utf8mb4"))
    return engine


def _load_label_encoder(filename: str) -> Optional[Any]:
    if os.path.exists(filename):
        logging.info("Loading label encoder from %s", filename)
        return joblib.load(filename)
    return None


def _load_from_candidates(
    filenames: Sequence[str],
    *,
    description: str,
) -> Optional[Any]:
    """Return the first joblib object found in MODEL_DIR or FALLBACK_MODEL_DIR."""

    for directory in (MODEL_DIR, FALLBACK_MODEL_DIR):
        for name in filenames:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                logging.info("Loading %s from %s", description, path)
                return joblib.load(path)
    logging.warning("No %s found in %s", description, filenames)
    return None


def _load_encoders() -> Dict[str, Any]:
    encoders: Dict[str, Any] = {}
    event_filenames = (
        "label_encoder_event_15min.pkl",
        "label_encoder_event_te_15m.pkl",
        "label_encoder_event_te_15min.pkl",
    )
    dep_filenames = (
        "label_encoder_dependence_15min.pkl",
        "label_encoder_dependence_te_15m.pkl",
        "label_encoder_dependence_te_15min.pkl",
    )

    encoders["event"] = _load_from_candidates(event_filenames, description="event label encoder")
    encoders["dependence"] = _load_from_candidates(
        dep_filenames, description="dependence label encoder"
    )

    if encoders["event"] is None:
        raise FileNotFoundError("Event label encoder not found in either primary or fallback directories")

    if encoders["dependence"] is None:
        logging.warning(
            "Dependence label encoder not found; dependence values will be encoded as unknown"
        )
    return encoders


def _encode_with_unknown(le: Any, values: Iterable[Any]) -> np.ndarray:
    values_list = list(values)
    if le is None:
        logging.debug("Label encoder is missing; defaulting to -1 for %d values", len(values_list))
        return np.full(len(values_list), -1, dtype=int)

    classes = getattr(le, "classes_", None)
    if classes is None:
        logging.warning(
            "Label encoder %s has no classes_; defaulting to -1 for %d values",
            getattr(le, "__class__", type(le)).__name__,
            len(values_list),
        )
        return np.full(len(values_list), -1, dtype=int)
    mapping = {cls: idx for idx, cls in enumerate(classes)}
    return np.array([mapping.get(v, -1) for v in values_list], dtype=int)


def _load_models() -> Dict[str, Any]:
    models: Dict[str, Any] = {}

    rf_candidates = (
        "model_direction_rf_15min.pkl",
        "model_te_direction_rf_15m.pkl",
    )
    xgb_candidates = (
        "model_direction_xgb_15min.pkl",
        "model_te_direction_xgb_15m.pkl",
        "model_te_dir2_xgb_15m.pkl",
    )
    reg_candidates = (
        "model_magnitude_15min.pkl",
        "model_te_magnitude_xgb_15m.pkl",
    )

    models["rf"] = _load_from_candidates(rf_candidates, description="direction RF model")
    models["xgb"] = _load_from_candidates(xgb_candidates, description="direction XGB model")
    models["reg"] = _load_from_candidates(reg_candidates, description="magnitude regression model")

    missing = [name for name, model in models.items() if model is None]
    if missing:
        raise FileNotFoundError(f"Missing required models: {', '.join(missing)}")

    return models


def _load_prices(engine: sqlalchemy.Engine) -> pd.DataFrame:
    logging.info("Loading price data starting %s", PRICE_START_DATE)
    query = text(
        """
        SELECT timestamp_utc, open, high, low, close
        FROM HistDataEURUSD
        WHERE timestamp_utc >= :start
        ORDER BY timestamp_utc
        """
    )
    prices = pd.read_sql(query, engine, params={"start": PRICE_START_DATE})
    if prices.empty:
        raise RuntimeError("No price data returned from HistDataEURUSD")
    prices["timestamp_utc"] = pd.to_datetime(prices["timestamp_utc"], utc=True).dt.tz_convert(None)
    prices["timestamp_utc"] = prices["timestamp_utc"].dt.round("min")
    prices = prices.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"])
    prices["high_pips"] = prices["high"] * 10000.0
    prices["low_pips"] = prices["low"] * 10000.0
    prices["close_pips"] = prices["close"] * 10000.0
    prices["RSI_14"] = ta.rsi(prices["close"], length=14).fillna(50.0)
    prices["SMA_20"] = ta.sma(prices["close"], length=20).bfill().fillna(prices["close"].mean())
    atr = ta.atr(prices["high_pips"], prices["low_pips"], prices["close_pips"], length=14)
    prices["ATR_14"] = atr.bfill().fillna(atr.mean()).fillna(0.0)
    logging.info("Price indicators prepared: %s rows", len(prices))
    return prices


def _prepare_daily_trends(prices: pd.DataFrame) -> pd.DataFrame:
    daily = prices.resample("1D", on="timestamp_utc").agg({"close": "mean"}).dropna().reset_index()
    daily = daily.sort_values("timestamp_utc")
    for months in [1, 3, 6, 12]:
        length = max(2, months * 30)
        daily[f"SMA_{months}M"] = ta.sma(daily["close"], length=length).bfill()
        daily[f"trend_{months}M"] = daily["close"].pct_change(periods=length).fillna(0.0)
    for quarters in [3, 6, 12]:
        length = max(2, quarters * 90)
        daily[f"SMA_{quarters}Q"] = ta.sma(daily["close"], length=length).bfill()
        daily[f"trend_{quarters}Q"] = daily["close"].pct_change(periods=length).fillna(0.0)
    daily["SMA_365"] = ta.sma(daily["close"], length=365).bfill()
    daily["trend_365"] = daily["close"].pct_change(periods=365).fillna(0.0)
    return daily


def _load_removed_news(engine: sqlalchemy.Engine) -> set:
    query = text(
        """
        SELECT id
        FROM news_removed_log
        WHERE removed_at >= UTC_TIMESTAMP() - INTERVAL 2 DAY
          AND imp_total > 0.1
        """
    )
    df = pd.read_sql(query, engine)
    return set(df["id"].astype(int)) if not df.empty else set()


def _load_news(engine: sqlalchemy.Engine) -> pd.DataFrame:
    base_query = """
        SELECT id, timestamp_utc, event, event_key, imp_total, imp_calculated, imp_trend,
               actual_minus_forecast, actual, direction, magnitude, dependence
        FROM economic_news_model_grok
    """
    if FUTURE_MODE:
        query = text(
            base_query
            + """
            WHERE timestamp_utc >= UTC_TIMESTAMP() - INTERVAL 3 HOUR
              AND imp_total > :imp_total
              AND event_key IS NOT NULL
            ORDER BY timestamp_utc
            """
        )
        df = pd.read_sql(query, engine, params={"imp_total": IMP_TOTAL_THRESHOLD})
    else:
        query = text(
            base_query
            + """
            WHERE timestamp_utc >= :start
              AND imp_total > 0.1
              AND event_key IS NOT NULL
            ORDER BY timestamp_utc
            """
        )
        df = pd.read_sql(query, engine, params={"start": START_DATE})
    if df.empty:
        logging.warning("No news fetched for current mode")
        return df
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True).dt.tz_convert(None)
    df["timestamp_utc"] = df["timestamp_utc"].dt.round("min")
    return df


def _load_correlations(
    engine: sqlalchemy.Engine,
    min_ts: datetime,
    max_ts: datetime,
    event_keys: Iterable[str],
) -> pd.DataFrame:
    if event_keys is None:
        return pd.DataFrame()

    event_keys_list = list(event_keys)
    if not event_keys_list:
        return pd.DataFrame()

    unique_keys = [
        ek for ek in {key for key in event_keys_list if isinstance(key, str)} if ek
    ]
    if not unique_keys:
        return pd.DataFrame()
    query = text(
        """
        SELECT timestamp_utc, event_key, correlation, price_change, direction AS corr_direction,
               probability, observations
        FROM correlation_trends_v2
        WHERE timestamp_utc BETWEEN :start AND :end
          AND event_key IN :event_keys
          AND observations > 100
          AND probability > 0.5
        """
    ).bindparams(bindparam("event_keys", expanding=True))
    params = {
        "start": (min_ts - timedelta(days=7)),
        "end": (max_ts + timedelta(days=7)),
        "event_keys": tuple(unique_keys),
    }
    df = pd.read_sql(query, engine, params=params)
    if df.empty:
        return df
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True).dt.tz_convert(None)
    df["timestamp_utc"] = df["timestamp_utc"].dt.round("min")
    return df


def _compute_news_impact(news: pd.DataFrame) -> pd.Series:
    if news.empty:
        return pd.Series(dtype=float)
    sorted_news = news.sort_values("timestamp_utc").set_index("timestamp_utc")
    window_sum = sorted_news["imp_total"].rolling("30min", center=True).sum().fillna(0.0)
    impact = np.log1p(window_sum)
    impact = impact.reindex(sorted_news.index)
    return impact.reset_index(drop=True)


def _compute_volatility(prices: pd.DataFrame, news: pd.DataFrame) -> pd.Series:
    if news.empty:
        return pd.Series(dtype=float)
    resampled = (
        prices.set_index("timestamp_utc")
        .resample("15T")
        .agg({"high": "max", "low": "min"})
        .dropna()
        .reset_index()
    )
    resampled["price_range"] = (resampled["high"] - resampled["low"]) * 10000.0
    resampled = resampled.sort_values("timestamp_utc")
    merged = pd.merge_asof(
        news.sort_values("timestamp_utc"),
        resampled[["timestamp_utc", "price_range"]],
        on="timestamp_utc",
        direction="backward",
        tolerance=pd.Timedelta("1D"),
    )
    return merged["price_range"].fillna(resampled["price_range"].mean()).fillna(0.0)


def _prepare_lag_features(news: pd.DataFrame) -> pd.DataFrame:
    work = news.sort_values("timestamp_utc").copy()
    for col_base, source_col in [("magnitude", "magnitude"), ("direction", "direction_numeric")]:
        for lag in range(1, 4):
            col_name = f"prev_{col_base}_{lag}" if col_base == "magnitude" else f"prev_direction_{lag}"
            work[col_name] = work[source_col].shift(lag)
    work["time_since_last_event"] = (
        work["timestamp_utc"].diff().dt.total_seconds().div(60.0).fillna(0.0).clip(lower=0.0)
    )
    return work


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str], default: float = 0.0) -> None:
    for col in columns:
        if col not in df.columns:
            df[col] = default


def _round_decimal(series: pd.Series, decimals: int) -> pd.Series:
    return series.apply(lambda x: round(float(x), decimals) if pd.notnull(x) else x)


def _attempt_create_unique_index(engine: sqlalchemy.Engine) -> None:
    ddl = text(
        """
        ALTER TABLE predictETH5
        ADD UNIQUE KEY ux_predictETH5_ts_event_model (ts_utc, event_key, model_tag)
        """
    )
    try:
        with engine.begin() as conn:
            conn.execute(ddl)
            logging.info("Unique index ux_predictETH5_ts_event_model created on predictETH5")
    except Exception as exc:  # noqa: BLE001
        logging.info("Unique index creation skipped: %s", exc)


def _table_has_updated_at(engine: sqlalchemy.Engine) -> bool:
    query = text("SHOW COLUMNS FROM predictETH5 LIKE 'updated_at'")
    with engine.begin() as conn:
        result = conn.execute(query).fetchone()
    return result is not None


def _serialize_extra_json(row: pd.Series, rf_prob: float, xgb_prob: float) -> str:
    payload = {
        "future_mode": FUTURE_MODE,
        "thr_dir": FUTURE_DIRECTION_PROB_THRESHOLD if FUTURE_MODE else HISTORY_DIRECTION_PROB_THRESHOLD,
        "thr_mag": FUTURE_MAGNITUDE_THRESHOLD if FUTURE_MODE else HISTORY_MAGNITUDE_THRESHOLD,
        "rf_p_mean": float(rf_prob),
        "xgb_p_mean": float(xgb_prob),
        "vol_pre": float(row.get("volatility_pre", 0.0)),
        "news_impact": float(row.get("news_impact", 0.0)),
    }
    return json.dumps(payload, ensure_ascii=False)


def _prepare_predictions_df(
    news: pd.DataFrame,
    prices: pd.DataFrame,
    models: Dict[str, Any],
    encoders: Dict[str, Any],
    engine: sqlalchemy.Engine,
) -> pd.DataFrame:
    if news.empty:
        return pd.DataFrame()

    news = news.copy()
    news = news.sort_values("timestamp_utc").reset_index(drop=True)
    news["hour"] = news["timestamp_utc"].dt.hour.astype(int)
    news["day_of_week"] = news["timestamp_utc"].dt.dayofweek.astype(int)

    news["actual_minus_forecast"] = pd.to_numeric(news.get("actual_minus_forecast"), errors="coerce").fillna(0.0)
    news["imp_calculated"] = pd.to_numeric(news.get("imp_calculated"), errors="coerce").fillna(0.0)
    news["imp_trend"] = pd.to_numeric(news.get("imp_trend"), errors="coerce").fillna(0.0)
    news["imp_total"] = pd.to_numeric(news.get("imp_total"), errors="coerce").fillna(0.0)
    news["actual"] = pd.to_numeric(news.get("actual"), errors="coerce").fillna(0.0)
    news["magnitude"] = pd.to_numeric(news.get("magnitude"), errors="coerce").fillna(0.0)

    direction_raw = news.get("direction")
    if direction_raw is not None:
        mapping = {"down": 0, "sell": 0, "bearish": 0, "flat": 1, "none": 1, "neutral": 1, "up": 2, "buy": 2, "bullish": 2}
        news["direction_numeric"] = pd.to_numeric(direction_raw, errors="coerce")
        mask_missing = news["direction_numeric"].isna()
        if mask_missing.any():
            news.loc[mask_missing, "direction_numeric"] = direction_raw[mask_missing].astype(str).str.lower().map(mapping)
        news["direction_numeric"] = news["direction_numeric"].fillna(1).astype(float)
    else:
        news["direction_numeric"] = 1.0

    news_lagged = _prepare_lag_features(news)
    for lag in range(1, 4):
        news[f"prev_magnitude_{lag}"] = news_lagged[f"prev_magnitude_{lag}"].fillna(0.0)
        news[f"prev_direction_{lag}"] = news_lagged[f"prev_direction_{lag}"].fillna(1.0)
    news["time_since_last_event"] = news_lagged["time_since_last_event"].fillna(0.0)

    news["imp_total_category"] = pd.cut(
        news["imp_total"],
        bins=[0.0, 0.3, 0.6, 1.0],
        labels=[0, 1, 2],
        include_lowest=True,
        right=True,
    ).astype("float").fillna(2.0)

    news["news_impact"] = _compute_news_impact(news)
    news["volatility_pre"] = _compute_volatility(prices, news)

    event_encoder = encoders["event"]
    dep_encoder = encoders["dependence"]
    news["event_key_encoded"] = _encode_with_unknown(event_encoder, news["event_key"].astype(str))
    news["dependence_encoded"] = _encode_with_unknown(dep_encoder, news["dependence"].fillna("__missing__").astype(str))

    price_cols = ["timestamp_utc", "close", "RSI_14", "SMA_20", "ATR_14"]
    merged = pd.merge_asof(
        news.sort_values("timestamp_utc"),
        prices[price_cols].sort_values("timestamp_utc"),
        on="timestamp_utc",
        direction="backward",
        tolerance=pd.Timedelta("1D"),
    )
    news[["close", "RSI_14", "SMA_20", "ATR_14"]] = merged[["close", "RSI_14", "SMA_20", "ATR_14"]]

    daily = _prepare_daily_trends(prices)
    news = pd.merge_asof(
        news.sort_values("timestamp_utc"),
        daily[[
            "timestamp_utc",
            "SMA_1M",
            "SMA_3M",
            "SMA_6M",
            "SMA_12M",
            "trend_1M",
            "trend_3M",
            "trend_6M",
            "trend_12M",
            "SMA_3Q",
            "SMA_6Q",
            "SMA_12Q",
            "trend_3Q",
            "trend_6Q",
            "trend_12Q",
            "SMA_365",
            "trend_365",
        ]].sort_values("timestamp_utc"),
        on="timestamp_utc",
        direction="backward",
        tolerance=pd.Timedelta("1D"),
    )

    daily_cols = [
        "SMA_365",
        "trend_365",
        "SMA_1M",
        "SMA_3M",
        "SMA_6M",
        "SMA_12M",
        "trend_1M",
        "trend_3M",
        "trend_6M",
        "trend_12M",
        "SMA_3Q",
        "SMA_6Q",
        "SMA_12Q",
        "trend_3Q",
        "trend_6Q",
        "trend_12Q",
    ]
    for col in daily_cols:
        if col in news.columns:
            if "trend" in col:
                news[col] = news[col].fillna(0.0)
            else:
                news[col] = news[col].fillna(method="bfill").fillna(news[col].mean())
        else:
            news[col] = 0.0

    correlations = _load_correlations(
        engine=engine,
        min_ts=news["timestamp_utc"].min(),
        max_ts=news["timestamp_utc"].max(),
        event_keys=news["event_key"].dropna().unique(),
    )
    if not correlations.empty:
        news = news.merge(
            correlations,
            on=["timestamp_utc", "event_key"],
            how="left",
        )
    _ensure_columns(news, ["correlation", "price_change", "corr_direction", "probability", "observations"], default=0.0)
    news[["correlation", "price_change", "corr_direction", "probability", "observations"]] = news[
        ["correlation", "price_change", "corr_direction", "probability", "observations"]
    ].fillna(0.0)

    news.rename(columns={"close": "price_entry"}, inplace=True)

    missing_features = [col for col in FEATURES if col not in news.columns]
    if missing_features:
        logging.warning("Missing features %s filled with zeros", missing_features)
        for col in missing_features:
            news[col] = 0.0

    feature_df = news[FEATURES].copy()
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    rf_probs = models["rf"].predict_proba(feature_df)
    xgb_probs = models["xgb"].predict_proba(feature_df)
    ensemble_probs = (rf_probs + xgb_probs) / 2.0
    direction_prob = ensemble_probs.max(axis=1)
    direction_pred = ensemble_probs.argmax(axis=1)

    reg_raw = models["reg"].predict(feature_df)
    reg_raw = np.maximum(reg_raw, 0.0)
    magnitude_pred = np.expm1(reg_raw)

    news["direction_pred"] = direction_pred.astype(int)
    news["direction_prob"] = direction_prob.astype(float)
    news["magnitude_pred"] = magnitude_pred.astype(float)
    news["rf_prob_pred"] = [rf_probs[i, dp] for i, dp in enumerate(direction_pred)]
    news["xgb_prob_pred"] = [xgb_probs[i, dp] for i, dp in enumerate(direction_pred)]

    price_exit = news["price_entry"].copy()
    mask_up = news["direction_pred"] == 2
    mask_down = news["direction_pred"] == 0
    pip_factor = news["magnitude_pred"] * 0.0001
    price_exit.loc[mask_up] = news.loc[mask_up, "price_entry"] + pip_factor.loc[mask_up]
    price_exit.loc[mask_down] = news.loc[mask_down, "price_entry"] - pip_factor.loc[mask_down]
    news["price_exit"] = price_exit

    news["priority"] = np.where(
        news["magnitude_pred"] >= MAGNITUDE_PRIORITY_THRESHOLD,
        "High",
        "Low",
    )

    news["currency_pair"] = CURRENCY_PAIR
    news["model_tag"] = MODEL_TAG

    news["extra_json"] = [
        _serialize_extra_json(row, row["rf_prob_pred"], row["xgb_prob_pred"])
        for _, row in news.iterrows()
    ]

    return news


def _filter_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    if preds.empty:
        return preds
    if FUTURE_MODE:
        direction_thr = FUTURE_DIRECTION_PROB_THRESHOLD
        magnitude_thr = FUTURE_MAGNITUDE_THRESHOLD
    else:
        direction_thr = HISTORY_DIRECTION_PROB_THRESHOLD
        magnitude_thr = HISTORY_MAGNITUDE_THRESHOLD
    mask = (preds["direction_prob"] > direction_thr) & (preds["magnitude_pred"] > magnitude_thr)
    filtered = preds.loc[mask].copy()
    logging.info(
        "Filtered predictions: %s kept / %s total (thr_dir=%.2f, thr_mag=%.2f)",
        len(filtered),
        len(preds),
        direction_thr,
        magnitude_thr,
    )
    return filtered


def _prepare_insert_frame(preds: pd.DataFrame, removed_ids: set) -> pd.DataFrame:
    if preds.empty:
        return preds
    preds = preds.copy()
    preds["is_removed"] = preds["id"].isin(removed_ids).astype(int)
    preds.rename(columns={"timestamp_utc": "ts_utc", "id": "src_id"}, inplace=True)

    preds = preds.sort_values("direction_prob", ascending=False)
    preds = preds.drop_duplicates(subset=["ts_utc", "event_key", "model_tag"], keep="first")

    preds["direction_prob"] = _round_decimal(preds["direction_prob"], 4)
    preds["magnitude_pred"] = _round_decimal(preds["magnitude_pred"], 3)
    preds["price_entry"] = _round_decimal(preds["price_entry"], 5)
    preds["price_exit"] = _round_decimal(preds["price_exit"], 5)

    return preds[
        [
            "ts_utc",
            "event",
            "event_key",
            "src_id",
            "direction_pred",
            "direction_prob",
            "magnitude_pred",
            "price_entry",
            "price_exit",
            "currency_pair",
            "priority",
            "is_removed",
            "model_tag",
            "extra_json",
        ]
    ]


def _insert_predictions(engine: sqlalchemy.Engine, df: pd.DataFrame, update_has_col: bool) -> None:
    if df.empty:
        logging.info("No predictions to insert")
        return
    records = df.to_dict(orient="records")
    for rec in records:
        if isinstance(rec["ts_utc"], pd.Timestamp):
            rec["ts_utc"] = rec["ts_utc"].to_pydatetime()
    base_sql = """
        INSERT INTO predictETH5
        (ts_utc, event, event_key, src_id, direction_pred, direction_prob, magnitude_pred,
         price_entry, price_exit, currency_pair, priority, is_removed, model_tag, extra_json)
        VALUES
        (:ts_utc, :event, :event_key, :src_id, :direction_pred, :direction_prob, :magnitude_pred,
         :price_entry, :price_exit, :currency_pair, :priority, :is_removed, :model_tag, :extra_json)
        ON DUPLICATE KEY UPDATE
         direction_pred=VALUES(direction_pred),
         direction_prob=VALUES(direction_prob),
         magnitude_pred=VALUES(magnitude_pred),
         price_entry=VALUES(price_entry),
         price_exit=VALUES(price_exit),
         priority=VALUES(priority),
         is_removed=VALUES(is_removed),
         extra_json=VALUES(extra_json)
    """
    if update_has_col:
        base_sql += ", updated_at=NOW()"
    stmt = text(base_sql)
    with engine.begin() as conn:
        conn.execute(stmt, records)
    logging.info("Inserted/updated %s records into predictETH5", len(records))


def main() -> None:
    logging.info("==== predict_V19F_to_db start ====")
    try:
        engine = _create_engine()
    except Exception as exc:  # noqa: BLE001
        logging.exception("Failed to create database engine: %s", exc)
        raise

    _attempt_create_unique_index(engine)
    has_updated_at = _table_has_updated_at(engine)

    prices = _load_prices(engine)
    news = _load_news(engine)
    if news.empty:
        logging.warning("No news to process, exiting")
        return

    removed_ids = _load_removed_news(engine)
    logging.info("Removed news ids loaded: %s", len(removed_ids))

    try:
        encoders = _load_encoders()
        models = _load_models()
    except Exception as exc:  # noqa: BLE001
        logging.exception("Failed to load models or encoders: %s", exc)
        raise

    predictions = _prepare_predictions_df(news, prices, models, encoders, engine)
    if predictions.empty:
        logging.warning("No predictions computed")
        return

    logging.info("Predictions computed: %s rows", len(predictions))
    class_distribution = predictions["direction_pred"].value_counts().to_dict()
    logging.info("Class distribution: %s", class_distribution)

    filtered = _filter_predictions(predictions)
    if filtered.empty:
        logging.info("No predictions passed filtering thresholds")
        return

    filtered["price_entry"] = filtered["price_entry"].fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    filtered["price_exit"] = filtered["price_exit"].fillna(filtered["price_entry"])

    to_insert = _prepare_insert_frame(filtered, removed_ids)
    logging.info("Prepared %s rows for insertion", len(to_insert))
    if not to_insert.empty:
        logging.info("Top predictions:\n%s", to_insert.head().to_string(index=False))

    _insert_predictions(engine, to_insert, has_updated_at)
    logging.info("==== predict_V19F_to_db completed ====")


if __name__ == "__main__":
    main()