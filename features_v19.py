"""Feature engineering utilities for V19 models."""
from __future__ import annotations

import logging
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

FEATURES_V19: list[str] = [
    # time/session
    'hour', 'day_of_week', 'is_weekday', 'is_eu_session', 'is_us_session',
    # event key / text meta
    'event_key_encoded', 'text_len',
    # sentiment
    'sent_finbert', 'sent_label_id', 'sent_pos', 'sent_neg', 'sent_pos_x_accel',
    # impact / market
    'imp_year_vol',
    'volatility_pre_15m', 'RSI_14', 'RSI_high', 'RSI_low', 'SMA_20', 'ATR_14',
    'SMA_365', 'trend_365',
    # daily windows
    'SMA_1M', 'SMA_3M', 'SMA_6M', 'SMA_12M', 'trend_1M', 'trend_3M', 'trend_6M', 'trend_12M',
    'SMA_3Q', 'SMA_6Q', 'SMA_12Q', 'trend_3Q', 'trend_6Q', 'trend_12Q',
    # correlations
    'correlation',
    # momentum prior to news
    'ret_pre_5m', 'ret_pre_15m', 'sign_ret15', 'ret_pre_15m_norm', 'accel_5_15',
]


def _ensure_timezone_naive(series: pd.Series) -> pd.Series:
    """Strip timezone information if present."""
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    try:
        tz = series.dt.tz
    except AttributeError:
        tz = None
    if tz is not None:
        return series.dt.tz_convert(None)
    try:
        return series.dt.tz_localize(None)
    except (TypeError, AttributeError):
        return series


def _prepare_label_encoder(le_event: LabelEncoder | None, event_series: pd.Series) -> Tuple[LabelEncoder, np.ndarray]:
    """Fit/transform event key encoder with support for unseen categories."""
    if le_event is None:
        le_event = LabelEncoder()

    event_series = event_series.fillna('__missing__').astype(str)
    event_series = event_series.replace({'': '__missing__'})

    if not hasattr(le_event, 'classes_'):
        fit_values = pd.concat(
            [event_series, pd.Series(['__unknown__'])],
            ignore_index=True,
        )
        le_event.fit(fit_values)
    else:
        if '__unknown__' not in le_event.classes_:
            le_event.classes_ = np.unique(np.append(le_event.classes_, '__unknown__'))

    safe_values = event_series.where(event_series.isin(le_event.classes_), '__unknown__')
    encoded = le_event.transform(safe_values)
    return le_event, encoded


def _map_country_to_ccy(country: str) -> str:
    if not isinstance(country, str) or not country.strip():
        return 'ALL'
    c = country.strip()
    if c.lower().startswith('euro'):
        return 'EUR'
    country_map = {
        'United States': 'USD', 'Euro Area': 'EUR', 'Eurozone': 'EUR', 'Germany': 'EUR',
        'France': 'EUR', 'United Kingdom': 'GBP', 'Japan': 'JPY', 'Canada': 'CAD',
        'Switzerland': 'CHF', 'Australia': 'AUD', 'China': 'CNY', 'India': 'INR',
        'Russia': 'RUB', 'Brazil': 'BRL', 'Mexico': 'MXN', 'ALL': 'ALL',
    }
    return country_map.get(c, 'ALL')


def build_features_for_news(
    news_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    imp_df: pd.DataFrame | None,
    corr_v5_df: Union[pd.DataFrame, Dict[str, pd.DataFrame], None],
    le_event: LabelEncoder | None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build deterministic feature matrix and accompanying metadata for news events."""
    if news_df.empty:
        return news_df[[]].copy(), news_df[['article_id', 'published_at', 'event_key', 'fx_weight']].copy()

    work_df = news_df.copy()
    work_df['_orig_index'] = work_df.index

    # Impact cache merge
    if imp_df is not None and not imp_df.empty:
        imp_cols = ['event_key', 'year', 'volatility', 'trend']
        imp_use = imp_df.loc[:, [c for c in imp_cols if c in imp_df.columns]].copy()
        imp_use = imp_use.rename(columns={'volatility': 'imp_year_vol', 'trend': 'imp_year_trend'})
        if {'event_key', 'year'}.issubset(imp_use.columns):
            work_df = work_df.merge(imp_use, on=['event_key', 'year'], how='left')
    work_df['imp_year_vol'] = work_df.get('imp_year_vol', 0.0).astype(float).fillna(0.0)
    work_df['imp_year_trend'] = work_df.get('imp_year_trend', 0.0).astype(float).fillna(0.0)

    # Base temporal features
    work_df['hour'] = work_df['published_at'].dt.hour.astype(int)
    work_df['day_of_week'] = work_df['published_at'].dt.dayofweek.astype(int)
    work_df['is_weekday'] = (work_df['day_of_week'] < 5).astype(int)
    work_df['is_eu_session'] = (
        (work_df['hour'] >= 7)
        & (work_df['hour'] <= 16)
        & (work_df['is_weekday'] == 1)
    ).astype(int)
    work_df['is_us_session'] = (
        (work_df['hour'] >= 12)
        & (work_df['hour'] <= 21)
        & (work_df['is_weekday'] == 1)
    ).astype(int)

    # Sentiment features
    work_df['sent_finbert'] = work_df['sent_finbert'].astype(float).fillna(0.0)
    work_df['sent_label'] = work_df['sent_label'].astype(str).fillna('neu')
    work_df['sent_label_id'] = work_df['sent_label'].map({'neg': 0, 'neu': 1, 'pos': 2}).fillna(1).astype(int)
    work_df['sent_pos'] = (work_df['sent_label_id'] == 2).astype(int)
    work_df['sent_neg'] = (work_df['sent_label_id'] == 0).astype(int)

    # Text metrics
    text_len = work_df['text_clean'].fillna('').astype(str).str.len()
    work_df['text_len'] = text_len.clip(upper=text_len.quantile(0.99)).astype(int)

    # Event key encoding
    le_event, encoded_event = _prepare_label_encoder(le_event, work_df['event_key'].astype(str))
    work_df['event_key_encoded'] = encoded_event.astype(int)

    # Correlation features
    corr_cols = ['correlation', 'price_change', 'corr_direction', 'probability', 'observations']
    for c in corr_cols:
        if c not in work_df.columns:
            work_df[c] = 0.0

    corr_event_df = None
    corr_fallback_df = None
    if isinstance(corr_v5_df, dict):
        corr_event_df = corr_v5_df.get('event')
        corr_fallback_df = corr_v5_df.get('fallback')
    else:
        corr_event_df = corr_v5_df

    if corr_event_df is not None and not corr_event_df.empty:
        available_cols = ['event_key'] + [c for c in corr_cols if c in corr_event_df.columns]
        corr_event_use = corr_event_df.loc[:, available_cols].drop_duplicates(subset=['event_key'])
        work_df = work_df.merge(corr_event_use, on='event_key', how='left', suffixes=('', '_corr'))
        for c in corr_cols:
            col_name = f"{c}_corr" if f"{c}_corr" in work_df.columns else c
            if col_name in work_df.columns:
                work_df[c] = work_df[col_name].fillna(work_df[c])
                if col_name != c:
                    work_df.drop(columns=[col_name], inplace=True)
        logging.info("Loaded correlation features from event-level data (correlation_trends_v5).")
    elif corr_fallback_df is not None and not corr_fallback_df.empty:
        corr_fallback = corr_fallback_df.copy()
        corr_fallback['currency'] = corr_fallback['currency'].str.upper()
        corr_fallback = corr_fallback.rename(columns={'correlation': 'corr_by_currency'})
        work_df['timestamp_utc'] = _ensure_timezone_naive(work_df['published_at'])
        work_df['news_year'] = pd.to_datetime(work_df['timestamp_utc']).dt.year
        work_df['news_currency'] = work_df['tags'].apply(_map_country_to_ccy).str.upper().fillna('ALL')
        work_df = work_df.merge(
            corr_fallback[['currency', 'year', 'corr_by_currency']],
            left_on=['news_currency', 'news_year'],
            right_on=['currency', 'year'],
            how='left',
        )
        corr_all = corr_fallback[corr_fallback['currency'] == 'ALL'][['year', 'corr_by_currency']].rename(
            columns={'corr_by_currency': 'corr_all'}
        )
        work_df = work_df.merge(corr_all, left_on='news_year', right_on='year', how='left')
        work_df['correlation'] = work_df['corr_by_currency'].fillna(work_df['corr_all']).fillna(0.0)
        for c in ['currency', 'year_x', 'year_y', 'corr_by_currency', 'corr_all', 'timestamp_utc']:
            if c in work_df.columns:
                work_df.drop(columns=[c], inplace=True)
        logging.info("Applied correlation fallback by currency/year.")
    else:
        work_df['correlation'] = work_df['correlation'].fillna(0.0)

    # Market features at event timestamp
    if prices_df is not None and not prices_df.empty:
        price_cols = [c for c in ['timestamp_utc', 'close', 'RSI_14', 'SMA_20', 'ATR_14'] if c in prices_df.columns]
        if {'timestamp_utc'}.issubset(price_cols):
            work_df = pd.merge_asof(
                work_df.sort_values('published_at'),
                prices_df[price_cols].sort_values('timestamp_utc'),
                left_on='published_at',
                right_on='timestamp_utc',
                direction='backward',
                tolerance=pd.Timedelta('15min'),
            )
            work_df = work_df.sort_values('_orig_index')
            if 'timestamp_utc' in work_df.columns:
                work_df.drop(columns=['timestamp_utc'], inplace=True)

    if daily_df is not None and not daily_df.empty:
        if 'timestamp_utc' in daily_df.columns:
            work_df = pd.merge_asof(
                work_df.sort_values('published_at'),
                daily_df.sort_values('timestamp_utc'),
                left_on='published_at',
                right_on='timestamp_utc',
                direction='backward',
                tolerance=pd.Timedelta('1D'),
            )
            work_df = work_df.sort_values('_orig_index')
            if 'timestamp_utc' in work_df.columns:
                work_df.drop(columns=['timestamp_utc'], inplace=True)

    # Ensure volatility/pre features
    if 'volatility_pre_15m' not in work_df.columns:
        if prices_df is not None and not prices_df.empty and 'timestamp_utc' in prices_df.columns:
            vol_15 = prices_df.resample('15min', on='timestamp_utc')['range_pips'].mean().to_frame('volatility_pre')
            vol_15 = vol_15.reset_index()
            work_df = pd.merge_asof(
                work_df.sort_values('published_at'),
                vol_15.sort_values('timestamp_utc'),
                left_on='published_at',
                right_on='timestamp_utc',
                direction='backward',
            )
            work_df = work_df.sort_values('_orig_index')
            work_df.rename(columns={'volatility_pre': 'volatility_pre_15m'}, inplace=True)
            if 'timestamp_utc' in work_df.columns:
                work_df.drop(columns=['timestamp_utc'], inplace=True)
        else:
            work_df['volatility_pre_15m'] = 0.0
    work_df['volatility_pre_15m'] = work_df['volatility_pre_15m'].astype(float)
    if work_df['volatility_pre_15m'].isna().all():
        work_df['volatility_pre_15m'] = work_df['volatility_pre_15m'].fillna(0.0)
    else:
        work_df['volatility_pre_15m'] = work_df['volatility_pre_15m'].fillna(
            work_df['volatility_pre_15m'].median()
        )

    # Additional feature engineering
    work_df['ret_pre_5m'] = work_df['ret_pre_5m'].astype(float).fillna(0.0)
    work_df['ret_pre_15m'] = work_df['ret_pre_15m'].astype(float).fillna(0.0)
    work_df['sign_ret15'] = np.sign(work_df['ret_pre_15m']).astype(int)
    denom = work_df['volatility_pre_15m'].replace(0, np.nan).fillna(work_df['volatility_pre_15m'].median())
    work_df['ret_pre_15m_norm'] = work_df['ret_pre_15m'] / denom
    work_df['accel_5_15'] = work_df['ret_pre_5m'] - (work_df['ret_pre_15m'] / 3.0)
    work_df['RSI_high'] = (work_df['RSI_14'] >= 60).astype(int)
    work_df['RSI_low'] = (work_df['RSI_14'] <= 40).astype(int)
    work_df['sent_pos_x_accel'] = np.clip(work_df['sent_finbert'], 0, None) * work_df['accel_5_15']

    for col in [
        'close', 'RSI_14', 'SMA_20', 'ATR_14', 'SMA_365', 'trend_365',
        'SMA_1M', 'SMA_3M', 'SMA_6M', 'SMA_12M', 'trend_1M', 'trend_3M', 'trend_6M', 'trend_12M',
        'SMA_3Q', 'SMA_6Q', 'SMA_12Q', 'trend_3Q', 'trend_6Q', 'trend_12Q',
    ]:
        if col in work_df.columns:
            if work_df[col].dtype.kind in 'biufc':
                work_df[col] = work_df[col].astype(float).fillna(work_df[col].median())
            else:
                work_df[col] = work_df[col].fillna(0)
        else:
            work_df[col] = 0.0

    work_df = work_df.sort_values('_orig_index')
    work_df.index = work_df.pop('_orig_index')
    work_df = work_df.reindex(news_df.index)

    meta_cols = ['article_id', 'published_at', 'event_key', 'fx_weight']
    meta = work_df[meta_cols].copy()
    X = work_df[FEATURES_V19].copy()
    return X, meta