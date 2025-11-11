#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import text

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE
import joblib
import pandas_ta as ta

# ================== CONFIG ==================
PROGRAM_DIR   = '/home/ilyamus/GPTGROKWORK'
MODEL_DIR     = f'{PROGRAM_DIR}/AITrainer_V5'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(f'{MODEL_DIR}/logs', exist_ok=True)

CURRENCY_PAIR = 'EURUSD'               # имя инструмента в ценовой таблице
PRICE_TABLE   = 'HistDataEURUSD'       # минутки EUR/USD
START_DATE    = '2008-01-01'           # можно с 2005, но 2008 быстрее
TIMEFRAME_FWD = 15                     # таргет: +15 минут
NEWS_PILE_THRESHOLD = 20               # отбрасываем «свалки» по точной метке времени

DB_CONNECTION = os.getenv(
    "DB_CONNECTION",
    "mysql+mysqlconnector://GPTFOREX:GPtushechkaForexUshechka@localhost/GPTFOREX"
)
# ============================================

#log_filename = f"{MODEL_DIR}/logs/train_model_TE_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_15min.log"
log_filename = f"{MODEL_DIR}/logs/train_model_TE_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_15min.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)sZ - %(levelname)s - %(message)s'
)
logging.info("==== TE V5 Trainer start ====")

# ---------- DB CONNECT ----------
logging.info("Connecting DB...")
engine = sqlalchemy.create_engine(DB_CONNECTION, pool_pre_ping=True, pool_recycle=3600)
with engine.begin() as conn:
    conn.execute(text("SET time_zone = '+00:00'"))
    conn.execute(text("SET NAMES utf8mb4"))
logging.info("DB OK")

# ---------- UTILS ----------
def slugify(s: str) -> str:
    if s is None:
        return ''
    s = str(s).strip()
    if s == '':
        return ''
    s = (s
         .replace('&amp;', '&')
         .replace('\u00A0', ' ')
         )
    s = s.lower()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append('_')
    s = ''.join(out)
    while '__' in s:
        s = s.replace('__', '_')
    return s.strip('_')

def build_event_key(categories: str, country: str) -> str:
    cat = (categories or '').split(',')[0].strip()
    cat_slug = slugify(cat) or 'misc'
    country_slug = slugify(country or '') or 'all'
    return f"te_{cat_slug}_{country_slug}"

def pips(x: float) -> float:
    return x * 10000.0

def attach_close_at(df: pd.DataFrame, time_col: str, out_col: str) -> pd.DataFrame:
    """Приклеить close на ближайший <= time_col с помощью merge_asof."""
    out = pd.merge_asof(
        df.sort_values(time_col),
        prices[['timestamp_utc','close']].sort_values('timestamp_utc'),
        left_on=time_col, right_on='timestamp_utc',
        direction='backward'
    ).rename(columns={'close': out_col}).drop(columns=['timestamp_utc'], errors='ignore')
    return out

# ---------- LOAD PRICES ----------
logging.info("Loading prices...")
q_prices = f"""
    SELECT timestamp_utc, open, high, low, close
    FROM {PRICE_TABLE}
    WHERE timestamp_utc >= :start
    ORDER BY timestamp_utc ASC
"""
prices = pd.read_sql(text(q_prices), engine, params={"start": START_DATE})
prices['timestamp_utc'] = pd.to_datetime(prices['timestamp_utc'], utc=True)
prices = prices.sort_values('timestamp_utc').reset_index(drop=True)

if prices.empty:
    raise RuntimeError("No price data loaded from PRICE_TABLE")

# feature prices
prices['high_pips']  = pips(prices['high'])
prices['low_pips']   = pips(prices['low'])
prices['close_pips'] = pips(prices['close'])

# Indicators
logging.info("Indicators (RSI/SMA/ATR)...")
prices['RSI_14'] = ta.rsi(prices['close'], length=14).fillna(50.0)
prices['SMA_20'] = ta.sma(prices['close'], length=20).fillna(prices['close'])
prices['ATR_14'] = ta.atr(prices['high_pips'], prices['low_pips'], prices['close_pips'], length=14)
if prices['ATR_14'].isna().any():
    prices['ATR_14'] = prices['ATR_14'].fillna(prices['close_pips'].rolling(200).std().bfill().fillna(10.0))

# Daily trends
logging.info("Daily trends (SMA/trend)...")
daily = prices.resample('1D', on='timestamp_utc').agg({'close': 'mean'}).dropna().reset_index()
for period_m in [1, 3, 6, 12]:
    daily[f'SMA_{period_m}M']    = ta.sma(daily['close'], length=max(2, period_m * 30)).bfill()
    daily[f'trend_{period_m}M']  = daily['close'].pct_change(periods=period_m * 30).fillna(0.0)
for period_q in [3, 6, 12]:
    daily[f'SMA_{period_q}Q']    = ta.sma(daily['close'], length=max(2, period_q * 30)).bfill()
    daily[f'trend_{period_q}Q']  = daily['close'].pct_change(periods=period_q * 30).fillna(0.0)
daily['SMA_365']  = ta.sma(daily['close'], length=365).bfill()
daily['trend_365'] = daily['close'].pct_change(periods=365).fillna(0.0)

# Pre-volatility (15m avg range)
prices['range_pips'] = pips(prices['high'] - prices['low'])
#vol_15 = prices.resample('15T', on='timestamp_utc')['range_pips'].mean().to_frame('volatility_pre').reset_index()
#log_filename = f"{MODEL_DIR}/logs/train_model_TE_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_15min.log"
# ---------- LOAD TE NEWS ----------
logging.info("Loading TE news...")
q_news = """
  SELECT
    article_id,
    published_at,
    tags,            -- country
    categories,      -- category
    sent_finbert,
    sent_label,
    text_clean,
    source_type,
    url
  FROM news_tradingeconomics
  WHERE published_at >= :start
  ORDER BY published_at ASC
"""
news = pd.read_sql(text(q_news), engine, params={"start": START_DATE})
if news.empty:
    raise RuntimeError("No news in news_tradingeconomics since START_DATE")

news['published_at'] = pd.to_datetime(news['published_at'], utc=True)
news = news.drop_duplicates(subset=['article_id']).reset_index(drop=True)

# event_key (совместимый с imp_cache_tradingeconomics)
news['event_key'] = [
    build_event_key(cat, country) for cat, country in zip(news['categories'], news['tags'])
]
news['year'] = news['published_at'].dt.year

# Плотность «свалок» по точному TS — отфильтруем совсем массовые точки
cnt_per_ts = news.groupby('published_at').size()
piles = cnt_per_ts[cnt_per_ts > NEWS_PILE_THRESHOLD].index
if len(piles) > 0:
    before = len(news)
    news = news[~news['published_at'].isin(piles)]
    logging.info(f"Filtered news piles >{NEWS_PILE_THRESHOLD}: {before-len(news)} removed")

# ---------- JOIN imp_cache_tradingeconomics ----------
logging.info("Loading imp_cache_tradingeconomics...")
q_imp = """
  SELECT currency, event_key, year, volatility, trend
  FROM imp_cache_tradingeconomics
  WHERE year >= :y0
"""
min_year = int(news['year'].min())
imp = pd.read_sql(text(q_imp), engine, params={"y0": min_year})
if imp.empty:
    logging.warning("imp_cache_tradingeconomics is empty or no rows >= min(news.year)")

# currency в imp_cache сейчас 'ALL' — оставляем, просто мержим по (event_key, year)
news = news.merge(
    imp[['event_key', 'year', 'volatility', 'trend']],
    on=['event_key', 'year'],
    how='left',
    suffixes=('', '_imp')
).rename(columns={'volatility': 'imp_year_vol', 'trend': 'imp_year_trend'})

# ---------- BUILD TARGETS FROM PRICES ----------
logging.info("Building targets (direction/magnitude) with +15m horizon...")

# close at t0 (last known before/at event)
news = pd.merge_asof(
    news.sort_values('published_at'),
    prices[['timestamp_utc', 'close']].sort_values('timestamp_utc'),
    left_on='published_at',
    right_on='timestamp_utc',
    direction='backward'
).rename(columns={'close': 'close_t0'}).drop(columns=['timestamp_utc'], errors='ignore')

# close at t+15m (last known before/at future mark)
news['t_plus'] = news['published_at'] + pd.to_timedelta(TIMEFRAME_FWD, unit='m')
news = pd.merge_asof(
    news.sort_values('t_plus'),
    prices[['timestamp_utc', 'close']].sort_values('timestamp_utc'),
    left_on='t_plus',
    right_on='timestamp_utc',
    direction='backward'
).rename(columns={'close': 'close_tplus'}).drop(columns=['timestamp_utc'], errors='ignore')

# --- 15-минутная волатильность из цен
# prices: (timestamp_utc, open, high, low, close) — уже загружена и отсортирована выше
prices['range_pips'] = (prices['high'] - prices['low']) * 10000.0

vol_15 = (
    prices
    .resample('15min', on='timestamp_utc')['range_pips']
    .mean()
    .to_frame('volatility_pre')
    .reset_index()
)

# attach pre-volatility
news = pd.merge_asof(
    news.sort_values('published_at'),
    vol_15.sort_values('timestamp_utc'),
    left_on='published_at',
    right_on='timestamp_utc',
    direction='backward'
).rename(columns={'volatility_pre':'volatility_pre_15m'}).drop(columns=['timestamp_utc'], errors='ignore')

# drop events with missing price on either side
news = news.dropna(subset=['close_t0', 'close_tplus']).reset_index(drop=True)

# --- цены за 5м и 15м до события
news['t_minus5']  = news['published_at'] - pd.to_timedelta(5,  'm')
news['t_minus15'] = news['published_at'] - pd.to_timedelta(15, 'm')

# приклеиваем close на моменты t_minus5 / t_minus15
news = pd.merge_asof(
    news.sort_values('t_minus5'),
    prices[['timestamp_utc','close']].sort_values('timestamp_utc'),
    left_on='t_minus5', right_on='timestamp_utc', direction='backward'
).rename(columns={'close':'close_pre5'}).drop(columns=['timestamp_utc'], errors='ignore')

news = pd.merge_asof(
    news.sort_values('t_minus15'),
    prices[['timestamp_utc','close']].sort_values('timestamp_utc'),
    left_on='t_minus15', right_on='timestamp_utc', direction='backward'
).rename(columns={'close':'close_pre15'}).drop(columns=['timestamp_utc'], errors='ignore')

# убрать возможные дубль-колонки и выровнять индекс
news = news.loc[:, ~news.columns.duplicated()].reset_index(drop=True)


# гарантия числового типа
news[['close_t0','close_pre5','close_pre15']] = (
    news[['close_t0','close_pre5','close_pre15']].apply(pd.to_numeric, errors='coerce')
)

# (не обязательно, но полезно для логов)
dups = [c for c in news.columns if (news.columns == c).sum() > 1]
if dups:
    logging.warning(f"Duplicate columns still present: {dups}")

news = pd.merge_asof(
    news.sort_values('t_minus5'),
    prices[['timestamp_utc','close']].sort_values('timestamp_utc'),
    left_on='t_minus5', right_on='timestamp_utc',
    direction='backward'
).rename(columns={'close':'close_pre5'}).drop(columns=['timestamp_utc'])

news = pd.merge_asof(
    news.sort_values('t_minus15'),
    prices[['timestamp_utc','close']].sort_values('timestamp_utc'),
    left_on='t_minus15', right_on='timestamp_utc',
    direction='backward'
).rename(columns={'close':'close_pre15'}).drop(columns=['timestamp_utc'])

# импульс до новости (в пипсах) и нормировка на локальную волу
#news['ret_pre_5m']  = pips(news['close_t0'] - news['close_pre5']).fillna(0.0)
#news['ret_pre_15m'] = pips(news['close_t0'] - news['close_pre15']).fillna(0.0)
news['ret_pre_5m']  = pips(news['close_t0'].to_numpy() - news['close_pre5'].to_numpy())
news['ret_pre_15m'] = pips(news['close_t0'].to_numpy() - news['close_pre15'].to_numpy())
news[['ret_pre_5m','ret_pre_15m']] = news[['ret_pre_5m','ret_pre_15m']].fillna(0.0)
news['ret_pre_15m_norm'] = news['ret_pre_15m'] / (news['volatility_pre_15m'].replace(0, np.nan)).fillna(
    news['volatility_pre_15m'].median()
)

# === Динамический горизонт: до следующей новости, но не дольше 60м ===
MAX_H = 60  # минут
# время следующей новости (по всей ленте TE; если хочешь по стране — делай groupby по tags)
news = news.sort_values('published_at').reset_index(drop=True)
news['t_next_news'] = news['published_at'].shift(-1)

# кап времени: published_at + 60м
news['t_cap'] = news['published_at'] + pd.to_timedelta(MAX_H, unit='m')

# берём поэлементно минимум (работает корректно для дат)
news['t_plus_dyn'] = pd.concat([news['t_next_news'], news['t_cap']], axis=1).min(axis=1)

# если до следующей новости менее 5 минут — считаем, что сигнал слишком короткий (можно отфильтровать)
MIN_H = 5
news['horizon_min'] = (news['t_plus_dyn'] - news['published_at']).dt.total_seconds() / 60.0
news = news[news['horizon_min'] >= MIN_H].reset_index(drop=True)

# подтянем цену в динамический момент
news = pd.merge_asof(
    news.sort_values('t_plus_dyn'),
    prices[['timestamp_utc','close']].sort_values('timestamp_utc'),
    left_on='t_plus_dyn', right_on='timestamp_utc',
    direction='backward'
).rename(columns={'close':'close_tplus_dyn'}).drop(columns=['timestamp_utc'], errors='ignore')

# Δ для динамического окна
news['delta_pips_dyn'] = pips(news['close_tplus_dyn'] - news['close_t0']).astype(float).fillna(0.0)
news['magnitude_pips_dyn'] = np.abs(news['delta_pips_dyn'])

# Подстройка порога под длину окна: масштабируем по sqrt(h/15) (диффузионная аппроксимация)
THR_MIN_PIPS = 1.0
THR_COEFF    = 0.35

base_thr = np.maximum(
    THR_MIN_PIPS,
    THR_COEFF * news['volatility_pre_15m'].fillna(news['volatility_pre_15m'].median())
)

scale = np.sqrt(np.clip(news['horizon_min'] / 15.0, 0.5, 4.0))
thr_dyn = (base_thr * scale).astype(float)

news['direction_cls_dyn'] = np.where(
    news['delta_pips_dyn'] >  thr_dyn, 2,
    np.where(news['delta_pips_dyn'] < -thr_dyn, 0, 1)
).astype(int)

# Δ in pips
news['delta_pips'] = pips(news['close_tplus'] - news['close_t0']).astype(float).fillna(0.0)
news['magnitude_pips'] = np.abs(news['delta_pips'])


# импульс до новости (в пипсах) и нормировка на локальную волу
news['ret_pre_5m']  = pips(news['close_t0'] - news['close_pre5']).fillna(0.0)
news['ret_pre_15m'] = pips(news['close_t0'] - news['close_pre15']).fillna(0.0)
news['ret_pre_15m_norm'] = news['ret_pre_15m'] / (news['volatility_pre_15m'].replace(0, np.nan)).fillna(
    news['volatility_pre_15m'].median()
)

# direction (3-классовый): порог = max(2.0, 0.5 * pre-vol)
#thr = np.maximum(3.0, 0.9 * news['volatility_pre_15m'].fillna(news['volatility_pre_15m'].median()))
#news['direction_cls'] = np.where(
#    news['delta_pips'] > thr, 2,
#    np.where(news['delta_pips'] < -thr, 0, 1)  # 0=down,1=flat,2=up
#).astype(int)


# --- гарантированно получить цены за 5м и 15м до события
if 't_minus5' not in news.columns:
    news['t_minus5']  = news['published_at'] - pd.to_timedelta(5,  unit='m')
if 't_minus15' not in news.columns:
    news['t_minus15'] = news['published_at'] - pd.to_timedelta(15, unit='m')

if 'close_pre5' not in news.columns:
    news = attach_close_at(news, 't_minus5',  'close_pre5')
if 'close_pre15' not in news.columns:
    news = attach_close_at(news, 't_minus15', 'close_pre15')

# нормализация: убрать дубликаты колонок, выровнять индекс
news = news.loc[:, ~news.columns.duplicated()].reset_index(drop=True)

# контроль: нужные колонки обязаны быть
needed = ['close_t0','close_pre5','close_pre15']
missing = [c for c in needed if c not in news.columns]
if missing:
    raise RuntimeError(f"Missing columns after merges: {missing}")

# приведение к числам
news[needed] = news[needed].apply(pd.to_numeric, errors='coerce')

# безопасная арифметика (по позиции), без выравнивания индексов
news['ret_pre_5m']  = pips(news['close_t0'].to_numpy() - news['close_pre5'].to_numpy())
news['ret_pre_15m'] = pips(news['close_t0'].to_numpy() - news['close_pre15'].to_numpy())
news[['ret_pre_5m','ret_pre_15m']] = news[['ret_pre_5m','ret_pre_15m']].fillna(0.0)

# нормировка 15м-ретёрна на локальную волу
news['ret_pre_15m_norm'] = news['ret_pre_15m'] / (
    news['volatility_pre_15m'].replace(0, np.nan)
).fillna(news['volatility_pre_15m'].median())

# === Порог для метки направления (меньше «flat») ===
THR_MIN_PIPS = 1.0
THR_COEFF    = 0.35
thr = np.maximum(
    THR_MIN_PIPS,
    THR_COEFF * news['volatility_pre_15m'].fillna(news['volatility_pre_15m'].median())
)
news['direction_cls'] = np.where(
    news['delta_pips'] >  thr, 2,
    np.where(news['delta_pips'] < -thr, 0, 1)
).astype(int)


# ---------- FEATURES ----------
logging.info("Building features...")

# базовые фичи времени
news['hour']        = news['published_at'].dt.hour.astype(int)
news['day_of_week'] = news['published_at'].dt.dayofweek.astype(int)  # 0=Mon

# sentiment
news['sent_finbert'] = news['sent_finbert'].astype(float).fillna(0.0)
news['sent_label']   = (news['sent_label'].astype(str).fillna('neu'))
news['sent_label_id'] = news['sent_label'].map({'neg':0,'neu':1,'pos':2}).fillna(1).astype(int)

# текстовые метрики
text_len = news['text_clean'].fillna('').astype(str).str.len()
news['text_len'] = text_len.clip(upper=text_len.quantile(0.99)).astype(int)

# event_key encoding
le_event = LabelEncoder()
news['event_key_encoded'] = le_event.fit_transform(news['event_key'].astype(str))
joblib.dump(le_event, f'{MODEL_DIR}/label_encoder_event_te_15m.pkl')

# корреляции из correlation_trends_v5 (опционально)
#logging.info("Loading correlation_trends...")
#q_corr = """
#  SELECT event_key, correlation, price_change, direction AS corr_direction, probability, observations
#  FROM correlation_trends
#"""
#corr = pd.read_sql(text(q_corr), engine)
#if not corr.empty:
#    corr_cols = ['correlation','price_change','corr_direction','probability','observations']
#    news = news.merge(corr[['event_key']+corr_cols], on='event_key', how='left')
#    news[corr_cols] = news[corr_cols].fillna(0.0)
#else:
#    news[['correlation','price_change','corr_direction','probability','observations']] = 0.0


# --- Корреляционные фичи: сначала пробуем v5 (по event_key), иначе фолбэк на currency/year.
logging.info("Loading correlation features...")
corr_cols = ['correlation','price_change','corr_direction','probability','observations']
for c in corr_cols:
    news[c] = 0.0

loaded_corr = False
try:
    # ПРАВИЛЬНАЯ таблица для event-level:
    q_corr_v5 = """
      SELECT event_key, correlation, price_change, direction AS corr_direction, probability, observations
      FROM correlation_trends_v5
    """
    corr_v5 = pd.read_sql(text(q_corr_v5), engine)
    if not corr_v5.empty and 'event_key' in corr_v5.columns:
        news = news.merge(corr_v5[['event_key']+corr_cols], on='event_key', how='left')
        for c in corr_cols:
            news[c] = news[c].fillna(0.0)
        loaded_corr = True
        logging.info("Loaded correlation_trends_v5 (event-level).")
except Exception as e:
    logging.warning(f"event-level correlations not usable: {e}")

if not loaded_corr:
    logging.info("Falling back to correlation_trends (currency/year).")
    corr_cy = pd.read_sql(text("SELECT currency, year, correlation FROM correlation_trends"), engine)
    if not corr_cy.empty:
        corr_cy['currency'] = corr_cy['currency'].str.upper()

        COUNTRY_TO_CCY = {
            # G10
            'United States':'USD','Euro Area':'EUR','Eurozone':'EUR','Germany':'EUR','France':'EUR','Italy':'EUR','Spain':'EUR','Netherlands':'EUR','Belgium':'EUR','Finland':'EUR','Ireland':'EUR','Portugal':'EUR','Greece':'EUR','Austria':'EUR',
            'United Kingdom':'GBP','Japan':'JPY','Canada':'CAD','Switzerland':'CHF','Australia':'AUD','New Zealand':'NZD','Norway':'NOK','Sweden':'SEK',
            # EM
            'China':'CNY','India':'INR','Russia':'RUB','Brazil':'BRL','Mexico':'MXN','South Africa':'ZAR','Turkey':'TRY','Poland':'PLN','Czech Republic':'CZK','Hungary':'HUF','Romania':'RON',
            'Korea':'KRW','South Korea':'KRW','Singapore':'SGD','Hong Kong':'HKD','Taiwan':'TWD','Thailand':'THB','Malaysia':'MYR','Indonesia':'IDR','Philippines':'PHP',
            # fallback
            'ALL':'ALL'
        }

        def map_country_to_ccy(country: str) -> str:
            if not isinstance(country, str) or not country.strip():
                return 'ALL'
            c = country.strip()
            if c.lower().startswith('euro area') or c.lower().startswith('euro zone'):
                return 'EUR'
            return COUNTRY_TO_CCY.get(c, 'ALL')

        # гарантируем timestamp_utc и производные
        if 'timestamp_utc' not in news.columns:
            news['timestamp_utc'] = pd.to_datetime(news['published_at'], utc=True).dt.tz_convert(None)
        news['news_year']     = pd.to_datetime(news['timestamp_utc']).dt.year
        news['news_currency'] = news['tags'].apply(map_country_to_ccy).str.upper().fillna('ALL')

        corr_cy = corr_cy.rename(columns={'correlation':'corr_by_currency'})
        news = news.merge(
            corr_cy, left_on=['news_currency','news_year'], right_on=['currency','year'], how='left'
        )

        corr_all = corr_cy[corr_cy['currency']=='ALL'][['year','corr_by_currency']].rename(
            columns={'corr_by_currency':'corr_all'}
        )
        news = news.merge(corr_all, left_on='news_year', right_on='year', how='left')

        news['correlation'] = news['corr_by_currency'].fillna(news['corr_all']).fillna(0.0)

        for c in ['currency','year','corr_by_currency','corr_all','year_x','year_y']:
            if c in news.columns:
                news.drop(columns=[c], inplace=True)
        logging.info("Applied currency/year correlation fallback.")


# приклеиваем быстродоступные рыночные фичи (RSI/SMA/ATR)
news = pd.merge_asof(
    news.sort_values('published_at'),
    prices[['timestamp_utc','close','RSI_14','SMA_20','ATR_14']].sort_values('timestamp_utc'),
    left_on='published_at',
    right_on='timestamp_utc',
    direction='backward',
    tolerance=pd.Timedelta('15min')
).drop(columns=['timestamp_utc'], errors='ignore')

# дневные тренды
news = pd.merge_asof(
    news.sort_values('published_at'),
    daily.sort_values('timestamp_utc'),
    left_on='published_at',
    right_on='timestamp_utc',
    direction='backward',
    tolerance=pd.Timedelta('1D')
).drop(columns=['timestamp_utc'], errors='ignore')

# pre-vol fill
news['volatility_pre_15m'] = news['volatility_pre_15m'].fillna(news['volatility_pre_15m'].median())

# fill market features
for col in ['close','RSI_14','SMA_20','ATR_14','SMA_365','trend_365',
            'SMA_1M','SMA_3M','SMA_6M','SMA_12M','trend_1M','trend_3M','trend_6M','trend_12M',
            'SMA_3Q','SMA_6Q','SMA_12Q','trend_3Q','trend_6Q','trend_12Q']:
    if col in news.columns:
        if news[col].dtype.kind in 'biufc':
            news[col] = news[col].astype(float).fillna(news[col].median())
        else:
            news[col] = news[col].fillna(0)

# импульсные фичи из imp_cache (годовые агрегаты)
news['imp_year_vol']   = news['imp_year_vol'].astype(float).fillna(0.0)
news['imp_year_trend'] = news['imp_year_trend'].astype(float).fillna(0.0)

# ---------- DATASET ----------
FEATURES = [
    # время/событие
    'hour','day_of_week','event_key_encoded','text_len',
    # сентимент
    'sent_finbert','sent_label_id',
    # импульс-агрегаты по году
    'imp_year_vol','imp_year_trend',
    # рынок вокруг события
    'volatility_pre_15m','RSI_14','SMA_20','ATR_14',
    # долгие тренды
    'SMA_365','trend_365',
    'SMA_1M','SMA_3M','SMA_6M','SMA_12M','trend_1M','trend_3M','trend_6M','trend_12M',
    'SMA_3Q','SMA_6Q','SMA_12Q','trend_3Q','trend_6Q','trend_12Q',
    # корр-слой (если есть)
    'correlation','price_change','corr_direction','probability','observations',
    # <<< новые >>>
    'ret_pre_5m','ret_pre_15m','ret_pre_15m_norm'
]

# drop rows with missing target
news = news.dropna(subset=['direction_cls','magnitude_pips']).reset_index(drop=True)

X = news[FEATURES].copy()
y_dir = news['direction_cls'].astype(int).values
y_mag = np.log1p(news['magnitude_pips'].astype(float).values)

# imputers
num_cols = X.select_dtypes(include=[np.number]).columns
imp_num = SimpleImputer(strategy='median')
X[num_cols] = imp_num.fit_transform(X[num_cols])

# ---------- TIME SPLIT (no leakage) ----------
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

def time_split_by_quantile(df_ts, X, y, ts_col='published_at', test_frac=0.2):
    split_date = df_ts[ts_col].quantile(1 - test_frac)
    mask = df_ts[ts_col] < split_date
    if mask.sum() == 0 or (~mask).sum() == 0:
        split_date = df_ts[ts_col].quantile(0.8)
        mask = df_ts[ts_col] < split_date
    if mask.sum() == 0 or (~mask).sum() == 0:
        split_date = df_ts[ts_col].quantile(0.7)
        mask = df_ts[ts_col] < split_date
    return X[mask].copy(), X[~mask].copy(), y[mask], y[~mask], split_date

# === Density metric: сколько новостей в окне ±15 минут вокруг каждой ===
win = pd.to_timedelta(15, unit='m')
starts = news['published_at'] - win
ends   = news['published_at'] + win

# Простое, но ОК по объёму решение: O(n^2) через IntervalIndex
iv = pd.IntervalIndex.from_arrays(starts, ends, closed='both')
counts = [iv.contains(ts).sum() for ts in news['published_at']]

news['news_count_pm15m'] = np.asarray(counts, dtype=int)
# буст для одиночных новостей (можешь варьировать 1.3–2.0)
news['singleton_boost'] = np.where(news['news_count_pm15m'] == 1, 1.5, 1.0)

X_train_dir, X_test_dir, y_train_dir, y_test_dir, SPLIT_DATE = time_split_by_quantile(
    news, X, y_dir, ts_col='published_at', test_frac=0.2
)
mask_time = (news['published_at'] < SPLIT_DATE)

# веса: одиночные важнее, «хвосты» классов важнее
w_single  = 1.25
w_group   = 0.75
is_single_train = (news.loc[mask_time, 'news_count_pm15m'] == 1).to_numpy()
w_train = np.where(is_single_train, w_single, w_group).astype(float)

# легкий буст редких направлений
class_boost = {0: 1.3, 1: 1.0, 2: 1.3}
w_train *= np.vectorize(class_boost.get)(y_train_dir)
# === Счётчик новостей в окне ±15м вокруг события ===
win = pd.to_timedelta(15, unit='m')
idx = news['published_at'].sort_values().reset_index(drop=True)

# на наших размерах можно простым O(n^2); если станет тяжело — заменим на двоичный поиск
counts = []
for t in news['published_at']:
    counts.append(((news['published_at'] >= t - win) & (news['published_at'] <= t + win)).sum())
news['news_count_pm15m'] = counts
# маска train по времени (пригодится и для регрессии, и для весов)

from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train_dir)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_dir)
cw_map = {c: w for c, w in zip(classes, class_weights)}

# базовые классовые веса + буст одиночек
w_train = np.array([cw_map[c] for c in y_dir[mask_time]])
w_train = w_train * news.loc[mask_time, 'singleton_boost'].to_numpy()
logging.info("Time split @ %s: train=%d, test=%d", SPLIT_DATE, len(X_train_dir), len(X_test_dir))
logging.info("Class balance (train): %s", Counter(y_train_dir))
logging.info("Class balance (test): %s", Counter(y_test_dir))

# Веса классов вместо SMOTE
classes = np.unique(y_train_dir)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_dir)
cls_weight_map = {c:w for c,w in zip(classes, class_weights)}
w_train = np.array([cls_weight_map[c] for c in y_train_dir])

# Бэйзлайны
maj = pd.Series(y_train_dir).mode()[0]
base_acc = (y_test_dir == maj).mean()
logging.info("Baseline (majority=%s) acc=%.4f", maj, base_acc)

# ---------- MODELS: DIRECTION ----------
logging.info("Train RandomForest (direction, class_weight=balanced)...")
rf_clf = RandomForestClassifier(
    n_estimators=300, max_depth=16, min_samples_split=20,
    random_state=42, n_jobs=-1, class_weight=None   # class_weight уже учтём в sample_weight
)
rf_clf.fit(X_train_dir, y_train_dir, sample_weight=w_train)

logging.info("GridSearch XGBClassifier (direction, sample_weight)...")
xgb_clf = XGBClassifier(random_state=42, eval_metric='mlogloss', nthread=-1)
param_grid_xgb = {
    'n_estimators': [200, 300],
    'learning_rate': [0.05, 0.1],
    'max_depth': [4, 6]
}
grid_xgb = GridSearchCV(
    xgb_clf, param_grid_xgb,
    cv=3, scoring='accuracy', n_jobs=-1, verbose=0
)
# Передаём sample_weight прямо в fit GridSearchCV — он корректно раскинет по фолдам
grid_xgb.fit(X_train_dir, y_train_dir, sample_weight=w_train)

best_xgb = grid_xgb.best_estimator_
best_xgb.fit(X_train_dir, y_train_dir, sample_weight=w_train)

#grid_xgb.fit(X_train_dir, y_train_dir, **{'xgbclassifier__sample_weight': w_train}
#             if hasattr(grid_xgb, 'best_estimator_') else {'sample_weight': w_train})
#best_xgb = grid_xgb.best_estimator_
try:
    best_xgb.fit(X_train_dir, y_train_dir, sample_weight=w_train)
except TypeError:
    # если версия xgboost в GridSearch завернула эстиматор, fallback без веса
    best_xgb.fit(X_train_dir, y_train_dir)

logging.info("Best XGB params: %s", grid_xgb.best_params_)


# Ensemble on test
rf_probs  = rf_clf.predict_proba(X_test_dir)
xgb_probs = best_xgb.predict_proba(X_test_dir)
ens_probs = (rf_probs + xgb_probs) / 2.0
ens_pred  = ens_probs.argmax(axis=1)

from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test_dir, ens_pred, labels=[0,1,2])
logging.info(f"\nConfusion matrix [rows=true (0,1,2), cols=pred]:\n{cm}")
logging.info("\n" + classification_report(y_test_dir, ens_pred, digits=3))


acc = accuracy_score(y_test_dir, ens_pred)
f1  = f1_score(y_test_dir, ens_pred, average='macro')
logging.info(f"Ensemble (RF+XGB) direction: acc={acc:.4f} f1={f1:.4f}")

# save direction models + metadata
joblib.dump(rf_clf,   f'{MODEL_DIR}/model_te_direction_rf_15m.pkl')
joblib.dump(best_xgb, f'{MODEL_DIR}/model_te_direction_xgb_15m.pkl')
with open(f'{MODEL_DIR}/features_direction_15m.txt', 'w') as f:
    f.write('\n'.join(FEATURES))

# ---------- MODELS: MAGNITUDE ----------
# без SMOTE, регрессия на лог-масштабе
#X_train_mag, X_test_mag, y_train_mag, y_test_mag = train_test_split(
#    X, y_mag, test_size=0.2, random_state=42
#)
# без SMOTE, регрессия: тот же time-split
#X_train_mag, X_test_mag = X[mask].copy(), X[~mask].copy()
#y_train_mag, y_test_mag = y_mag[mask], y_mag[~mask]
X_train_mag, X_test_mag = X[mask_time].copy(), X[~mask_time].copy()
y_train_mag, y_test_mag = y_mag[mask_time], y_mag[~mask_time]

logging.info("GridSearch XGBRegressor (magnitude)...")
reg = XGBRegressor(random_state=42, nthread=-1)
param_grid_reg = {
    'n_estimators': [200, 400],
    'learning_rate': [0.05, 0.1],
    'max_depth': [4, 6]
}
grid_reg = GridSearchCV(
    reg, param_grid_reg,
    cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0
)
grid_reg.fit(X_train_mag, y_train_mag)
best_reg = grid_reg.best_estimator_
logging.info("Best XGBReg params: %s", grid_reg.best_params_)

y_pred_test = best_reg.predict(X_test_mag)
rmse = np.sqrt(mean_squared_error(np.expm1(y_test_mag), np.expm1(y_pred_test)))
logging.info(f"RMSE magnitude (pips) @+{TIMEFRAME_FWD}m: {rmse:.4f}")

joblib.dump(best_reg, f'{MODEL_DIR}/model_te_magnitude_xgb_15m.pkl')
with open(f'{MODEL_DIR}/features_magnitude_15m.txt', 'w') as f:
    f.write('\n'.join(FEATURES))

logging.info("==== TE V5 Trainer done ====")
print("Done. Logs:", log_filename)