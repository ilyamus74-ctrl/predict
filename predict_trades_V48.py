# predict_trades_V47.py
import pandas as pd
import numpy as np
import joblib
import sqlalchemy
from datetime import datetime, timedelta, UTC
import os
import logging
import pandas_ta as ta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Настройки ---
# FUTURE_MODE: True = прогнозы для будущих новостей (future_trades.csv), False = история (trades.csv)
FUTURE_MODE = True
# USE_DATABASE: True = брать новости из базы MySQL, False = из CSV-файлов (news.csv, future_news.csv)
USE_DATABASE = True
# DB_CONNECTION: строка подключения к базе MySQL (имя:пароль@хост/база)
DB_CONNECTION = "mysql+mysqlconnector://GPTFOREX:GPtushechkaForexUshechka@localhost/GPTFOREX"
# IMP_TOTAL_THRESHOLD: минимальная важность новости для будущих сигналов (0.01 = брать новости с imp_total > 0.01)
IMP_TOTAL_THRESHOLD = 0.1
# FUTURE_DIRECTION_PROB_THRESHOLD: минимальная уверенность модели для будущих сигналов (0.4 = ≥40%)
FUTURE_DIRECTION_PROB_THRESHOLD = 0.4
# FUTURE_MAGNITUDE_THRESHOLD: минимальное движение цены для будущих сигналов (2.5 = ≥2.5 пунктов)
FUTURE_MAGNITUDE_THRESHOLD = 2.5
# HISTORY_DIRECTION_PROB_THRESHOLD: минимальная уверенность для истории (0.65 = ≥65%)
HISTORY_DIRECTION_PROB_THRESHOLD = 0.65
# HISTORY_MAGNITUDE_THRESHOLD: минимальное движение цены для истории (7.0 = ≥7 пунктов)
HISTORY_MAGNITUDE_THRESHOLD = 7.0
# GROUP_WINDOW_MINUTES: окно для группировки сигналов (30 минут = объединять сигналы в 30-минутные окна)
GROUP_WINDOW_MINUTES = 30
# NEWS_LIMIT: максимум новостей для обработки (None = без лимита)
NEWS_LIMIT = 10000
# START_DATE: фильтр новостей для истории (с 2005-01-01)
START_DATE = '2005-01-01'
# PRICE_START_DATE: фильтр цен для ускорения загрузки (с 2022-01-01)
PRICE_START_DATE = '2022-01-01'
# SAVE_NEWS_WITH_FEATURES: сохранять новости с фичами в news_with_features.csv? (False = не сохранять)
SAVE_NEWS_WITH_FEATURES = False
# CURRENCY_PAIR: валютная пара для сигналов (EUR/USD = торговать EUR/USD)
CURRENCY_PAIR = 'EUR/USD'
# MAGNITUDE_THRESHOLD: порог для определения приоритета сигнала (6.6227 = порог из модели V29)
MAGNITUDE_THRESHOLD = 6.6227

# --- Пути к файлам ---
PROGRAM_DIR = '/home/ilyamus/GPTGROKWORK'
news_csv = f'{PROGRAM_DIR}/AITrainer_V2/news.csv'
future_news_csv = f'{PROGRAM_DIR}/AITrainer_V2/future_news.csv'
prices_csv = f'{PROGRAM_DIR}/AITrainer_V2/prices.csv'
output_csv = f'{PROGRAM_DIR}/AITrainer_V2/historical_trades_{datetime.now(UTC).strftime("%Y%m%d")}.csv' if not FUTURE_MODE else f'{PROGRAM_DIR}/AITrainer_V2/future_trades.csv'
news_with_features_csv = f'{PROGRAM_DIR}/AITrainer_V2/news_with_features.csv'
label_encoder_event_path = f'{PROGRAM_DIR}/AITrainer_V2/label_encoder_event_15min.pkl'
label_encoder_dep_path = f'{PROGRAM_DIR}/AITrainer_V2/label_encoder_dependence_15min.pkl'

# --- Настройка логирования ---
log_dir = f'{PROGRAM_DIR}/AITrainer_V2/logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir, mode=0o775)
log_file = os.path.join(log_dir, 'predict_trades.log')
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Подключение к базе ---
if USE_DATABASE:
    engine = sqlalchemy.create_engine(DB_CONNECTION)

# --- Выгрузка цен ---
logging.info("Проверяем цены...")
prices_query = f"""
    SELECT timestamp_utc, open, high, low, close
    FROM HistDataEURUSD
    WHERE timestamp_utc BETWEEN '{PRICE_START_DATE}' AND NOW()
"""
if USE_DATABASE:
    prices = pd.read_sql(prices_query, engine)
    prices['timestamp_utc'] = pd.to_datetime(prices['timestamp_utc'])
    if not os.path.exists(prices_csv):
        logging.info("Файл цен не существует, выгружаем цены...")
        prices.to_csv(prices_csv, index=False)
        logging.info(f"Цены выгружены: {len(prices)} строк в {prices_csv}")
    else:
        existing_prices = pd.read_csv(prices_csv)
        existing_prices['timestamp_utc'] = pd.to_datetime(existing_prices['timestamp_utc'])
        max_existing_time = existing_prices['timestamp_utc'].max()
        max_db_time = prices['timestamp_utc'].max()
        if max_db_time > max_existing_time:
            logging.info("Обновляем цены: новые данные в базе...")
            prices.to_csv(prices_csv, index=False)
            logging.info(f"Цены обновлены: {len(prices)} строк в {prices_csv}")
        else:
            logging.info("Цены актуальны, загружаем из файла...")
            prices = existing_prices
            logging.info(f"Цены загружены из {prices_csv}: {len(prices)} строк")
else:
    prices = pd.read_csv(prices_csv)
    prices['timestamp_utc'] = pd.to_datetime(prices['timestamp_utc'])
    logging.info(f"Цены загружены из {prices_csv}: {len(prices)} строк")
prices['timestamp_utc'] = prices['timestamp_utc'].dt.round('min')
prices = prices.sort_values('timestamp_utc')

# --- Расчёт индикаторов ---
logging.info("Считаем индикаторы (RSI, SMA, ATR)...")
prices['high_pips'] = prices['high'] * 10000
prices['low_pips'] = prices['low'] * 10000
prices['close_pips'] = prices['close'] * 10000
prices['RSI_14'] = ta.rsi(prices['close'], length=14).fillna(50)
prices['SMA_20'] = ta.sma(prices['close'], length=20).fillna(prices['close'])
prices['ATR_14'] = ta.atr(prices['high_pips'], prices['low_pips'], prices['close_pips'], length=14).fillna(prices['close_pips'].std())
logging.info("Индикаторы рассчитаны")

# --- Расчёт трендов (месячных, квартальных, годовых) ---
logging.info("Считаем тренды (месячные, квартальные, годовые)...")
daily_prices = prices.resample('1D', on='timestamp_utc').agg({'close': 'mean'}).dropna().reset_index()

for period in [1, 3, 6, 12]:  # Месяцы
    daily_prices[f'SMA_{period}M'] = ta.sma(daily_prices['close'], length=period * 30).bfill()
    daily_prices[f'trend_{period}M'] = daily_prices['close'].pct_change(periods=period * 30).fillna(0)

for period in [3, 6, 12]:  # Кварталы (в месяцах)
    daily_prices[f'SMA_{period}Q'] = ta.sma(daily_prices['close'], length=period * 30).bfill()
    daily_prices[f'trend_{period}Q'] = daily_prices['close'].pct_change(periods=period * 30).fillna(0)

daily_prices['SMA_365'] = ta.sma(daily_prices['close'], length=365).bfill()
daily_prices['trend_365'] = daily_prices['close'].pct_change(periods=365).fillna(0)
logging.info("Тренды рассчитаны")

# --- Загрузка моделей и LabelEncoder ---
logging.info("Загружаем модели и LabelEncoder...")
rf_clf = joblib.load(f'{PROGRAM_DIR}/AITrainer_V2/model_direction_rf_15min.pkl')
xgb_clf = joblib.load(f'{PROGRAM_DIR}/AITrainer_V2/model_direction_xgb_15min.pkl')
reg = joblib.load(f'{PROGRAM_DIR}/AITrainer_V2/model_magnitude_15min.pkl')
le_event = joblib.load(label_encoder_event_path)
le_dep = joblib.load(label_encoder_dep_path)

# --- Загрузка новостей ---
logging.info("Загружаем новости...")
if FUTURE_MODE:
    df = pd.DataFrame()
    if USE_DATABASE:
        try:
            removed_query = """
                SELECT id, imp_total
                FROM news_removed_log
                WHERE removed_at >= UTC_TIMESTAMP() - INTERVAL 2 DAY
            """
            df_removed = pd.read_sql(removed_query, engine)
            removed_ids = set(df_removed[df_removed['imp_total'] > 0.1]['id'])
            logging.info(f"Загружено удалённых новостей с imp_total > 0.1: {len(removed_ids)}")

            news_query = """
                SELECT actual_minus_forecast, imp_calculated, imp_trend, imp_total, dependence,
                       HOUR(timestamp_utc) AS hour, DAYOFWEEK(timestamp_utc) AS day_of_week, actual,
                       direction, magnitude, timestamp_utc, id, event, event_key
                FROM economic_news_model_grok
                WHERE timestamp_utc >= UTC_TIMESTAMP() - INTERVAL 3 HOUR
                AND imp_total > %s
                AND event_key IS NOT NULL
            """ % IMP_TOTAL_THRESHOLD
            logging.info(f"Выполняем запрос: {news_query}")
            df = pd.read_sql(news_query, engine)
            df['is_removed'] = df['id'].isin(removed_ids).astype(int)
            logging.info(f"Из базы: {len(df)} строк")
            if not df.empty:
                logging.info("Найденные новости:")
                logging.info(df[['timestamp_utc', 'imp_total', 'dependence', 'magnitude', 'event', 'event_key', 'is_removed']].to_string())
        except Exception as e:
            logging.error(f"Ошибка базы: {e}")
    if df.empty and os.path.exists(future_news_csv):
        try:
            df = pd.read_csv(future_news_csv)
            if 'event_key' not in df.columns:
                logging.warning(f"Столбец 'event_key' отсутствует в {future_news_csv}. Пропускаем.")
                df = pd.DataFrame()
            else:
                df = df[df['timestamp_utc'] >= (datetime.now(UTC) - timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S')]
                df['is_removed'] = 0
                logging.info(f"Из {future_news_csv}: {len(df)} строк")
        except Exception as e:
            logging.error(f"Ошибка {future_news_csv}: {e}")
    if df.empty:
        logging.warning("Нет будущих новостей. Создайте future_news.csv или добавьте в базу.")
else:
    if USE_DATABASE:
        try:
            news_query = """
                SELECT actual_minus_forecast, imp_calculated, imp_trend, imp_total, dependence,
                       HOUR(timestamp_utc) AS hour, DAYOFWEEK(timestamp_utc) AS day_of_week, actual,
                       direction, magnitude, timestamp_utc, id, event, event_key
                FROM economic_news_model_grok
                WHERE imp_total > 0.1
                AND event_key IS NOT NULL
            """
            if START_DATE:
                news_query += f" AND timestamp_utc >= '{START_DATE}'"
            logging.info(f"Выполняем запрос: {news_query}")
            df = pd.read_sql(news_query, engine)
            df['is_removed'] = 0
        except Exception as e:
            logging.error(f"Ошибка базы: {e}")
            df = pd.DataFrame()
    else:
        if os.path.exists(news_csv):
            try:
                df = pd.read_csv(news_csv)
                if 'event_key' not in df.columns:
                    logging.error(f"Столбец 'event_key' отсутствует в {news_csv}. Завершаем.")
                    df = pd.DataFrame()
                else:
                    df['is_removed'] = 0
                    logging.info(f"{news_csv}: {len(df)} строк до фильтрации")
                    logging.info("Первые строки:\n%s", df.head().to_string())
            except Exception as e:
                logging.error(f"Ошибка {news_csv}: {e}")
                df = pd.DataFrame()
        else:
            logging.error(f"Файл {news_csv} не найден.")
            df = pd.DataFrame()
        if START_DATE and not df.empty:
            df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], errors='coerce')
            df = df[df['timestamp_utc'] >= START_DATE]
            logging.info(f"После фильтра START_DATE: {len(df)} строк")
logging.info(f"Новости: {len(df)} строк")

# --- Проверка на пустые данные ---
if df.empty:
    logging.warning("Нет новостей для обработки, завершаем.")
    df_grouped = pd.DataFrame(columns=['timestamp_utc', 'direction_pred', 'magnitude_pred', 'price_entry', 'price_exit', 'direction_prob', 'currency_pair', 'event', 'is_removed', 'priority'])
    df_grouped.to_csv(output_csv, index=False)
    logging.info(f"Сохранено: {output_csv} (пусто)")
    exit(0)

# --- Проверка event_key ---
logging.info(f"Столбцы в df: {list(df.columns)}")
if 'event_key' not in df.columns:
    logging.error("Столбец 'event_key' отсутствует в данных. Завершаем.")
    exit(1)

# --- Обработка NaN в event_key и dependence ---
logging.info("Обрабатываем event_key и dependence...")
df['event_key'] = df['event_key'].fillna('unknown').astype(str)
df['dependence'] = df['dependence'].fillna('unknown').astype(str)

# --- Кодирование event_key ---
logging.info("Кодируем event_key...")
try:
    df['event_key_encoded'] = le_event.transform(df['event_key'])
except ValueError as e:
    logging.warning(f"Неизвестные event_key: {e}. Присваиваем -1 для новых значений.")
    known_labels = le_event.classes_
    df['event_key_encoded'] = df['event_key'].apply(
        lambda x: le_event.transform([x])[0] if x in known_labels else -1
    )

# --- Кодирование dependence ---
logging.info("Кодируем dependence...")
try:
    df['dependence_encoded'] = le_dep.transform(df['dependence'])
except ValueError as e:
    logging.warning(f"Неизвестные dependence: {e}. Присваиваем -1 для новых значений.")
    known_labels = le_dep.classes_
    df['dependence_encoded'] = df['dependence'].apply(
        lambda x: le_dep.transform([x])[0] if x in known_labels else -1
    )

# --- Форматирование времени ---
df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], errors='coerce')
df['timestamp_utc'] = df['timestamp_utc'].dt.round('min')
df['id'] = df['id'].astype(np.int64)

# --- Загрузка данных из correlation_trends_v2 ---
logging.info("Загружаем данные из correlation_trends_v2...")
query_corr = f"""
    SELECT timestamp_utc, event_key, currency, correlation, price_change, direction AS corr_direction, 
           probability, observations
    FROM correlation_trends_v2
    WHERE timestamp_utc >= '2006-10-06'
    AND observations > 100
    AND probability > 0.5
"""
df_corr = pd.read_sql(query_corr, engine)
df_corr['timestamp_utc'] = pd.to_datetime(df_corr['timestamp_utc'])
logging.info(f"Данные из correlation_trends_v2 загружены: {len(df_corr)} строк")

# --- Объединяем данные ---
logging.info("Объединяем данные...")
df = pd.merge(df, df_corr, on=['timestamp_utc', 'event_key'], how='left')
logging.info(f"После объединения: {len(df)} строк")

# --- Обработка пропусков ---
logging.info("Обрабатываем пропуски...")
df['correlation'] = df['correlation'].fillna(0)
df['price_change'] = df['price_change'].fillna(df['magnitude'].mean())
df['corr_direction'] = df['corr_direction'].fillna(df['direction'])
df['probability'] = df['probability'].fillna(0.5)
df['observations'] = df['observations'].fillna(0)
logging.info(f"NaN после обработки: correlation={df['correlation'].isna().sum()}, "
             f"price_change={df['price_change'].isna().sum()}, "
             f"corr_direction={df['corr_direction'].isna().sum()}, "
             f"probability={df['probability'].isna().sum()}, "
             f"observations={df['observations'].isna().sum()}")

# --- Расчёт time_since_last_event ---
logging.info("Считаем time_since_last_event...")
df['time_since_last_event'] = df['timestamp_utc'].diff().dt.total_seconds() / 60.0
df['time_since_last_event'] = df['time_since_last_event'].fillna(0)

# --- Расчёт лаговых фич ---
logging.info("Добавляем лаговые фичи...")
df['prev_magnitude_1'] = df['magnitude'].shift(1).fillna(0)
df['prev_direction_1'] = df['direction'].shift(1).fillna(0)
df['prev_magnitude_2'] = df['magnitude'].shift(2).fillna(0)
df['prev_direction_2'] = df['direction'].shift(2).fillna(0)
df['prev_magnitude_3'] = df['magnitude'].shift(3).fillna(0)
df['prev_direction_3'] = df['direction'].shift(3).fillna(0)

# --- Категоризация imp_total ---
logging.info("Категоризируем imp_total...")
df['imp_total_category'] = pd.cut(
    df['imp_total'],
    bins=[0, 0.3, 0.6, 1.0],
    labels=[0, 1, 2],
    include_lowest=True
).astype(int)

# --- Расчёт volatility_pre ---
logging.info("Считаем volatility_pre...")
prices['price_range'] = (prices['high'] - prices['low']) * 10000
volatility = prices.groupby(pd.Grouper(key='timestamp_utc', freq='15min'))['price_range'].mean().reset_index()
volatility = volatility.rename(columns={'price_range': 'volatility_pre'})
df = pd.merge_asof(
    df.sort_values('timestamp_utc'),
    volatility.sort_values('timestamp_utc'),
    on='timestamp_utc',
    direction='backward'
)
df['volatility_pre'] = df['volatility_pre'].fillna(df['volatility_pre'].mean())
logging.info(f"volatility_pre статистика: min={df['volatility_pre'].min():.2f}, max={df['volatility_pre'].max():.2f}, mean={df['volatility_pre'].mean():.2f}")

# --- Расчёт news_impact с многопоточностью ---
logging.info("Считаем news_impact в окне ±15 минут с многопоточностью...")
df['window_start'] = df['timestamp_utc'] - pd.Timedelta(minutes=15)
df['window_end'] = df['timestamp_utc'] + pd.Timedelta(minutes=15)

def calculate_news_impact(row, df):
    window_df = df[(df['timestamp_utc'] >= row['window_start']) & (df['timestamp_utc'] <= row['window_end'])]
    return window_df['imp_total'].sum() if not window_df.empty else 0

def parallel_news_impact(df, num_threads=4):
    results = [None] * len(df)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(calculate_news_impact, row, df): idx for idx, row in df.iterrows()}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    return results

df['news_impact'] = parallel_news_impact(df, num_threads=4)
df['news_impact'] = np.log1p(df['news_impact'])
df = df.drop(columns=['window_start', 'window_end'])
logging.info(f"news_impact статистика: min={df['news_impact'].min():.2f}, max={df['news_impact'].max():.2f}, mean={df['news_impact'].mean():.2f}")

# --- Объединение с ценами для индикаторов ---
logging.info("Объединяем с ценами для индикаторов...")
df = pd.merge_asof(
    df.sort_values('timestamp_utc'),
    prices[['timestamp_utc', 'close', 'RSI_14', 'SMA_20', 'ATR_14']].sort_values('timestamp_utc'),
    on='timestamp_utc',
    direction='backward',
    tolerance=pd.Timedelta('15min')
)
df['close'] = df['close'].fillna(df['close'].mean())
df['RSI_14'] = df['RSI_14'].fillna(50)
df['SMA_20'] = df['SMA_20'].fillna(df['close'])
df['ATR_14'] = df['ATR_14'].fillna(df['ATR_14'].mean())
logging.info(f"RSI_14 статистика: min={df['RSI_14'].min():.2f}, max={df['RSI_14'].max():.2f}, mean={df['RSI_14'].mean():.2f}")
logging.info(f"ATR_14 статистика: min={df['ATR_14'].min():.2f}, max={df['ATR_14'].max():.2f}, mean={df['ATR_14'].mean():.2f}")

# --- Добавление трендов ---
logging.info("Добавляем тренды в новости...")
df = pd.merge_asof(
    df.sort_values('timestamp_utc'),
    daily_prices[['timestamp_utc', 'SMA_365', 'trend_365',
                  'SMA_1M', 'SMA_3M', 'SMA_6M', 'SMA_12M', 'trend_1M', 'trend_3M', 'trend_6M', 'trend_12M',
                  'SMA_3Q', 'SMA_6Q', 'SMA_12Q', 'trend_3Q', 'trend_6Q', 'trend_12Q']].sort_values('timestamp_utc'),
    on='timestamp_utc',
    direction='backward',
    tolerance=pd.Timedelta('1D')
)

# Заполнение пропусков для трендов
df['SMA_365'] = df['SMA_365'].fillna(df['SMA_365'].mean())
df['SMA_1M'] = df['SMA_1M'].fillna(df['SMA_1M'].mean())
df['SMA_3M'] = df['SMA_3M'].fillna(df['SMA_3M'].mean())
df['SMA_6M'] = df['SMA_6M'].fillna(df['SMA_6M'].mean())
df['SMA_12M'] = df['SMA_12M'].fillna(df['SMA_12M'].mean())
df['SMA_3Q'] = df['SMA_3Q'].fillna(df['SMA_3Q'].mean())
df['SMA_6Q'] = df['SMA_6Q'].fillna(df['SMA_6Q'].mean())
df['SMA_12Q'] = df['SMA_12Q'].fillna(df['SMA_12Q'].mean())
df['trend_365'] = df['trend_365'].fillna(0)
df['trend_1M'] = df['trend_1M'].fillna(0)
df['trend_3M'] = df['trend_3M'].fillna(0)
df['trend_6M'] = df['trend_6M'].fillna(0)
df['trend_12M'] = df['trend_12M'].fillna(0)
df['trend_3Q'] = df['trend_3Q'].fillna(0)
df['trend_6Q'] = df['trend_6Q'].fillna(0)
df['trend_12Q'] = df['trend_12Q'].fillna(0)

# --- Заполнение пропусков для будущих новостей ---
if FUTURE_MODE:
    for col in ['actual_minus_forecast', 'actual', 'direction', 'magnitude']:
        if col in df.columns:
            df[col] = df[col].fillna(0.0).infer_objects(copy=False)

# --- Подготовка фич для модели ---
logging.info("Готовим фичи...")
FEATURES = [
    'actual_minus_forecast', 'imp_calculated', 'imp_trend', 'imp_total', 'dependence_encoded',
    'hour', 'day_of_week', 'actual', 'news_impact', 'volatility_pre', 'event_key_encoded',
    'prev_magnitude_1', 'prev_direction_1', 'prev_magnitude_2', 'prev_direction_2',
    'prev_magnitude_3', 'prev_direction_3', 'imp_total_category', 'RSI_14', 'SMA_20',
    'time_since_last_event', 'ATR_14',
    'correlation', 'price_change', 'corr_direction', 'probability', 'observations',
    'SMA_365', 'trend_365', 'SMA_1M', 'SMA_3M', 'SMA_6M', 'SMA_12M', 'trend_1M', 'trend_3M', 'trend_6M', 'trend_12M',
    'SMA_3Q', 'SMA_6Q', 'SMA_12Q', 'trend_3Q', 'trend_6Q', 'trend_12Q'
]
X = df[FEATURES].copy()
X = X.fillna(X.mean(numeric_only=True))  # Заполняем пропуски средними значениями

# --- Предсказания ---
logging.info("Предсказываем...")
try:
    print("Запускаем предсказания для direction (ансамбль RandomForest + XGBoost)...")
    rf_probs = rf_clf.predict_proba(X)
    xgb_probs = xgb_clf.predict_proba(X)
    ensemble_probs = (rf_probs + xgb_probs) / 2
    df['direction_pred'] = np.argmax(ensemble_probs, axis=1)
    df['direction_prob'] = ensemble_probs.max(axis=1)
    print("Предсказания для direction завершены")
    df['magnitude_pred'] = np.expm1(np.maximum(0, reg.predict(X)))
    print("Предсказания для magnitude завершены")
except Exception as e:
    logging.error(f"Ошибка в предсказаниях: {e}")
    print(f"Ошибка в предсказаниях: {e}")
    raise

df['price_entry'] = df['close']
df['price_exit'] = df['price_entry'] + df['magnitude_pred'] * (2 * df['direction_pred'] - 1) * 0.0001
df['currency_pair'] = CURRENCY_PAIR

# --- Определение приоритета ---
df['priority'] = df['magnitude_pred'].apply(lambda x: 'High' if x >= MAGNITUDE_THRESHOLD else 'Low')

# --- Сохранение news_with_features.csv (если включено) ---
if SAVE_NEWS_WITH_FEATURES:
    df.to_csv(news_with_features_csv, index=False)
    logging.info(f"Сохранено: {news_with_features_csv}")

# --- Отладка ---
logging.info(f"До фильтров: {len(df)} новостей")
logging.info(f"Баланс direction_pred: {df['direction_pred'].value_counts().to_dict()}")
logging.info(f"Средняя direction_prob для 0: {df[df['direction_pred'] == 0]['direction_prob'].mean():.2f}")
logging.info(f"Средняя direction_prob для 1: {df[df['direction_pred'] == 1]['direction_prob'].mean():.2f}")
logging.info(f"direction_prob: min={df['direction_prob'].min():.2f}, max={df['direction_prob'].max():.2f}, mean={df['direction_prob'].mean():.2f}")
logging.info(f"magnitude_pred: min={df['magnitude_pred'].min():.2f}, max={df['magnitude_pred'].max():.2f}, mean={df['magnitude_pred'].mean():.2f}")

# --- Фильтры ---
DIRECTION_PROB_THRESHOLD = FUTURE_DIRECTION_PROB_THRESHOLD if FUTURE_MODE else HISTORY_DIRECTION_PROB_THRESHOLD
MAGNITUDE_THRESHOLD_FILTER = FUTURE_MAGNITUDE_THRESHOLD if FUTURE_MODE else HISTORY_MAGNITUDE_THRESHOLD
df_filtered = df[(df['direction_prob'] > DIRECTION_PROB_THRESHOLD) & (df['magnitude_pred'] > MAGNITUDE_THRESHOLD_FILTER)].copy()
logging.info(f"После фильтров: {len(df_filtered)} сигналов")

# --- Группировка сигналов ---
if not df_filtered.empty:
    logging.info("Группируем сигналы...")
    df_filtered['is_removed'] = df_filtered.apply(
        lambda x: 1 if x['is_removed'] == 1 and x['imp_total'] > 0.1 else 0, axis=1
    )
    df_filtered['time_window'] = df_filtered['timestamp_utc'].dt.floor(f'{GROUP_WINDOW_MINUTES}min')
    df_grouped = df_filtered.groupby('time_window').agg({
        'timestamp_utc': 'first',
        'direction_pred': lambda x: x.mode()[0],
        'direction_prob': 'max',
        'magnitude_pred': 'mean',
        'price_entry': 'first',
        'price_exit': 'last',
        'currency_pair': 'first',
        'event': lambda x: ';'.join(x + ' (' + df_filtered.loc[x.index, 'event_key'] + ')'),
        'is_removed': 'max',
        'priority': lambda x: 'High' if 'High' in x.values else 'Low'  # Приоритет группы
    }).reset_index()
    df_grouped['price_exit'] = df_grouped['price_entry'] + df_grouped['magnitude_pred'] * \
                              (2 * df_grouped['direction_pred'] - 1) * 0.0001
    logging.info(f"После группировки: {len(df_grouped)} сигналов")
else:
    df_grouped = df_filtered
    logging.info("Нет сигналов для группировки")

# --- Сохранение сигналов ---
if not df_grouped.empty:
    df_grouped[['timestamp_utc', 'direction_pred', 'magnitude_pred', 'price_entry', 'price_exit', 'direction_prob', 'currency_pair', 'event', 'is_removed', 'priority']].to_csv(output_csv, index=False)
    logging.info(f"Сохранено: {output_csv}")
    logging.info(df_grouped[['timestamp_utc', 'direction_pred', 'magnitude_pred', 'price_entry', 'price_exit', 'currency_pair', 'event', 'is_removed', 'priority']].head().to_string())
else:
    logging.info(f"Сохранено: {output_csv} (пусто)")
    df_grouped = pd.DataFrame(columns=['timestamp_utc', 'direction_pred', 'magnitude_pred', 'price_entry', 'price_exit', 'direction_prob', 'currency_pair', 'event', 'is_removed', 'priority'])
    df_grouped.to_csv(output_csv, index=False)