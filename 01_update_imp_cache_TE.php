<?php
// 02.01_update_imp_cache_TE.php
// Считает годовой импульс (среднюю волатильность в окне ±15 мин вокруг новости)
// для новостей из news_tradingeconomics и пишет в imp_cache_tradingeconomics.

include("/home/ilyamus/GPTFOREX/config/connectDB.php"); // mysqli $dbcnx

// --------------------- настройки ---------------------
$price_table   = 'HistDataEURUSD';                 // таблица минутных OHLC
$log_file      = "/home/ilyamus/GPTGROKWORK/logs/update_imp_cache_te.log";
$window_minutes = 15;                               // окно ±N минут
$batch_size     = 1000;                             // пакетная вставка

// --------------------- утилиты -----------------------
function log_message($msg) {
    global $log_file;
    if (!is_dir(dirname($log_file))) {
        @mkdir(dirname($log_file), 0755, true);
    }
    $ts = gmdate('Y-m-d H:i:s');
    file_put_contents($log_file, "$ts: $msg\n", FILE_APPEND);
}

// simple slug: "Stock Market" -> "stock_market", "United States" -> "united_states"
function slugify($s) {
    $s = trim((string)$s);
    if ($s === '') return '';
    $s = html_entity_decode($s, ENT_QUOTES | ENT_HTML5, 'UTF-8');
    $s = mb_strtolower($s, 'UTF-8');
    $s = preg_replace('/[^a-z0-9]+/iu', '_', $s);   // всё, что не буква/цифра -> _
    $s = preg_replace('/_+/', '_', $s);
    $s = trim($s, '_');
    return $s;
}

// строим event_key из категорий и страны
function build_event_key($categories, $country) {
    // возьмём первую категорию, если их несколько через запятую
    $cat = $categories ?? '';
    if (strpos($cat, ',') !== false) {
        $cat = explode(',', $cat)[0];
    }
    $cat_slug = slugify($cat);
    if ($cat_slug === '') $cat_slug = 'misc';

    $country_slug = slugify($country ?? '');
    if ($country_slug === '') $country_slug = 'all';

    return "te_{$cat_slug}_{$country_slug}";
}

// --------------------- старт -------------------------
$dbcnx->set_charset('utf8mb4');
$dbcnx->query("SET time_zone = '+00:00'");

// определим диапазон лет по данным news_tradingeconomics
$years_q = $dbcnx->query("SELECT MIN(published_at) AS min_dt, MAX(published_at) AS max_dt FROM news_tradingeconomics");
$rowYears = $years_q->fetch_assoc();
$years_q->close();

$start_year = $rowYears['min_dt'] ? (int)gmdate('Y', strtotime($rowYears['min_dt'])) : 2005;
$end_year   = $rowYears['max_dt'] ? (int)gmdate('Y', strtotime($rowYears['max_dt'])) : (int)gmdate('Y');

log_message("Годы данных: $start_year .. $end_year");

// очистим imp_cache_tradingeconomics полностью (или можно очистить по диапазону)
log_message("TRUNCATE imp_cache_tradingeconomics...");
if (!$dbcnx->query("TRUNCATE TABLE imp_cache_tradingeconomics")) {
    log_message("Ошибка TRUNCATE: " . $dbcnx->error);
    exit(1);
}

// подготовим стейтмент для цен
$price_sql = "
    SELECT high, low
    FROM {$price_table}
    WHERE timestamp_utc BETWEEN ? AND ?
    ORDER BY timestamp_utc ASC
";
$stmt_price = $dbcnx->prepare($price_sql);
if (!$stmt_price) {
    log_message("Ошибка prepare цены: " . $dbcnx->error);
    exit(1);
}

$all_vol_by_key_year = []; // ["ALL|event_key|year"] => [vol1, vol2, ...]
$skipped_total = 0;

for ($year = $start_year; $year <= $end_year; $year++) {
    $start = "$year-01-01 00:00:00";
    $end   = "$year-12-31 23:59:59";

    // берём только записи, у которых есть published_at; country = tags, category = categories
    $news_sql = "
        SELECT article_id, published_at, categories, tags
        FROM news_tradingeconomics
        WHERE published_at BETWEEN ? AND ?
        ORDER BY published_at ASC
    ";
    $stmt_news = $dbcnx->prepare($news_sql);
    if (!$stmt_news) {
        log_message("Ошибка prepare news: " . $dbcnx->error);
        exit(1);
    }
    $stmt_news->bind_param("ss", $start, $end);
    $stmt_news->execute();
    $res_news = $stmt_news->get_result();

    $count_news = $res_news->num_rows;
    log_message("Год $year: найдено новостей = $count_news");

    $skipped_year = 0;
    $handled_year = 0;

    while ($n = $res_news->fetch_assoc()) {
        $ts = $n['published_at'];                 // уже UTC
        if (!$ts) { $skipped_year++; continue; }

        $country   = $n['tags'] ?? '';
        $category  = $n['categories'] ?? '';
        $event_key = build_event_key($category, $country);

        // окно
        $window_start = gmdate('Y-m-d H:i:s', strtotime($ts . " -{$window_minutes} minutes"));
        $window_end   = gmdate('Y-m-d H:i:s', strtotime($ts . " +{$window_minutes} minutes"));

        // цены в окне
        $stmt_price->bind_param("ss", $window_start, $window_end);
        $stmt_price->execute();
        $res_price = $stmt_price->get_result();

        if ($res_price->num_rows === 0) {
            $skipped_year++;
            continue;
        }

        // собираем волатильность: |high - low| * 10000 для каждого бара
        $vals = [];
        while ($p = $res_price->fetch_assoc()) {
            $vol = abs((float)$p['high'] - (float)$p['low']) * 10000.0;
            if ($vol > 0) $vals[] = $vol;
        }
        $res_price->close();

        if (!empty($vals)) {
            $k = "ALL|{$event_key}|{$year}"; // currency=ALL (ты считаешь по EURUSD)
            if (!isset($all_vol_by_key_year[$k])) $all_vol_by_key_year[$k] = [];
            // складываем все значения (как у тебя было)
            $all_vol_by_key_year[$k] = array_merge($all_vol_by_key_year[$k], $vals);
            $handled_year++;
        } else {
            $skipped_year++;
        }
    }

    $stmt_news->close();
    log_message("Год $year: учтено новостей={$handled_year}, пропущено из-за отсутствия цен/данных={$skipped_year}");
    $skipped_total += $skipped_year;
}

// Вставка агрегатов по годам
log_message("Агрегация и вставка в imp_cache_tradingeconomics...");
$values = [];
$inserted = 0;

foreach ($all_vol_by_key_year as $k => $arr) {
    if (empty($arr)) continue;

    list($currency, $event_key, $year) = explode('|', $k, 3);
    $avg_vol = array_sum($arr) / count($arr);

    $currency_esc = $dbcnx->real_escape_string($currency);
    $event_key_esc = $dbcnx->real_escape_string($event_key);
    $year_int = (int)$year;
    $avg_vol_f = (float)$avg_vol;

    $values[] = "('$currency_esc', '$event_key_esc', $year_int, $avg_vol_f, NULL, UTC_TIMESTAMP())";

    if (count($values) >= $batch_size) {
        $sql = "INSERT INTO imp_cache_tradingeconomics (currency, event_key, year, volatility, trend, updated_at) VALUES "
             . implode(',', $values);
        if (!$dbcnx->query($sql)) {
            log_message("Ошибка вставки батча: " . $dbcnx->error);
            exit(1);
        }
        $inserted += count($values);
        $values = [];
    }
}

// остаток
if (!empty($values)) {
    $sql = "INSERT INTO imp_cache_tradingeconomics (currency, event_key, year, volatility, trend, updated_at) VALUES "
         . implode(',', $values);
    if (!$dbcnx->query($sql)) {
        log_message("Ошибка вставки остатка: " . $dbcnx->error);
        exit(1);
    }
    $inserted += count($values);
    $values = [];
}

log_message("Вставлено строк: $inserted");

// расчёт тренда: разница к предыдущему году (по каждому currency+event_key)
log_message("Расчёт trend...");
$upd_total = 0;

// соберём все ключи для которых есть более 1 года
$keys_rs = $dbcnx->query("
    SELECT currency, event_key
    FROM imp_cache_tradingeconomics
    GROUP BY currency, event_key
    HAVING COUNT(*) > 1
");
while ($kr = $keys_rs->fetch_assoc()) {
    $cur  = $dbcnx->real_escape_string($kr['currency']);
    $evk  = $dbcnx->real_escape_string($kr['event_key']);

    // возьмём пары (year, volatility) по ключу
    $yr_rs = $dbcnx->query("
        SELECT year, volatility
        FROM imp_cache_tradingeconomics
        WHERE currency='$cur' AND event_key='$evk'
        ORDER BY year ASC
    ");
    $prev = null;
    $to_update = [];
    while ($yr = $yr_rs->fetch_assoc()) {
        $y = (int)$yr['year'];
        $v = (float)$yr['volatility'];
        if ($prev !== null) {
            $trend = $v - $prev; // разница к прошлому году
            $to_update[$y] = $trend;
        }
        $prev = $v;
    }
    $yr_rs->close();

    foreach ($to_update as $y => $t) {
        $t = (float)$t;
        $u_sql = "
            UPDATE imp_cache_tradingeconomics
            SET trend = $t
            WHERE currency='$cur' AND event_key='$evk' AND year=$y
        ";
        if ($dbcnx->query($u_sql)) {
            $upd_total += $dbcnx->affected_rows;
        } else {
            log_message("Ошибка UPDATE trend: " . $dbcnx->error);
            exit(1);
        }
    }
}
$keys_rs->close();

log_message("Готово. Пропущено новостей из-за отсутствия цен всего: $skipped_total");
log_message("Обновлено строк с trend: $upd_total");
?>