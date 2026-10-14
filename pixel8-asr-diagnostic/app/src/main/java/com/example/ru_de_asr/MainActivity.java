package com.example.ru_de_asr;

import android.Manifest;
import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Typeface;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.speech.ModelDownloadListener;
import android.speech.RecognitionListener;
import android.speech.RecognitionSupport;
import android.speech.RecognitionSupportCallback;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.Locale;
import java.util.List;

public final class MainActivity extends Activity implements RecognitionListener {
    private static final int MIC_REQUEST = 92;
    private static final int SAVE_REQUEST = 93;
    private static final String[] LANGUAGES = {"ru-RU", "de-DE"};
    private final Handler handler = new Handler(Looper.getMainLooper());
    private final StringBuilder journal = new StringBuilder();

    private SpeechRecognizer recognizer;
    private Boolean localRecognizer;
    private Spinner engineSpinner, modeSpinner, switchSpinner;
    private CheckBox restartCheck;
    private TextView statusView, resultView, logView, languageView, switchView, modelsView;
    private ScrollView logScroll;
    private boolean running, listening;
    private int sessionNumber, errorCount;
    private long sessionStart, speechStart;
    private String currentLanguage = "не сообщён";
    private String lastSwitch = "нет события";
    private String ruModelState = "не проверено", deModelState = "не проверено";
    private int detectionEvents, switchSuccess, switchFailed, switchNoModel, switchNotAttempted;
    private int lastConfidence = -1, lastSwitchCode = -99;
    private String lastReportedLanguage = "";
    private Runnable pendingRestart;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        makeInterface();
        log("Устройство: " + Build.MANUFACTURER + " " + Build.MODEL);
        log("Android " + Build.VERSION.RELEASE + "; API " + Build.VERSION.SDK_INT);
        log("SpeechRecognizer: " + SpeechRecognizer.isRecognitionAvailable(this));
        log("On-device: " + SpeechRecognizer.isOnDeviceRecognitionAvailable(this));
        log("Приложение не переводит, не сохраняет аудио и не отправляет его разработчику.");
    }

    private int dip(int value) {
        return (int)(value * getResources().getDisplayMetrics().density + 0.5f);
    }

    private TextView label(String text, int size) {
        TextView view = new TextView(this);
        view.setText(text);
        view.setTextSize(size);
        view.setTextColor(Color.rgb(25, 30, 40));
        view.setPadding(0, dip(5), 0, dip(5));
        return view;
    }

    private void addButton(LinearLayout parent, String title, View.OnClickListener action) {
        Button b = new Button(this);
        b.setText(title);
        b.setAllCaps(false);
        b.setOnClickListener(action);
        parent.addView(b, new LinearLayout.LayoutParams(-1, -2));
    }

    private Spinner addSpinner(LinearLayout parent, String[] values) {
        Spinner s = new Spinner(this);
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, values);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        s.setAdapter(adapter);
        parent.addView(s, new LinearLayout.LayoutParams(-1, dip(54)));
        return s;
    }

    private void makeInterface() {
        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dip(16), dip(12), dip(16), dip(20));
        scroll.addView(root);
        setContentView(scroll);

        TextView title = label("RU ↔ DE · проверка распознавания", 21);
        title.setTypeface(null, Typeface.BOLD);
        root.addView(title);
        root.addView(label("Pixel 8 / Android 17, без Whisper", 13));
        root.addView(label("Движок", 15));
        engineSpinner = addSpinner(root, new String[]{
                "Локальный (on-device)",
                "Обычный системный (может использовать сеть)"
        });
        root.addView(label("Режим", 15));
        modeSpinner = addSpinner(root, new String[]{
                "Авто RU ↔ DE, основной русский",
                "Авто RU ↔ DE, основной немецкий",
                "Только русский",
                "Только немецкий"
        });
        root.addView(label("Автоматическое переключение: чувствительность", 15));
        switchSpinner = addSpinner(root, new String[]{
                "QUICK_RESPONSE (раннее переключение)",
                "BALANCED (требуется уверенность)"
        });
        root.addView(label("В режимах «Только русский/немецкий» переключение выключено.", 12));
        restartCheck = new CheckBox(this);
        restartCheck.setText("Автоматически продолжать после реплики");
        restartCheck.setChecked(true);
        root.addView(restartCheck);
        addButton(root, "1. Проверить установку языков RU и DE", v -> checkSupport());
        addButton(root, "2. Начать распознавание", v -> startTest());
        addButton(root, "Остановить", v -> stopTest());
        addButton(root, "Запросить офлайн-модель RU", v -> requestModel("ru-RU"));
        addButton(root, "Запросить офлайн-модель DE", v -> requestModel("de-DE"));
        addButton(root, "Скопировать журнал", v -> copyJournal());
        addButton(root, "Сохранить журнал TXT", v -> saveJournal());

        modelsView = label("Локальные модели: RU — не проверено; DE — не проверено", 13);
        root.addView(modelsView);
        languageView = label("Обнаруженный язык: нет события", 14);
        root.addView(languageView);
        switchView = label("Переключение ASR: нет события", 14);
        root.addView(switchView);
        statusView = label("Ожидание", 15);
        statusView.setTypeface(null, Typeface.BOLD);
        root.addView(statusView);
        resultView = label("Распознанный текст появится здесь", 17);
        root.addView(resultView);
        root.addView(label("События и задержки", 13));

        logScroll = new ScrollView(this);
        logScroll.setBackgroundColor(Color.rgb(242, 244, 247));
        logView = label("", 12);
        logView.setTypeface(Typeface.MONOSPACE);
        logView.setTextIsSelectable(true);
        logView.setPadding(dip(8), dip(6), dip(8), dip(8));
        logScroll.addView(logView);
        root.addView(logScroll, new LinearLayout.LayoutParams(-1, dip(320)));
    }

    private void log(String text) {
        String clock = new SimpleDateFormat("HH:mm:ss.SSS", Locale.ROOT).format(new Date());
        String delay = sessionStart > 0
                ? " +" + (SystemClock.elapsedRealtime() - sessionStart) + "ms" : "";
        journal.append(clock).append(delay).append(" [").append(sessionNumber)
                .append("] ").append(text).append('\n');
        int from = Math.max(0, journal.length() - 30000);
        logView.setText(journal.substring(from));
        logScroll.post(() -> logScroll.fullScroll(View.FOCUS_DOWN));
    }

    private void status(String s) {
        statusView.setText("Состояние: " + s + " | язык: " + currentLanguage);
    }

    private boolean micAllowed() {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO)
                == PackageManager.PERMISSION_GRANTED) return true;
        requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, MIC_REQUEST);
        return false;
    }

    @Override public void onRequestPermissionsResult(int code, String[] permissions, int[] grants) {
        super.onRequestPermissionsResult(code, permissions, grants);
        if (code == MIC_REQUEST) {
            log(grants.length > 0 && grants[0] == PackageManager.PERMISSION_GRANTED
                    ? "Доступ к микрофону дан, нажмите запуск ещё раз"
                    : "Доступ к микрофону отклонён");
        }
    }

    private boolean prepareRecognizer() {
        boolean wantLocal = engineSpinner.getSelectedItemPosition() == 0;
        if (recognizer != null && localRecognizer != null
                && localRecognizer.booleanValue() == wantLocal) return true;

        destroyRecognizer();
        if (wantLocal && !SpeechRecognizer.isOnDeviceRecognitionAvailable(this)) {
            log("Нет локального RecognitionService");
            return false;
        }
        if (!wantLocal && !SpeechRecognizer.isRecognitionAvailable(this)) {
            log("Нет системного RecognitionService");
            return false;
        }
        try {
            recognizer = wantLocal
                    ? SpeechRecognizer.createOnDeviceSpeechRecognizer(this)
                    : SpeechRecognizer.createSpeechRecognizer(this);
            recognizer.setRecognitionListener(this);
            localRecognizer = wantLocal;
            log("Создан " + (wantLocal ? "ON_DEVICE" : "DEFAULT") + " SpeechRecognizer");
            return true;
        } catch (Exception e) {
            log("Не удалось создать распознаватель: " + e);
            return false;
        }
    }

    private void destroyRecognizer() {
        if (recognizer != null) {
            recognizer.cancel();
            recognizer.destroy();
            recognizer = null;
        }
        localRecognizer = null;
    }

    private Intent recognitionIntent(String lang, boolean auto) {
        Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, lang);
        intent.putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true);
        intent.putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3);
        if (auto) {
            intent.putExtra(RecognizerIntent.EXTRA_ENABLE_LANGUAGE_DETECTION, true);
            intent.putExtra(RecognizerIntent.EXTRA_ENABLE_LANGUAGE_SWITCH,
                    switchSpinner.getSelectedItemPosition() == 0
                            ? RecognizerIntent.LANGUAGE_SWITCH_QUICK_RESPONSE
                            : RecognizerIntent.LANGUAGE_SWITCH_BALANCED);
            intent.putStringArrayListExtra(
                    RecognizerIntent.EXTRA_LANGUAGE_DETECTION_ALLOWED_LANGUAGES,
                    new ArrayList<>(Arrays.asList(LANGUAGES)));
            intent.putStringArrayListExtra(
                    RecognizerIntent.EXTRA_LANGUAGE_SWITCH_ALLOWED_LANGUAGES,
                    new ArrayList<>(Arrays.asList(LANGUAGES)));
        }
        return intent;
    }

    private Intent selectedIntent() {
        int index = modeSpinner.getSelectedItemPosition();
        String language = index == 1 || index == 3 ? "de-DE" : "ru-RU";
        return recognitionIntent(language, index < 2);
    }

    private boolean containsLanguage(List<String> list, String tag) {
        for (String item : list) {
            if (tag.equalsIgnoreCase(item)) return true;
        }
        return false;
    }

    private void displayModelStates() {
        modelsView.setText("Модели локально: RU — " + ruModelState + "; DE — " + deModelState);
    }

    private void checkSupport() {
        if (running) {
            log("Для проверки сначала остановите распознавание");
            return;
        }
        if (!micAllowed() || !prepareRecognizer()) return;
        ruModelState = "проверяется";
        deModelState = "проверяется";
        displayModelStates();
        for (String target : LANGUAGES) {
            try {
                recognizer.checkRecognitionSupport(recognitionIntent(target, false),
                        getMainExecutor(), new RecognitionSupportCallback() {
                    @Override public void onSupportResult(RecognitionSupport support) {
                        boolean installed = containsLanguage(
                                support.getInstalledOnDeviceLanguages(), target);
                        boolean available = containsLanguage(
                                support.getSupportedOnDeviceLanguages(), target);
                        boolean pending = containsLanguage(
                                support.getPendingOnDeviceLanguages(), target);
                        String state = installed ? "УСТАНОВЛЕНА" :
                                pending ? "ожидает загрузки" :
                                available ? "ДОСТУПНА ДЛЯ ЗАГРУЗКИ" : "не подтверждена";
                        if ("ru-RU".equals(target)) ruModelState = state;
                        else deModelState = state;
                        displayModelStates();
                        log("MODEL " + target + ": " + state
                                + " | installed=" + support.getInstalledOnDeviceLanguages()
                                + " | downloadable=" + support.getSupportedOnDeviceLanguages()
                                + " | pending=" + support.getPendingOnDeviceLanguages());
                    }
                    @Override public void onError(int error) {
                        if ("ru-RU".equals(target)) ruModelState = "ошибка " + error;
                        else deModelState = "ошибка " + error;
                        displayModelStates();
                        log("MODEL " + target + ": CHECK_ERROR " + error + " " + errorLabel(error));
                    }
                });
            } catch (Exception e) {
                log("MODEL " + target + ": CHECK_EXCEPTION " + e);
            }
        }

        try {
            recognizer.checkRecognitionSupport(selectedIntent(), getMainExecutor(),
                    new RecognitionSupportCallback() {
                @Override public void onSupportResult(RecognitionSupport result) {
                    log("AUTO SUPPORT callback OK. Реальное переключение этим НЕ подтверждается.");
                }
                @Override public void onError(int code) {
                    log("AUTO SUPPORT error " + code + " " + errorLabel(code));
                }
            });
        } catch (Exception e) {
            log("AUTO SUPPORT exception " + e);
        }
    }

    private void requestModel(String lang) {
        if (running) { log("Сначала остановите распознавание"); return; }
        if (!micAllowed() || !prepareRecognizer()) return;
        log("MODEL_DOWNLOAD request=" + lang);
        try {
            recognizer.triggerModelDownload(recognitionIntent(lang, false),
                    getMainExecutor(), new ModelDownloadListener() {
                @Override public void onProgress(int percent) {
                    log("MODEL_DOWNLOAD " + lang + ": " + percent + "%");
                }
                @Override public void onSuccess() {
                    log("MODEL_DOWNLOAD " + lang + ": SUCCESS; обновляю список установленных");
                    checkSupport();
                }
                @Override public void onScheduled() {
                    log("MODEL_DOWNLOAD " + lang + ": SCHEDULED; позже повторите проверку");
                }
                @Override public void onError(int error) {
                    log("MODEL_DOWNLOAD " + lang + ": ERROR " + error + " " + errorLabel(error));
                    log("Проверьте доступность модели кнопкой проверки.");
                }
            });
        } catch (Exception e) {
            log("MODEL_DOWNLOAD exception " + lang + ": " + e);
        }
    }

    private void startTest() {
        if (!micAllowed()) return;
        if (running) return;
        if (!prepareRecognizer()) return;
        running = true;
        errorCount = 0;
        detectionEvents = switchSuccess = switchFailed = switchNoModel = switchNotAttempted = 0;
        log("RUN CONFIG engine=" + engineSpinner.getSelectedItem()
                + " mode=" + modeSpinner.getSelectedItem()
                + " switchSensitivity=" + switchSpinner.getSelectedItem()
                + " models=" + ruModelState + "/" + deModelState);
        startSession();
    }

    private void startSession() {
        if (!running || listening || recognizer == null) return;
        sessionNumber++;
        sessionStart = SystemClock.elapsedRealtime();
        speechStart = 0;
        currentLanguage = "не определён";
        lastSwitch = "нет события";
        lastReportedLanguage = "";
        lastSwitchCode = -99;
        lastConfidence = -1;
        languageView.setText("Обнаруженный язык: нет события в сессии");
        switchView.setText("Переключение ASR: нет события в сессии");
        listening = true;
        status("слушаю");
        try {
            recognizer.startListening(selectedIntent());
            log("START: " + modeSpinner.getSelectedItem()
                    + "; " + engineSpinner.getSelectedItem()
                    + "; switch=" + switchSpinner.getSelectedItem());
        } catch (Exception e) {
            listening = false;
            running = false;
            status("сбой запуска");
            log("START exception: " + e);
        }
    }

    private void finishSession(boolean error) {
        listening = false;
        if (!running) return;
        errorCount = error ? errorCount + 1 : 0;
        if (errorCount >= 5) {
            logRunSummary();
            running = false;
            status("остановлено после пяти ошибок");
            return;
        }
        if (!restartCheck.isChecked()) {
            logRunSummary();
            running = false;
            status("готово");
            return;
        }
        long pause = error ? Math.min(2500L, 500L * errorCount) : 300L;
        pendingRestart = this::startSession;
        handler.postDelayed(pendingRestart, pause);
        log("RESTART через " + pause + " ms");
    }

    private void stopTest() {
        running = false;
        listening = false;
        if (pendingRestart != null) handler.removeCallbacks(pendingRestart);
        pendingRestart = null;
        logRunSummary();
        destroyRecognizer();
        status("остановлено");
        log("STOP");
    }

    @Override public void onReadyForSpeech(Bundle b) {
        log("onReadyForSpeech");
        status("готов слушать");
    }
    @Override public void onBeginningOfSpeech() {
        speechStart = SystemClock.elapsedRealtime();
        log("onBeginningOfSpeech");
        status("речь");
    }
    @Override public void onRmsChanged(float level) {}
    @Override public void onBufferReceived(byte[] bytes) {}
    @Override public void onEndOfSpeech() {
        log("onEndOfSpeech");
        status("обработка");
    }
    @Override public void onError(int code) {
        log("ERROR " + code + " " + errorLabel(code));
        finishSession(true);
    }
    @Override public void onResults(Bundle b) {
        long now = SystemClock.elapsedRealtime();
        String text = bestText(b);
        log("FINAL detectedLanguage=" + currentLanguage + " switch=" + lastSwitch + " text=" + text
                + " sinceSpeechStart=" + (speechStart == 0 ? "n/a" : (now - speechStart) + "ms")
                + " scores=" + scores(b));
        resultView.setText("Обнаружено: " + currentLanguage + "\nПереключение: " + lastSwitch + "\nФинал: " + text);
        status("результат");
        finishSession(false);
    }
    @Override public void onPartialResults(Bundle b) {
        String text = bestText(b);
        resultView.setText("Обнаружено: " + currentLanguage + "\nПереключение: " + lastSwitch + "\nЧастично: " + text);
        log("PARTIAL detected=" + currentLanguage + " switch=" + lastSwitch + " text=" + text);
    }
    private String switchLabel(int result) {
        switch (result) {
            case SpeechRecognizer.LANGUAGE_SWITCH_RESULT_NOT_ATTEMPTED:
                return "NOT_ATTEMPTED (0)";
            case SpeechRecognizer.LANGUAGE_SWITCH_RESULT_SUCCEEDED:
                return "SUCCEEDED (1)";
            case SpeechRecognizer.LANGUAGE_SWITCH_RESULT_FAILED:
                return "FAILED (2)";
            case SpeechRecognizer.LANGUAGE_SWITCH_RESULT_SKIPPED_NO_MODEL:
                return "SKIPPED_NO_MODEL (3)";
            default:
                return "не сообщён (" + result + ")";
        }
    }

    private void logRunSummary() {
        log("RUN SUMMARY detection=" + detectionEvents
                + " switchSucceeded=" + switchSuccess
                + " switchFailed=" + switchFailed
                + " switchSkippedNoModel=" + switchNoModel
                + " switchNotAttempted=" + switchNotAttempted);
    }

    @Override public void onLanguageDetection(Bundle b) {
        if (b == null) return;
        String detected = b.getString(SpeechRecognizer.DETECTED_LANGUAGE, "не указан");
        int confidence = b.getInt(SpeechRecognizer.LANGUAGE_DETECTION_CONFIDENCE_LEVEL, -1);
        int result = b.containsKey(SpeechRecognizer.LANGUAGE_SWITCH_RESULT)
                ? b.getInt(SpeechRecognizer.LANGUAGE_SWITCH_RESULT, -1) : -1;
        currentLanguage = detected;
        lastSwitch = switchLabel(result);
        detectionEvents++;

        switch (result) {
            case SpeechRecognizer.LANGUAGE_SWITCH_RESULT_NOT_ATTEMPTED:
                switchNotAttempted++;
                break;
            case SpeechRecognizer.LANGUAGE_SWITCH_RESULT_SUCCEEDED:
                switchSuccess++;
                break;
            case SpeechRecognizer.LANGUAGE_SWITCH_RESULT_FAILED:
                switchFailed++;
                break;
            case SpeechRecognizer.LANGUAGE_SWITCH_RESULT_SKIPPED_NO_MODEL:
                switchNoModel++;
                break;
        }
        languageView.setText("Обнаруженный язык: " + detected
                + " (confidence=" + confidence + ")");
        switchView.setText("Переключение ASR: " + lastSwitch
                + " | успешно=" + switchSuccess
                + ", ошибки=" + switchFailed
                + ", нет модели=" + switchNoModel);

        // Не засоряем лог сотнями идентичных событий каждые несколько миллисекунд.
        if (!detected.equals(lastReportedLanguage)
                || confidence != lastConfidence
                || result != lastSwitchCode
                || result == SpeechRecognizer.LANGUAGE_SWITCH_RESULT_SUCCEEDED
                || result == SpeechRecognizer.LANGUAGE_SWITCH_RESULT_FAILED
                || result == SpeechRecognizer.LANGUAGE_SWITCH_RESULT_SKIPPED_NO_MODEL) {
            log("LANGUAGE detected=" + detected + " confidence=" + confidence
                    + " switch=" + switchLabel(result));
        }
        lastReportedLanguage = detected;
        lastConfidence = confidence;
        lastSwitchCode = result;
        status("язык обнаружен (не обязательно переключён)");
    }
    @Override public void onEvent(int event, Bundle b) {
        log("EVENT " + event);
    }

    private String bestText(Bundle b) {
        if (b == null) return "<пусто>";
        ArrayList<String> list = b.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
        return list == null || list.isEmpty() ? "<нет текста>" : list.get(0);
    }

    private String scores(Bundle b) {
        float[] values = b == null ? null : b.getFloatArray(SpeechRecognizer.CONFIDENCE_SCORES);
        return values == null ? "n/a" : Arrays.toString(values);
    }

    private String errorLabel(int e) {
        switch (e) {
            case SpeechRecognizer.ERROR_NETWORK_TIMEOUT: return "NETWORK_TIMEOUT";
            case SpeechRecognizer.ERROR_NETWORK: return "NETWORK";
            case SpeechRecognizer.ERROR_AUDIO: return "AUDIO";
            case SpeechRecognizer.ERROR_SERVER: return "SERVER";
            case SpeechRecognizer.ERROR_CLIENT: return "CLIENT";
            case SpeechRecognizer.ERROR_SPEECH_TIMEOUT: return "SPEECH_TIMEOUT";
            case SpeechRecognizer.ERROR_NO_MATCH: return "NO_MATCH";
            case SpeechRecognizer.ERROR_RECOGNIZER_BUSY: return "BUSY";
            case SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS: return "PERMISSION";
            case SpeechRecognizer.ERROR_TOO_MANY_REQUESTS: return "TOO_MANY_REQUESTS";
            case SpeechRecognizer.ERROR_SERVER_DISCONNECTED: return "DISCONNECTED";
            case SpeechRecognizer.ERROR_LANGUAGE_NOT_SUPPORTED: return "LANGUAGE_NOT_SUPPORTED";
            case SpeechRecognizer.ERROR_LANGUAGE_UNAVAILABLE: return "LANGUAGE_UNAVAILABLE";
            case SpeechRecognizer.ERROR_CANNOT_CHECK_SUPPORT: return "CANNOT_CHECK_SUPPORT";
            default: return "UNKNOWN";
        }
    }

    private void copyJournal() {
        ClipboardManager clipboard = (ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
        clipboard.setPrimaryClip(ClipData.newPlainText("RU-DE ASR diagnostic", journal.toString()));
        Toast.makeText(this, "Журнал скопирован", Toast.LENGTH_SHORT).show();
    }

    private void saveJournal() {
        Intent i = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        i.setType("text/plain");
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.putExtra(Intent.EXTRA_TITLE, "pixel8_ru_de_asr_log.txt");
        startActivityForResult(i, SAVE_REQUEST);
    }

    @Override protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (request != SAVE_REQUEST || result != RESULT_OK || data == null
                || data.getData() == null) return;
        try (OutputStream stream = getContentResolver().openOutputStream(data.getData())) {
            if (stream == null) throw new IllegalStateException("No output stream");
            stream.write(journal.toString().getBytes(StandardCharsets.UTF_8));
            Toast.makeText(this, "Журнал сохранён", Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            log("SAVE failed: " + e);
        }
    }

    @Override protected void onDestroy() {
        if (pendingRestart != null) handler.removeCallbacks(pendingRestart);
        destroyRecognizer();
        super.onDestroy();
    }
}
