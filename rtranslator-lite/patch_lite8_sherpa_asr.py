#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "RTranslator")
recognizer_path = ROOT / "app/src/main/java/nie/translator/rtranslator/voice_translation/neural_networks/voice/Recognizer.java"
access_path = ROOT / "app/src/main/java/nie/translator/rtranslator/access/AccessActivity.java"
build_gradle = ROOT / "app/build.gradle"
df2_path = ROOT / "app/src/main/java/nie/translator/rtranslator/access/DownloadFragment2.java"
provider_files = [
    ROOT / "app/src/main/AndroidManifest.xml",
    ROOT / "app/src/main/java/nie/translator/rtranslator/access/UserDataFragment.java",
    ROOT / "app/src/main/java/nie/translator/rtranslator/settings/SettingsFragment.java",
]

for p in [recognizer_path, access_path, build_gradle, df2_path, *provider_files]:
    if not p.exists():
        raise SystemExit(f"Required file missing: {p}")

# ---------------------------------------------------------------------------
# 1) Replace RTranslator's hand-written Whisper ORT pipeline with sherpa-onnx.
#    Keep the public Recognizer API unchanged so WalkieTalkie/Conversation do
#    not need invasive changes.
# ---------------------------------------------------------------------------
recognizer_java = r'''/*
 * RTranslator Lite8 sherpa-onnx recognition engine.
 * Based on RTranslator's public Recognizer API and sherpa-onnx Apache-2.0 API.
 */
package nie.translator.rtranslator.voice_translation.neural_networks.voice;

import android.content.Context;
import android.util.Log;

import org.w3c.dom.Document;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.Set;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import com.k2fsa.sherpa.onnx.FeatureConfig;
import com.k2fsa.sherpa.onnx.OfflineModelConfig;
import com.k2fsa.sherpa.onnx.OfflineRecognizer;
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig;
import com.k2fsa.sherpa.onnx.OfflineRecognizerResult;
import com.k2fsa.sherpa.onnx.OfflineStream;
import com.k2fsa.sherpa.onnx.OfflineWhisperModelConfig;
import com.k2fsa.sherpa.onnx.SileroVadModelConfig;
import com.k2fsa.sherpa.onnx.SpeechSegment;
import com.k2fsa.sherpa.onnx.Vad;
import com.k2fsa.sherpa.onnx.VadModelConfig;

import nie.translator.rtranslator.Global;
import nie.translator.rtranslator.R;
import nie.translator.rtranslator.tools.CustomLocale;
import nie.translator.rtranslator.tools.ErrorCodes;
import nie.translator.rtranslator.voice_translation.neural_networks.NeuralNetworkApi;

public class Recognizer extends NeuralNetworkApi {
    private static final String TAG = "RTranslatorLite8";
    public static final String UNDEFINED_TEXT = "[(und)]";

    private static final int SAMPLE_RATE = 16000;
    private static final int FEATURE_DIM = 80;
    private static final int VAD_WINDOW_SIZE = 512;

    private static final String SHERPA_DIR = "sherpa/";
    private static final String ENCODER = SHERPA_DIR + "small-encoder.int8.onnx";
    private static final String DECODER = SHERPA_DIR + "small-decoder.int8.onnx";
    private static final String TOKENS = SHERPA_DIR + "small-tokens.txt";
    private static final String VAD_MODEL = SHERPA_DIR + "silero_vad.int8.onnx";

    private final ArrayList<RecognizerListener> callbacks = new ArrayList<>();
    private final ArrayList<RecognizerMultiListener> multiCallbacks = new ArrayList<>();
    private final ArrayDeque<DataContainer> dataToRecognize = new ArrayDeque<>();
    private final Object queueLock = new Object();
    private final Object engineLock = new Object();

    private volatile boolean recognizing = false;
    private OfflineRecognizer offlineRecognizer;
    private OfflineRecognizerConfig recognizerConfig;
    private OfflineWhisperModelConfig whisperConfig;
    private Vad vad;

    public Recognizer(Global global, final boolean returnResultOnlyAtTheEnd,
                      final NeuralNetworkApi.InitListener initListener) {
        this.global = global;
        new Thread(() -> {
            try {
                initializeSherpa();
                Log.i(TAG, "Sherpa ASR initialized: Whisper Small INT8 + Silero VAD INT8");
                initListener.onInitializationFinished();
            } catch (Throwable t) {
                Log.e(TAG, "Sherpa ASR initialization failed", t);
                initListener.onError(new int[]{ErrorCodes.ERROR_LOADING_MODEL}, 0);
            }
        }, "sherpa-init").start();
    }

    private void initializeSherpa() {
        FeatureConfig featConfig = new FeatureConfig();
        featConfig.setSampleRate(SAMPLE_RATE);
        featConfig.setFeatureDim(FEATURE_DIM);
        featConfig.setDither(0.0f);

        whisperConfig = new OfflineWhisperModelConfig();
        whisperConfig.setEncoder(ENCODER);
        whisperConfig.setDecoder(DECODER);
        // Empty language means Whisper language auto-detection in sherpa-onnx.
        whisperConfig.setLanguage("");
        whisperConfig.setTask("transcribe");
        whisperConfig.setTailPaddings(1000);
        whisperConfig.setEnableTokenTimestamps(false);
        whisperConfig.setEnableSegmentTimestamps(false);

        OfflineModelConfig modelConfig = new OfflineModelConfig();
        modelConfig.setWhisper(whisperConfig);
        modelConfig.setTokens(TOKENS);
        modelConfig.setNumThreads(4);
        modelConfig.setDebug(false);
        modelConfig.setProvider("cpu");

        recognizerConfig = new OfflineRecognizerConfig();
        recognizerConfig.setFeatConfig(featConfig);
        recognizerConfig.setModelConfig(modelConfig);
        recognizerConfig.setDecodingMethod("greedy_search");
        recognizerConfig.setMaxActivePaths(4);

        offlineRecognizer = new OfflineRecognizer(global.getAssets(), recognizerConfig);

        SileroVadModelConfig silero = new SileroVadModelConfig();
        silero.setModel(VAD_MODEL);
        silero.setThreshold(0.5f);
        silero.setMinSilenceDuration(0.25f);
        silero.setMinSpeechDuration(0.10f);
        silero.setWindowSize(VAD_WINDOW_SIZE);
        silero.setMaxSpeechDuration(30.0f);

        VadModelConfig vadConfig = new VadModelConfig();
        vadConfig.setSileroVadModelConfig(silero);
        vadConfig.setSampleRate(SAMPLE_RATE);
        vadConfig.setNumThreads(1);
        vadConfig.setProvider("cpu");
        vadConfig.setDebug(false);
        vad = new Vad(global.getAssets(), vadConfig);
    }

    public void recognize(final float[] data, int beamSize, final String languageCode) {
        enqueue(new DataContainer(data, beamSize, languageCode, null));
    }

    public void recognize(final float[] data, int beamSize,
                          final String languageCode1, final String languageCode2) {
        enqueue(new DataContainer(data, beamSize, languageCode1, languageCode2));
    }

    private void enqueue(DataContainer data) {
        if (data.data == null || data.data.length == 0) {
            return;
        }
        synchronized (queueLock) {
            dataToRecognize.addLast(data);
            if (!recognizing) {
                recognizing = true;
                new Thread(this::drainQueue, "sherpa-recognizer").start();
            }
        }
    }

    private void drainQueue() {
        while (true) {
            DataContainer item;
            synchronized (queueLock) {
                item = dataToRecognize.pollFirst();
                if (item == null) {
                    recognizing = false;
                    return;
                }
            }

            try {
                float[] speech = extractSpeech(item.data);
                if (item.languageCode2 == null) {
                    RecognitionResult result = decode(speech, normalizeLanguage(item.languageCode));
                    String text = correctText(result.text);
                    if (text.isEmpty()) text = UNDEFINED_TEXT;
                    notifyResult(text, item.languageCode, text.equals(UNDEFINED_TEXT) ? 0.0 : 1.0, true);
                } else {
                    // One multilingual Whisper pass instead of RTranslator's two forced-language passes.
                    // Empty language activates sherpa Whisper language detection. We return the same
                    // transcript in both slots: existing RTranslator language detection then selects
                    // the correct side of the selected language pair without a second ASR pass.
                    RecognitionResult result = decode(speech, "");
                    String text = correctText(result.text);
                    if (text.isEmpty()) text = UNDEFINED_TEXT;
                    double confidence = text.equals(UNDEFINED_TEXT) ? 0.0 : 1.0;
                    Log.i(TAG, "Sherpa auto language=" + result.language +
                            " pair=" + item.languageCode + "/" + item.languageCode2);
                    notifyMultiResult(text, item.languageCode, confidence,
                            text, item.languageCode2, confidence);
                }
            } catch (Throwable t) {
                Log.e(TAG, "Sherpa recognition failed", t);
                notifyError(new int[]{ErrorCodes.ERROR_EXECUTING_MODEL}, 0);
            }
        }
    }

    private RecognitionResult decode(float[] samples, String language) {
        synchronized (engineLock) {
            whisperConfig.setLanguage(language == null ? "" : language);
            offlineRecognizer.setConfig(recognizerConfig);

            OfflineStream stream = offlineRecognizer.createStream();
            try {
                stream.acceptWaveform(samples, SAMPLE_RATE);
                offlineRecognizer.decode(stream);
                OfflineRecognizerResult result = offlineRecognizer.getResult(stream);
                return new RecognitionResult(result.getText(), result.getLang());
            } finally {
                stream.release();
            }
        }
    }

    /**
     * RTranslator's Recorder already performs endpointing. Silero is an additional
     * local guard that trims long silence/noise and reduces Whisper hallucinations.
     */
    private float[] extractSpeech(float[] input) {
        if (vad == null || input.length < VAD_WINDOW_SIZE) {
            return input;
        }
        synchronized (engineLock) {
            try {
                vad.reset();
                for (int pos = 0; pos < input.length; pos += VAD_WINDOW_SIZE) {
                    float[] frame = new float[VAD_WINDOW_SIZE];
                    int count = Math.min(VAD_WINDOW_SIZE, input.length - pos);
                    System.arraycopy(input, pos, frame, 0, count);
                    vad.acceptWaveform(frame);
                }
                vad.flush();

                ArrayList<float[]> segments = new ArrayList<>();
                int total = 0;
                while (!vad.empty()) {
                    SpeechSegment segment = vad.front();
                    float[] samples = segment.getSamples();
                    if (samples != null && samples.length > 0) {
                        segments.add(samples);
                        total += samples.length;
                    }
                    vad.pop();
                }

                // If Silero is unsure, preserve RTranslator Recorder's original segment.
                if (total < SAMPLE_RATE / 10) {
                    return input;
                }
                float[] out = new float[total];
                int offset = 0;
                for (float[] segment : segments) {
                    System.arraycopy(segment, 0, out, offset, segment.length);
                    offset += segment.length;
                }
                return out;
            } catch (Throwable t) {
                Log.w(TAG, "Silero VAD fallback to Recorder segment", t);
                return input;
            }
        }
    }

    private static String normalizeLanguage(String code) {
        if (code == null) return "";
        String value = code.trim().toLowerCase();
        int dash = value.indexOf('-');
        if (dash > 0) value = value.substring(0, dash);
        int underscore = value.indexOf('_');
        if (underscore > 0) value = value.substring(0, underscore);
        return value;
    }

    private static String correctText(String text) {
        if (text == null) return "";
        String corrected = text.replaceAll("<\\|[^>]*\\|>", " ").trim();
        corrected = corrected.replace("...", "").trim();
        if (!corrected.isEmpty()) {
            char first = corrected.charAt(0);
            if (Character.isLowerCase(first)) {
                corrected = Character.toUpperCase(first) + corrected.substring(1);
            }
        }
        return corrected;
    }

    public static ArrayList<CustomLocale> getSupportedLanguages(Context context) {
        Set<String> codes = new LinkedHashSet<>();
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        try {
            DocumentBuilder builder = factory.newDocumentBuilder();
            Document document = builder.parse(context.getResources().openRawResource(R.raw.whisper_supported_languages));
            NodeList list = document.getElementsByTagName("code");
            for (int i = 0; i < list.getLength(); i++) {
                codes.add(list.item(i).getTextContent());
            }
        } catch (IOException | SAXException | ParserConfigurationException e) {
            Log.w(TAG, "Cannot read legacy Whisper language list", e);
        }

        // Balkan Lite must always expose the target travel languages even if the
        // upstream quality-filtered XML hides them.
        codes.add("en");
        codes.add("de");
        codes.add("ru");
        codes.add("uk");
        codes.add("sr");
        codes.add("mk");

        ArrayList<CustomLocale> languages = new ArrayList<>();
        for (String code : codes) {
            languages.add(CustomLocale.getInstance(code));
        }
        return languages;
    }

    public int getLanguageID(String language) {
        // Kept for source compatibility. Sherpa uses language strings directly.
        return -1;
    }

    public void addCallback(final RecognizerListener callback) {
        callbacks.add(callback);
    }

    public void removeCallback(RecognizerListener callback) {
        callbacks.remove(callback);
    }

    public void addMultiCallback(final RecognizerMultiListener callback) {
        multiCallbacks.add(callback);
    }

    public void removeMultiCallback(RecognizerMultiListener callback) {
        multiCallbacks.remove(callback);
    }

    private void notifyResult(String text, String languageCode, double confidenceScore, boolean isFinal) {
        for (RecognizerListener callback : new ArrayList<>(callbacks)) {
            callback.onSpeechRecognizedResult(text, languageCode, confidenceScore, isFinal);
        }
    }

    private void notifyMultiResult(String text1, String languageCode1, double confidenceScore1,
                                   String text2, String languageCode2, double confidenceScore2) {
        for (RecognizerMultiListener callback : new ArrayList<>(multiCallbacks)) {
            callback.onSpeechRecognizedResult(text1, languageCode1, confidenceScore1,
                    text2, languageCode2, confidenceScore2);
        }
    }

    private void notifyError(int[] reasons, long value) {
        for (RecognizerListener callback : new ArrayList<>(callbacks)) {
            callback.onError(reasons, value);
        }
        for (RecognizerMultiListener callback : new ArrayList<>(multiCallbacks)) {
            callback.onError(reasons, value);
        }
    }

    public void destroy() {
        synchronized (engineLock) {
            if (vad != null) {
                vad.release();
                vad = null;
            }
            if (offlineRecognizer != null) {
                offlineRecognizer.release();
                offlineRecognizer = null;
            }
        }
    }

    private static final class RecognitionResult {
        final String text;
        final String language;
        RecognitionResult(String text, String language) {
            this.text = text == null ? "" : text;
            this.language = language == null ? "" : language;
        }
    }

    private static final class DataContainer {
        final float[] data;
        final int beamSize;
        final String languageCode;
        final String languageCode2;

        DataContainer(float[] data, int beamSize, String languageCode, String languageCode2) {
            this.data = data;
            this.beamSize = beamSize;
            this.languageCode = languageCode;
            this.languageCode2 = languageCode2;
        }
    }
}
'''
recognizer_path.write_text(recognizer_java, encoding="utf-8")

# ---------------------------------------------------------------------------
# 2) Models are bundled into assets by CI, so the legacy Whisper downloader is
#    not needed. Preserve onboarding/permissions but jump directly to Loading.
# ---------------------------------------------------------------------------
access = access_path.read_text(encoding="utf-8")
if "import nie.translator.rtranslator.LoadingActivity;" not in access:
    access = access.replace(
        "import nie.translator.rtranslator.Global;",
        "import nie.translator.rtranslator.Global;\nimport nie.translator.rtranslator.LoadingActivity;",
        1,
    )
old_download_case = r'''case DOWNLOAD_FRAGMENT: {
                DownloadFragment2 downloadFragment = new DownloadFragment2();
                if (bundle != null) {
                    downloadFragment.setArguments(bundle);
                }
                FragmentTransaction transaction = getSupportFragmentManager().beginTransaction();
                transaction.setTransition(FragmentTransaction.TRANSIT_FRAGMENT_FADE);
                transaction.replace(R.id.fragment_initialization_container, downloadFragment);
                transaction.commit();
                fragment = downloadFragment;
                break;
            }'''
new_download_case = r'''case DOWNLOAD_FRAGMENT: {
                // Lite8: sherpa Whisper INT8 + Silero VAD are bundled in APK assets.
                // Skip the obsolete NLLB/legacy-Whisper downloader entirely.
                Global global = (Global) getApplication();
                global.setFirstStart(false);
                Intent intent = new Intent(this, LoadingActivity.class);
                intent.putExtra("activity", "download");
                startActivity(intent);
                finish();
                break;
            }'''
if old_download_case not in access:
    raise SystemExit("Could not find AccessActivity DOWNLOAD_FRAGMENT block")
access = access.replace(old_download_case, new_download_case, 1)
access_path.write_text(access, encoding="utf-8")

# Mark the inactive old downloader so APK/static inspection cannot confuse it with active onboarding.
df2 = df2_path.read_text(encoding="utf-8")
df2 = df2.replace("LITE7 | ACTIVE Downloader2 | Whisper only", "LITE8 | LEGACY Downloader2 DISABLED", 1)
df2_path.write_text(df2, encoding="utf-8")

# ---------------------------------------------------------------------------
# 3) Align ORT with sherpa 1.13.5 and add the stripped arm64 AAR.
# ---------------------------------------------------------------------------
bg = build_gradle.read_text(encoding="utf-8")
required = [
    'applicationId "nie.translator.rtranslator.lite7"',
    'versionCode 30009',
    "versionName '3.0.0-alpha3-lite7'",
    "implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.23.2'",
]
for value in required:
    if value not in bg:
        raise SystemExit(f"Expected Lite7/ORT Gradle value not found: {value}")

bg = bg.replace('applicationId "nie.translator.rtranslator.lite7"',
                'applicationId "nie.translator.rtranslator.lite8"', 1)
bg = bg.replace('versionCode 30009', 'versionCode 30010', 1)
bg = bg.replace("versionName '3.0.0-alpha3-lite7'",
                "versionName '3.0.0-alpha3-lite8'", 1)
bg = bg.replace("implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.23.2'",
                "implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.27.1'", 1)

anchor = "implementation files('libs/sqlite4java-android-release.aar')"
if anchor not in bg:
    raise SystemExit("Could not find dependency insertion anchor")
extra_deps = """implementation files('libs/sherpa-onnx-1.13.5-no-ort-arm64.aar')
    implementation 'org.jetbrains.kotlin:kotlin-stdlib:1.7.20'
    """
bg = bg.replace(anchor, extra_deps + anchor, 1)

# Keep large ONNX assets uncompressed to avoid an additional decompression/RAM copy.
if "noCompress 'onnx', 'txt'" not in bg:
    marker = "    buildFeatures {\n        prefab true\n    }"
    if marker not in bg:
        raise SystemExit("Could not find buildFeatures block for aaptOptions insertion")
    bg = bg.replace(marker, marker + "\n\n    aaptOptions {\n        noCompress 'onnx', 'txt'\n    }", 1)

build_gradle.write_text(bg, encoding="utf-8")

# ---------------------------------------------------------------------------
# 4) Lite8 visible/package identity.
# ---------------------------------------------------------------------------
changed_labels = 0
for strings_xml in (ROOT / "app/src/main/res").glob("values*/strings.xml"):
    text = strings_xml.read_text(encoding="utf-8")
    if "RTranslator Lite7" in text:
        strings_xml.write_text(text.replace("RTranslator Lite7", "RTranslator Lite8"), encoding="utf-8")
        changed_labels += 1
if changed_labels == 0:
    raise SystemExit("No RTranslator Lite7 app_name resource found for Lite8")

for path in provider_files:
    text = path.read_text(encoding="utf-8")
    if "com.gallery.RTranslator.lite7.provider" not in text:
        raise SystemExit(f"Expected Lite7 FileProvider authority not found in {path}")
    path.write_text(text.replace(
        "com.gallery.RTranslator.lite7.provider",
        "com.gallery.RTranslator.lite8.provider",
    ), encoding="utf-8")

# Update Lite7 runtime log tag in the WalkieTalkie safety patch result.
activity_path = ROOT / "app/src/main/java/nie/translator/rtranslator/voice_translation/VoiceTranslationActivity.java"
activity = activity_path.read_text(encoding="utf-8")
activity = activity.replace('"RTranslatorLite7"', '"RTranslatorLite8"')
activity_path.write_text(activity, encoding="utf-8")

print("Lite8 sherpa-onnx ASR patch complete:")
print(" - Recognizer API replaced by sherpa OfflineRecognizer")
print(" - Whisper Small INT8 + Silero VAD INT8 assets expected under app/src/main/assets/sherpa")
print(" - two-language mode uses one auto-language Whisper pass")
print(" - legacy model downloader bypassed")
print(" - ORT aligned to 1.27.1")
print(" - sherpa stripped arm64 AAR dependency added")
print(" - applicationId nie.translator.rtranslator.lite8")
print(" - version 3.0.0-alpha3-lite8")
