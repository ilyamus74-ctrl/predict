#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "RTranslator")
translator = ROOT / "app/src/main/java/nie/translator/rtranslator/voice_translation/neural_networks/translation/Translator.java"
engine_path = ROOT / "app/src/main/java/nie/translator/rtranslator/voice_translation/neural_networks/translation/LlamaCppTranslationEngine.java"
manager = ROOT / "app/src/main/java/nie/translator/rtranslator/settings/ModelManagerFragment.java"
layout = ROOT / "app/src/main/res/layout/fragment_model_manager.xml"
build = ROOT / "app/build.gradle"
provider_files = [
    ROOT / "app/src/main/AndroidManifest.xml",
    ROOT / "app/src/main/java/nie/translator/rtranslator/access/UserDataFragment.java",
    ROOT / "app/src/main/java/nie/translator/rtranslator/settings/SettingsFragment.java",
]
for p in [translator, manager, layout, build, *provider_files]:
    if not p.exists():
        raise SystemExit(f"Missing required file: {p}")

engine_path.write_text(r'''package nie.translator.rtranslator.voice_translation.neural_networks.translation;

import android.os.Environment;
import android.util.Log;

import java.io.File;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Locale;

import com.arm.aichat.TranslationBridge;

import nie.translator.rtranslator.Global;
import nie.translator.rtranslator.tools.CustomLocale;

/** Lite10 unified GGUF translation backend using llama.cpp Android. */
public final class LlamaCppTranslationEngine {
    private static final String TAG = "RTranslatorLite10";
    private final Global global;
    private final int mode;
    private final Object lock = new Object();
    private TranslationBridge bridge;
    private File modelFile;

    public LlamaCppTranslationEngine(Global global, int mode) {
        this.global = global;
        this.mode = mode;
    }

    public void load() throws Exception {
        synchronized (lock) {
            modelFile = findModelFile(mode);
            if (modelFile == null) {
                throw new IllegalStateException("GGUF model not found in " + getModelDirectory(mode).getAbsolutePath());
            }
            Log.i(TAG, "Loading GGUF model: " + modelFile.getAbsolutePath());
            bridge = TranslationBridge.create(global);
            bridge.loadModel(modelFile.getAbsolutePath());
            Log.i(TAG, "GGUF model ready: " + modelFile.getName());
        }
    }

    public String translate(String text, CustomLocale source, CustomLocale target) throws Exception {
        synchronized (lock) {
            if (bridge == null) throw new IllegalStateException("llama.cpp engine not loaded");
            String prompt = buildPrompt(text, source, target);
            int maxTokens = Math.max(96, Math.min(1024, text.length() * 2 + 64));
            String out = bridge.generateRaw(prompt, maxTokens);
            return cleanup(out, target);
        }
    }

    public void close() {
        synchronized (lock) {
            if (bridge != null) {
                try {
                    bridge.unload();
                } catch (Throwable t) {
                    Log.w(TAG, "Failed to unload GGUF", t);
                }
                bridge = null;
            }
        }
    }

    public static boolean isLlamaMode(int mode) {
        return mode == Translator.MILMMT || mode == Translator.TINY_AYA || mode == Translator.TRANSLATE_GEMMA;
    }

    public static File getModelDirectory(int mode) {
        String base = Environment.getExternalStorageDirectory().getPath() + "/models/Translation/";
        if (mode == Translator.MILMMT) return new File(base + "MiLMMT");
        if (mode == Translator.TINY_AYA) return new File(base + "TinyAya");
        if (mode == Translator.TRANSLATE_GEMMA) return new File(base + "TranslateGemma");
        return new File(base);
    }

    public static File findModelFile(int mode) {
        File dir = getModelDirectory(mode);
        File[] files = dir.listFiles((d, name) -> name.toLowerCase(Locale.ROOT).endsWith(".gguf"));
        if (files == null || files.length == 0) return null;
        Arrays.sort(files, Comparator.comparing(File::getName, String.CASE_INSENSITIVE_ORDER));
        return files[0];
    }

    public static String[] getSupportedLanguageCodes(int mode) {
        if (mode == Translator.MILMMT) {
            return new String[]{"ar","az","bg","bn","ca","cs","da","de","el","en","es","fa","fi","fr","he","hi","hr","hu","id","it","ja","kk","km","ko","lo","ms","my","nl","no","pl","pt","ro","ru","sk","sl","sv","ta","th","tl","tr","ur","uz","vi","yue","zh"};
        }
        if (mode == Translator.TINY_AYA) {
            return new String[]{"en","de","fr","es","it","pt","ru","sr","uk","pl","cs","bg","ro","tr","ar","hi","id","vi","ja","ko","zh"};
        }
        // Curated TranslateGemma set for Lite10. Includes the Balkan pairs used by this project.
        return new String[]{"en","de","fr","es","it","pt","ru","uk","mk","sr","hr","sl","bg","ro","pl","cs","sk","tr","ar","he","hi","id","vi","ja","ko","zh"};
    }

    private String buildPrompt(String text, CustomLocale src, CustomLocale dst) {
        String srcCode = normalizeCode(src.getCode());
        String dstCode = normalizeCode(dst.getCode());
        String srcName = languageName(src);
        String dstName = languageName(dst);

        if (mode == Translator.MILMMT) {
            return "Translate this from " + srcName + " to " + dstName + ":\n" +
                    srcName + ": " + text + "\n" + dstName + ":";
        }

        if (mode == Translator.TINY_AYA) {
            return "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>" +
                    "You are a machine translation engine. Return only the translation, without notes or explanations." +
                    "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>" +
                    "Translate from " + srcName + " to " + dstName + ". Return only the " + dstName + " translation:\n" + text +
                    "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>";
        }

        return "<start_of_turn>user\nYou are a professional " + srcName + " (" + srcCode + ") to " + dstName + " (" + dstCode + ") translator. " +
                "Your goal is to accurately convey the meaning and nuances of the original " + srcName + " text while adhering to " + dstName + " grammar, vocabulary, and cultural sensitivities.\n" +
                "Produce only the " + dstName + " translation, without any additional explanations or commentary. Please translate the following " + srcName + " text into " + dstName + ":\n\n\n" +
                text + "<end_of_turn>\n<start_of_turn>model\n";
    }

    private static String languageName(CustomLocale locale) {
        String name = locale.getLocale().getDisplayLanguage(Locale.ENGLISH);
        return (name == null || name.trim().isEmpty()) ? locale.getCode() : name;
    }

    private static String normalizeCode(String code) {
        if (code == null) return "";
        String v = code.toLowerCase(Locale.ROOT);
        int p = v.indexOf('-');
        if (p > 0) v = v.substring(0, p);
        p = v.indexOf('_');
        if (p > 0) v = v.substring(0, p);
        return v;
    }

    private static String cleanup(String value, CustomLocale target) {
        if (value == null) return "";
        String out = value.trim();
        out = out.replace("<end_of_turn>", "").replace("<|END_OF_TURN_TOKEN|>", "").trim();
        String targetName = languageName(target) + ":";
        if (out.regionMatches(true, 0, targetName, 0, targetName.length())) {
            out = out.substring(targetName.length()).trim();
        }
        return out;
    }
}
''', encoding="utf-8")

# Translator: add model IDs and llama backend field.
t = translator.read_text(encoding="utf-8")
needle = '    public static final int HY_MT = 8;\n'
if needle not in t:
    raise SystemExit("HY_MT constant missing")
t = t.replace(needle, needle +
              '    public static final int MILMMT = 20;\n'
              '    public static final int TINY_AYA = 21;\n'
              '    public static final int TRANSLATE_GEMMA = 22;\n', 1)
needle = '    private LanguageResourcesManager languageResourcesManager;\n'
if needle not in t:
    raise SystemExit("languageResourcesManager field missing")
t = t.replace(needle, needle + '    private LlamaCppTranslationEngine llamaCppEngine;\n', 1)

# LLM modes do not initialize ONNX Runtime at all.
needle = '''            public void run() {\n                try {\n                    Log.i("RTranslatorLite9", "Initializing matched ONNX Runtime 1.27.1");'''
replacement = '''            public void run() {\n                if (LlamaCppTranslationEngine.isLlamaMode(mode)) {\n                    try {\n                        CustomLocale firstTextLanguage = global.getFirstTextLanguage(true);\n                        CustomLocale secondTextLanguage = global.getSecondTextLanguage(true);\n                        CustomLocale firstLanguage = global.getFirstLanguage(true);\n                        CustomLocale secondLanguage = global.getSecondLanguage(true);\n                        if (!restart || languageResourcesManager == null) {\n                            languageResourcesManager = new LanguageResourcesManager(global, mode, firstTextLanguage, secondTextLanguage, firstLanguage, secondLanguage);\n                        } else {\n                            languageResourcesManager.setModelMode(mode);\n                        }\n                        llamaCppEngine = new LlamaCppTranslationEngine(global, mode);\n                        llamaCppEngine.load();\n                        mainHandler.post(() -> initListener.onSuccess());\n                    } catch (Throwable e) {\n                        Log.e("RTranslatorLite10", "llama.cpp translation model initialization failed", e);\n                        mainHandler.post(() -> initListener.onFailure(new int[]{ErrorCodes.ERROR_LOADING_MODEL}, 0));\n                    }\n                    return;\n                }\n                try {\n                    Log.i("RTranslatorLite10", "Initializing matched ONNX Runtime 1.27.1");'''
if needle not in t:
    raise SystemExit("Lite9 ORT init marker not found")
t = t.replace(needle, replacement, 1)
t = t.replace('Log.i("RTranslatorLite9", "ONNX Runtime environment ready");',
              'Log.i("RTranslatorLite10", "ONNX Runtime environment ready");', 1)
t = t.replace('Log.e("RTranslatorLite9", "ONNX Runtime initialization failed", t);',
              'Log.e("RTranslatorLite10", "ONNX Runtime initialization failed", t);', 1)

needle = '''                    if(!restart){\n                        languageResourcesManager = new LanguageResourcesManager(global, mode, firstTextLanguage, secondTextLanguage, firstLanguage, secondLanguage);\n                    }\n'''
replacement = '''                    if(!restart || languageResourcesManager == null){\n                        languageResourcesManager = new LanguageResourcesManager(global, mode, firstTextLanguage, secondTextLanguage, firstLanguage, secondLanguage);\n                    } else {\n                        languageResourcesManager.setModelMode(mode);\n                    }\n'''
if needle not in t:
    raise SystemExit("LanguageResources init block missing")
t = t.replace(needle, replacement, 1)

# Model-specific safe unload/restart.
start = t.find('    private void destroy(GeneralListener listener){')
end = t.find('\n    public void restart(int mode, GeneralListener listener){', start)
if start < 0 or end < 0:
    raise SystemExit("destroy method bounds missing")
new_destroy = '''    private void destroy(GeneralListener listener){\n        final Thread t = new Thread("textTranslation") {\n            public void run() {\n                try {\n                    if (LlamaCppTranslationEngine.isLlamaMode(mode)) {\n                        if (llamaCppEngine != null) llamaCppEngine.close();\n                        llamaCppEngine = null;\n                        mainHandler.post(() -> listener.onSuccess());\n                        return;\n                    }\n                    if (mode == NLLB || mode == NLLB_CACHE || mode == MADLAD || mode == MADLAD_CACHE) {\n                        if (encoderSession != null) encoderSession.close();\n                        if (decoderSession != null) decoderSession.close();\n                        if (cacheInitSession != null) cacheInitSession.close();\n                        if (mode == MADLAD_CACHE) {\n                            if (embedSession != null) embedSession.close();\n                        } else {\n                            if (embedAndLmHeadSession != null) embedAndLmHeadSession.close();\n                        }\n                    } else if(mode == HY_MT) {\n                        if (decoderSession != null) decoderSession.close();\n                    } else if(mode == MOZILLA){\n                        unloadAllMozillaResources(null);\n                    }\n                    if (onnxEnv != null) {\n                        onnxEnv.close();\n                        onnxEnv = null;\n                    }\n                    mainHandler.post(() -> listener.onSuccess());\n                } catch (Throwable e) {\n                    Log.e("RTranslatorLite10", "Translator shutdown failed", e);\n                    mainHandler.post(() -> listener.onFailure(new int[]{ErrorCodes.ERROR_LOADING_MODEL},0));\n                }\n            }\n        };\n        t.start();\n    }\n'''
t = t[:start] + new_destroy + t[end:]

# Use llama.cpp directly before the tokenization/splitting path.
needle = '''            if(mode != MOZILLA){\n                int maxLength = 200;'''
replacement = '''            if (LlamaCppTranslationEngine.isLlamaMode(mode)) {\n                finalResult = llamaCppEngine.translate(textToTranslate, inputLanguage, outputLanguage);\n            } else if(mode != MOZILLA){\n                int maxLength = 200;'''
if needle not in t:
    raise SystemExit("performTextTranslation branch missing")
t = t.replace(needle, replacement, 1)

# Per-model language lists.
needle = '''        DocumentBuilderFactory documentBuilderFactory = DocumentBuilderFactory.newInstance();\n        try {\n            if(mode != MOZILLA) {'''
replacement = '''        if (LlamaCppTranslationEngine.isLlamaMode(mode)) {\n            for (String code : LlamaCppTranslationEngine.getSupportedLanguageCodes(mode)) {\n                languages.add(CustomLocale.getInstance(code));\n            }\n            return languages;\n        }\n        DocumentBuilderFactory documentBuilderFactory = DocumentBuilderFactory.newInstance();\n        try {\n            if(mode != MOZILLA) {'''
if needle not in t:
    raise SystemExit("getSupportedLanguages entry missing")
t = t.replace(needle, replacement, 1)
translator.write_text(t, encoding="utf-8")

# Replace upstream model manager with Lite10 selector.
manager.write_text(r'''package nie.translator.rtranslator.settings;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.RadioGroup;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import nie.translator.rtranslator.Global;
import nie.translator.rtranslator.R;
import nie.translator.rtranslator.voice_translation.neural_networks.translation.LlamaCppTranslationEngine;
import nie.translator.rtranslator.voice_translation.neural_networks.translation.Translator;

public class ModelManagerFragment extends Fragment {
    private SettingsActivity activity;
    private Global global;
    private RadioGroup radioGroup;
    private Button applyButton;
    private ProgressBar loading;
    private int activeModel;
    private int selectedModel;

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_model_manager, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        radioGroup = view.findViewById(R.id.model_radios);
        applyButton = view.findViewById(R.id.apply_button);
        loading = view.findViewById(R.id.loading_models);
    }

    @Override
    public void onActivityCreated(@Nullable Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
        activity = (SettingsActivity) requireActivity();
        global = (Global) activity.getApplication();
        SharedPreferences sp = global.getSharedPreferences("default", Context.MODE_PRIVATE);
        activeModel = global.getTranslator() != null ? global.getTranslator().getMode() : sp.getInt("selectedTranslationModel", Translator.MADLAD_CACHE);
        selectedModel = activeModel;
        checkRadio(activeModel);

        radioGroup.setOnCheckedChangeListener((group, checkedId) -> {
            if (checkedId == R.id.madlad_radio) selectedModel = Translator.MADLAD_CACHE;
            else if (checkedId == R.id.milmmt_radio) selectedModel = Translator.MILMMT;
            else if (checkedId == R.id.tiny_aya_radio) selectedModel = Translator.TINY_AYA;
            else if (checkedId == R.id.translate_gemma_radio) selectedModel = Translator.TRANSLATE_GEMMA;
            applyButton.setVisibility(selectedModel == activeModel ? View.INVISIBLE : View.VISIBLE);
        });
        applyButton.setOnClickListener(v -> applySelectedModel());
    }

    private void applySelectedModel() {
        if (LlamaCppTranslationEngine.isLlamaMode(selectedModel) && LlamaCppTranslationEngine.findModelFile(selectedModel) == null) {
            Toast.makeText(activity, "GGUF not found. Put the model in: " + LlamaCppTranslationEngine.getModelDirectory(selectedModel).getAbsolutePath(), Toast.LENGTH_LONG).show();
            selectedModel = activeModel;
            checkRadio(activeModel);
            applyButton.setVisibility(View.INVISIBLE);
            return;
        }
        setUiBusy(true);
        SharedPreferences sp = global.getSharedPreferences("default", Context.MODE_PRIVATE);
        sp.edit().putInt("selectedTranslationModel", selectedModel).apply();
        final int requested = selectedModel;
        final int fallback = activeModel;
        global.restartTranslator(new Translator.GeneralListener() {
            @Override
            public void onSuccess() {
                activeModel = requested;
                selectedModel = requested;
                setUiBusy(false);
                applyButton.setVisibility(View.INVISIBLE);
                Toast.makeText(activity, "Translation model loaded", Toast.LENGTH_SHORT).show();
            }

            @Override
            public void onFailure(int[] reasons, long value) {
                sp.edit().putInt("selectedTranslationModel", fallback).apply();
                selectedModel = fallback;
                checkRadio(fallback);
                Toast.makeText(activity, "Model failed to load. Restoring previous translator...", Toast.LENGTH_LONG).show();
                global.restartTranslator(new Translator.GeneralListener() {
                    @Override
                    public void onSuccess() {
                        activeModel = fallback;
                        setUiBusy(false);
                        applyButton.setVisibility(View.INVISIBLE);
                    }

                    @Override
                    public void onFailure(int[] r, long v) {
                        setUiBusy(false);
                    }
                });
            }
        });
    }

    private void setUiBusy(boolean busy) {
        radioGroup.setEnabled(!busy);
        applyButton.setEnabled(!busy);
        loading.setVisibility(busy ? View.VISIBLE : View.INVISIBLE);
    }

    private void checkRadio(int mode) {
        if (mode == Translator.MILMMT) radioGroup.check(R.id.milmmt_radio);
        else if (mode == Translator.TINY_AYA) radioGroup.check(R.id.tiny_aya_radio);
        else if (mode == Translator.TRANSLATE_GEMMA) radioGroup.check(R.id.translate_gemma_radio);
        else radioGroup.check(R.id.madlad_radio);
    }
}
''', encoding="utf-8")

layout.write_text(r'''<?xml version="1.0" encoding="utf-8"?>
<androidx.constraintlayout.widget.ConstraintLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    android:layout_width="match_parent"
    android:layout_height="match_parent">

    <ScrollView
        android:layout_width="0dp"
        android:layout_height="0dp"
        app:layout_constraintTop_toTopOf="parent"
        app:layout_constraintBottom_toTopOf="@id/apply_button"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintEnd_toEndOf="parent">
        <LinearLayout
            android:layout_width="match_parent"
            android:layout_height="wrap_content"
            android:orientation="vertical"
            android:padding="24dp">
            <TextView
                android:layout_width="wrap_content"
                android:layout_height="wrap_content"
                android:text="Translation model"
                android:textStyle="bold"
                android:textSize="18sp" />
            <RadioGroup
                android:id="@+id/model_radios"
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:layout_marginTop="12dp">
                <RadioButton android:id="@+id/madlad_radio" android:layout_width="wrap_content" android:layout_height="wrap_content" android:text="MADLAD-400 3B INT4 (ONNX)" />
                <RadioButton android:id="@+id/milmmt_radio" android:layout_width="wrap_content" android:layout_height="wrap_content" android:text="MiLMMT-46-1B v0.1 (GGUF)" />
                <RadioButton android:id="@+id/tiny_aya_radio" android:layout_width="wrap_content" android:layout_height="wrap_content" android:text="Tiny Aya Global 3.35B (GGUF)" />
                <RadioButton android:id="@+id/translate_gemma_radio" android:layout_width="wrap_content" android:layout_height="wrap_content" android:text="TranslateGemma 4B (GGUF)" />
            </RadioGroup>
            <TextView
                android:layout_width="match_parent"
                android:layout_height="wrap_content"
                android:layout_marginTop="20dp"
                android:text="GGUF files are not bundled. Put one .gguf file in:\n/models/Translation/MiLMMT/\n/models/Translation/TinyAya/\n/models/Translation/TranslateGemma/\n\nOnly the selected translation model is loaded into RAM." />
        </LinearLayout>
    </ScrollView>

    <Button
        android:id="@+id/apply_button"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="Apply"
        android:visibility="invisible"
        android:layout_marginBottom="16dp"
        app:layout_constraintBottom_toBottomOf="parent"
        app:layout_constraintStart_toStartOf="parent"
        app:layout_constraintEnd_toEndOf="parent" />

    <ProgressBar
        android:id="@+id/loading_models"
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:visibility="invisible"
        android:layout_marginEnd="24dp"
        app:layout_constraintEnd_toEndOf="parent"
        app:layout_constraintTop_toTopOf="@id/apply_button"
        app:layout_constraintBottom_toBottomOf="@id/apply_button" />
</androidx.constraintlayout.widget.ConstraintLayout>
''', encoding="utf-8")

# Add local llama AAR and runtime dependencies; create separate Lite10 identity.
b = build.read_text(encoding="utf-8")
if 'applicationId "nie.translator.rtranslator.lite9"' not in b:
    raise SystemExit("Lite9 app id missing")
b = b.replace('applicationId "nie.translator.rtranslator.lite9"', 'applicationId "nie.translator.rtranslator.lite10"', 1)
b = b.replace('versionCode 30011', 'versionCode 30012', 1)
b = b.replace("versionName '3.0.0-alpha3-lite9'", "versionName '3.0.0-alpha3-lite10'", 1)
needle = "implementation files('libs/sherpa-onnx-1.13.5-no-ort-arm64.aar')"
if needle not in b:
    raise SystemExit("Sherpa local AAR dependency missing")
b = b.replace(needle, needle + "\n    implementation files('libs/llama-android-lite10.aar')", 1)
b = b.replace("implementation 'org.jetbrains.kotlin:kotlin-stdlib:1.7.20'",
              "implementation 'org.jetbrains.kotlin:kotlin-stdlib:2.3.0'\n    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2'", 1)
build.write_text(b, encoding="utf-8")

changed_labels = 0
for strings_xml in (ROOT / "app/src/main/res").glob("values*/strings.xml"):
    s = strings_xml.read_text(encoding="utf-8")
    if "RTranslator Lite9" in s:
        strings_xml.write_text(s.replace("RTranslator Lite9", "RTranslator Lite10"), encoding="utf-8")
        changed_labels += 1
if changed_labels == 0:
    raise SystemExit("Lite9 app label missing")

for p in provider_files:
    s = p.read_text(encoding="utf-8")
    if "com.gallery.RTranslator.lite9.provider" not in s:
        raise SystemExit(f"Lite9 provider missing in {p}")
    p.write_text(s.replace("com.gallery.RTranslator.lite9.provider", "com.gallery.RTranslator.lite10.provider"), encoding="utf-8")

for p in (ROOT / "app/src/main/java").rglob("*.java"):
    s = p.read_text(encoding="utf-8")
    if "RTranslatorLite9" in s:
        p.write_text(s.replace("RTranslatorLite9", "RTranslatorLite10"), encoding="utf-8")

print("Lite10 llama.cpp model selector patch complete:")
print(" - MADLAD remains ONNX fallback/default")
print(" - MiLMMT / Tiny Aya / TranslateGemma GGUF modes added")
print(" - one selected GGUF is loaded into RAM")
print(" - missing GGUF is rejected safely by Model Manager")
print(" - model-specific raw prompt adapters added")
print(" - applicationId nie.translator.rtranslator.lite10")
