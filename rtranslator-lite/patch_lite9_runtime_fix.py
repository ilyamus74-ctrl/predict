#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "RTranslator")
build_gradle = ROOT / "app/build.gradle"
translator_path = ROOT / "app/src/main/java/nie/translator/rtranslator/voice_translation/neural_networks/translation/Translator.java"
recognizer_path = ROOT / "app/src/main/java/nie/translator/rtranslator/voice_translation/neural_networks/voice/Recognizer.java"
access_path = ROOT / "app/src/main/java/nie/translator/rtranslator/access/AccessActivity.java"
activity_path = ROOT / "app/src/main/java/nie/translator/rtranslator/voice_translation/VoiceTranslationActivity.java"
provider_files = [
    ROOT / "app/src/main/AndroidManifest.xml",
    ROOT / "app/src/main/java/nie/translator/rtranslator/access/UserDataFragment.java",
    ROOT / "app/src/main/java/nie/translator/rtranslator/settings/SettingsFragment.java",
]

for p in [build_gradle, translator_path, recognizer_path, access_path, activity_path, *provider_files]:
    if not p.exists():
        raise SystemExit(f"Required file missing: {p}")

# ---------------------------------------------------------------------------
# 1) Replace the Lite8 mixed-version ORT arrangement with the Android package
#    built by csukuangfj/onnxruntime-libs for sherpa-onnx. Its native core and
#    Java JNI bridge are both produced from ONNX Runtime 1.27.1.
# ---------------------------------------------------------------------------
bg = build_gradle.read_text(encoding="utf-8")
required = [
    'applicationId "nie.translator.rtranslator.lite8"',
    'versionCode 30010',
    "versionName '3.0.0-alpha3-lite8'",
    "implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.27.1'",
    "implementation files('libs/sherpa-onnx-1.13.5-no-ort-arm64.aar')",
]
for value in required:
    if value not in bg:
        raise SystemExit(f"Expected Lite8 value not found: {value}")

bg = bg.replace(
    "implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.27.1'",
    "implementation files('libs/onnxruntime-android-1.27.1-full-arm64.aar')",
    1,
)
bg = bg.replace('applicationId "nie.translator.rtranslator.lite8"',
                'applicationId "nie.translator.rtranslator.lite9"', 1)
bg = bg.replace('versionCode 30010', 'versionCode 30011', 1)
bg = bg.replace("versionName '3.0.0-alpha3-lite8'",
                "versionName '3.0.0-alpha3-lite9'", 1)
build_gradle.write_text(bg, encoding="utf-8")

# ---------------------------------------------------------------------------
# 2) OrtEnvironment.getEnvironment() used to sit outside Translator's existing
#    Exception handler. A linkage Error therefore killed the process at startup.
#    Catch all Java linkage/runtime failures and surface the normal model error.
#    Native SIGSEGV cannot be caught here, hence the matched 1.27.1 native pair.
# ---------------------------------------------------------------------------
translator = translator_path.read_text(encoding="utf-8")
old_env = '                onnxEnv = OrtEnvironment.getEnvironment();\n'
new_env = '''                try {\n                    Log.i("RTranslatorLite9", "Initializing matched ONNX Runtime 1.27.1");\n                    onnxEnv = OrtEnvironment.getEnvironment();\n                    Log.i("RTranslatorLite9", "ONNX Runtime environment ready");\n                } catch (Throwable t) {\n                    Log.e("RTranslatorLite9", "ONNX Runtime initialization failed", t);\n                    mainHandler.post(() -> initListener.onFailure(new int[]{ErrorCodes.ERROR_LOADING_MODEL}, 0));\n                    return;\n                }\n'''
if old_env not in translator:
    raise SystemExit("Could not find OrtEnvironment startup line")
translator = translator.replace(old_env, new_env, 1)
translator_path.write_text(translator, encoding="utf-8")

# ---------------------------------------------------------------------------
# 3) Separate install identity so the broken Lite8 preferences/crash-loop are
#    not reused. Update all known hard-coded authorities and visible markers.
# ---------------------------------------------------------------------------
changed_labels = 0
for strings_xml in (ROOT / "app/src/main/res").glob("values*/strings.xml"):
    text = strings_xml.read_text(encoding="utf-8")
    if "RTranslator Lite8" in text:
        strings_xml.write_text(text.replace("RTranslator Lite8", "RTranslator Lite9"), encoding="utf-8")
        changed_labels += 1
if changed_labels == 0:
    raise SystemExit("No RTranslator Lite8 app_name resource found for Lite9")

for path in provider_files:
    text = path.read_text(encoding="utf-8")
    if "com.gallery.RTranslator.lite8.provider" not in text:
        raise SystemExit(f"Expected Lite8 FileProvider authority not found in {path}")
    path.write_text(text.replace(
        "com.gallery.RTranslator.lite8.provider",
        "com.gallery.RTranslator.lite9.provider",
    ), encoding="utf-8")

recognizer = recognizer_path.read_text(encoding="utf-8")
if '"RTranslatorLite8"' not in recognizer:
    raise SystemExit("Lite8 recognizer log tag not found")
recognizer_path.write_text(recognizer.replace('"RTranslatorLite8"', '"RTranslatorLite9"'), encoding="utf-8")

activity = activity_path.read_text(encoding="utf-8")
activity_path.write_text(activity.replace('"RTranslatorLite8"', '"RTranslatorLite9"'), encoding="utf-8")

access = access_path.read_text(encoding="utf-8")
access_path.write_text(access.replace("Lite8: sherpa Whisper INT8", "Lite9: sherpa Whisper INT8"), encoding="utf-8")

print("Lite9 runtime fix complete:")
print(" - matched ONNX Runtime 1.27.1 full Android AAR expected")
print(" - sherpa 1.13.5 stays no-ORT to avoid duplicate core")
print(" - Translator catches Java ORT linkage/runtime initialization failures")
print(" - applicationId nie.translator.rtranslator.lite9")
print(" - version 3.0.0-alpha3-lite9")
print(" - FileProvider authority com.gallery.RTranslator.lite9.provider")
