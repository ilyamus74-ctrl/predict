#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "RTranslator")
translator = ROOT / "app/src/main/java/nie/translator/rtranslator/voice_translation/neural_networks/translation/Translator.java"
global_java = ROOT / "app/src/main/java/nie/translator/rtranslator/Global.java"
manifest = ROOT / "app/src/main/AndroidManifest.xml"
user_data = ROOT / "app/src/main/java/nie/translator/rtranslator/access/UserDataFragment.java"
settings_fragment = ROOT / "app/src/main/java/nie/translator/rtranslator/settings/SettingsFragment.java"
download_fragment = ROOT / "app/src/main/java/nie/translator/rtranslator/access/DownloadFragment.java"
build_gradle = ROOT / "app/build.gradle"

for path in (translator, global_java, manifest, user_data, settings_fragment, download_fragment, build_gradle):
    if not path.exists():
        raise SystemExit(f"Required RTranslator file not found: {path}")

# --- MADLAD INT4 / low-memory backend ---
text = translator.read_text(encoding="utf-8")
start_marker = '}else if(mode == MADLAD || mode == MADLAD_CACHE){  //madlad\n'
end_marker = '        }else {  //hy-mt\n'
start = text.find(start_marker)
end = text.find(end_marker, start)
if start < 0 or end < 0:
    raise SystemExit("Could not locate MADLAD initialization block; upstream changed")

replacement = '''}else if(mode == MADLAD || mode == MADLAD_CACHE){  //madlad\n            // RTranslator Lite: use the actual layout of the official Madlad.zip first.\n            String madladBasePath = Environment.getExternalStorageDirectory().getPath() + "/models/Translation/Madlad";\n\n            File officialInt4Dir = new File(madladBasePath + "/Int4Acc4");\n            File legacyInt4Dir = new File(madladBasePath + "/Int4_16");\n            File int4Dir = officialInt4Dir.exists() ? officialInt4Dir : legacyInt4Dir;\n            File int4Encoder = new File(int4Dir, "madlad_encoder_4bit.onnx");\n            File int4Decoder = new File(int4Dir, "madlad_decoder_4bit.onnx");\n            File int4Cache = new File(int4Dir, "madlad_cache_initializer_4bit.onnx");\n            boolean hasInt4Madlad = int4Encoder.exists() && int4Decoder.exists() && int4Cache.exists();\n\n            if(hasInt4Madlad) {\n                encoderPath = int4Encoder.getPath();\n                decoderPath = int4Decoder.getPath();\n                vocabPath = madladBasePath + "/spiece.model";\n                embedAndLmHeadPath = madladBasePath + "/madlad_embed_8bit.onnx";\n                cacheInitializerPath = int4Cache.getPath();\n                Log.i("RTranslatorLite", "Using MADLAD INT4 low-memory model from " + int4Dir.getName());\n            } else {\n                encoderPath = madladBasePath + "/Int8WO/madlad_encoder_8bit.onnx";\n                decoderPath = madladBasePath + "/Int8WO/madlad_decoder_8bit.onnx";\n                vocabPath = madladBasePath + "/spiece.model";\n                embedAndLmHeadPath = madladBasePath + "/madlad_embed_8bit.onnx";\n                cacheInitializerPath = madladBasePath + "/Int8WO/madlad_cache_initializer_8bit.onnx";\n                Log.w("RTranslatorLite", "MADLAD INT4 files not found; trying legacy INT8WO fallback");\n            }\n'''
text = text[:start] + replacement + text[end:]

old_arena = '                        boolean arena = true;\n'
new_arena = ('                        // Reduce peak native memory on phones with limited RAM.\n'
             '                        boolean arena = !(mode == MADLAD || mode == MADLAD_CACHE);\n')
if old_arena not in text:
    raise SystemExit("Could not locate ORT arena setting; upstream changed")
text = text.replace(old_arena, new_arena, 1)
translator.write_text(text, encoding="utf-8")

# MADLAD cache is the Lite default translator.
g = global_java.read_text(encoding="utf-8")
needle = 'sharedPreferences.getInt("selectedTranslationModel", Translator.MOZILLA)'
count = g.count(needle)
if count < 2:
    raise SystemExit(f"Expected at least 2 default translator selectors, found {count}")
g = g.replace(needle, 'sharedPreferences.getInt("selectedTranslationModel", Translator.MADLAD_CACHE)')
global_java.write_text(g, encoding="utf-8")

# Unique FileProvider authority so stock and Lite can coexist.
old_authority = 'com.gallery.RTranslator.2.0.provider'
new_authority = 'com.gallery.RTranslator.lite.provider'
for path in (manifest, user_data, settings_fragment):
    value = path.read_text(encoding="utf-8")
    if old_authority not in value:
        raise SystemExit(f"Expected FileProvider authority not found in {path}")
    path.write_text(value.replace(old_authority, new_authority), encoding="utf-8")

# --- Fix alpha3 onboarding: Lite does not use NLLB for translation. ---
# Keep only the six Whisper files needed by Recognizer. This drops the first-start
# download from roughly 1.24 GB to roughly 290 MB and avoids the NLLB dead-end.
df = download_fragment.read_text(encoding="utf-8")

whisper_urls = '''public static final String[] DOWNLOAD_URLS = {
            "https://github.com/niedev/RTranslator/releases/download/2.0.0/Whisper_cache_initializer.onnx",
            "https://github.com/niedev/RTranslator/releases/download/2.0.0/Whisper_cache_initializer_batch.onnx",
            "https://github.com/niedev/RTranslator/releases/download/2.0.0/Whisper_decoder.onnx",
            "https://github.com/niedev/RTranslator/releases/download/2.0.0/Whisper_detokenizer.onnx",
            "https://github.com/niedev/RTranslator/releases/download/2.0.0/Whisper_encoder.onnx",
            "https://github.com/niedev/RTranslator/releases/download/2.0.0/Whisper_initializer.onnx"
    };'''
whisper_names = '''public static final String[] DOWNLOAD_NAMES = {
            "Whisper_cache_initializer.onnx",
            "Whisper_cache_initializer_batch.onnx",
            "Whisper_decoder.onnx",
            "Whisper_detokenizer.onnx",
            "Whisper_encoder.onnx",
            "Whisper_initializer.onnx"
    };'''
whisper_sizes = '''public static final int[] DOWNLOAD_SIZES = {
            14000,
            14000,
            173000,
            461,
            88000,
            69
    };'''

df, n1 = re.subn(r'public static final String\[\] DOWNLOAD_URLS = \{.*?\n    \};', whisper_urls, df, count=1, flags=re.S)
df, n2 = re.subn(r'public static final String\[\] DOWNLOAD_NAMES = \{.*?\n    \};', whisper_names, df, count=1, flags=re.S)
df, n3 = re.subn(r'public static final int\[\] DOWNLOAD_SIZES = \{.*?\n    \};', whisper_sizes, df, count=1, flags=re.S)
if (n1, n2, n3) != (1, 1, 1):
    raise SystemExit(f"Could not replace downloader arrays: {(n1, n2, n3)}")

# Recover automatically if Android DownloadManager lost the saved ID, a previous
# download failed, or the saved ID belongs to an old NLLB URL no longer used by Lite.
old_state = '''            SharedPreferences sharedPreferences = global.getSharedPreferences("default", Context.MODE_PRIVATE);\n            long currentDownloadId = sharedPreferences.getLong("currentDownloadId", -1);\n\n            if(currentDownloadId == -1){'''
new_state = '''            SharedPreferences sharedPreferences = global.getSharedPreferences("default", Context.MODE_PRIVATE);\n            long currentDownloadId = sharedPreferences.getLong("currentDownloadId", -1);\n\n            boolean staleDownload = false;\n            if(currentDownloadId == -3){\n                staleDownload = true;\n            }else if(currentDownloadId >= 0){\n                int status = downloader.getRunningDownloadStatus();\n                boolean knownUrl = downloader.findDownloadUrlIndex(currentDownloadId) >= 0;\n                staleDownload = !knownUrl || status == -1 || status == DownloadManager.STATUS_FAILED;\n            }\n\n            if(staleDownload){\n                downloader.cancelRunningDownload();\n                SharedPreferences.Editor recoveryEditor = sharedPreferences.edit();\n                recoveryEditor.putLong("currentDownloadId", -1);\n                recoveryEditor.putString("lastDownloadSuccess", "");\n                recoveryEditor.putString("lastTransferSuccess", "");\n                recoveryEditor.putString("lastTransferFailure", "");\n                recoveryEditor.apply();\n                currentDownloadId = -1;\n            }\n\n            if(currentDownloadId == -1){'''
if old_state not in df:
    raise SystemExit("Could not locate downloader state block; upstream changed")
df = df.replace(old_state, new_state, 1)
download_fragment.write_text(df, encoding="utf-8")

# Package/version metadata.
bg = build_gradle.read_text(encoding="utf-8")
bg = bg.replace('applicationId "nie.translator.rtranslator"', 'applicationId "nie.translator.rtranslator.lite"', 1)
bg = re.sub(r'versionCode\s+\d+', 'versionCode 30006', bg, count=1)
bg = re.sub(r"versionName\s+'[^']+'", "versionName '3.0.0-alpha3-lite4'", bg, count=1)
build_gradle.write_text(bg, encoding="utf-8")

print("Patched:")
print(" - MADLAD INT4 / Int4Acc4 low-memory backend")
print(" - MADLAD_CACHE default")
print(" - unique Lite FileProvider authority")
print(" - first-start download reduced to Whisper only (~290 MB)")
print(" - stale/failed/old NLLB DownloadManager state auto-recovers")
print(" - version 3.0.0-alpha3-lite4")
