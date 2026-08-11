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
build_gradle = ROOT / "app/build.gradle"

for path in (translator, global_java, manifest, user_data, settings_fragment, build_gradle):
    if not path.exists():
        raise SystemExit(f"Required RTranslator file not found: {path}")

text = translator.read_text(encoding="utf-8")

start_marker = '}else if(mode == MADLAD || mode == MADLAD_CACHE){  //madlad\n'
end_marker = '        }else {  //hy-mt\n'
start = text.find(start_marker)
end = text.find(end_marker, start)
if start < 0 or end < 0:
    raise SystemExit("Could not locate MADLAD initialization block; upstream changed")

replacement = '''}else if(mode == MADLAD || mode == MADLAD_CACHE){  //madlad\n            // RTranslator Lite: use the actual layout of the official Madlad.zip first.\n            // The v3.00 source still contains an older commented path (Int4_16), while\n            // the published package uses Int4Acc4. Keep legacy/fallback layouts too.\n            String madladBasePath = Environment.getExternalStorageDirectory().getPath() + "/models/Translation/Madlad";\n\n            File officialInt4Dir = new File(madladBasePath + "/Int4Acc4");\n            File legacyInt4Dir = new File(madladBasePath + "/Int4_16");\n            File int4Dir = officialInt4Dir.exists() ? officialInt4Dir : legacyInt4Dir;\n            File int4Encoder = new File(int4Dir, "madlad_encoder_4bit.onnx");\n            File int4Decoder = new File(int4Dir, "madlad_decoder_4bit.onnx");\n            File int4Cache = new File(int4Dir, "madlad_cache_initializer_4bit.onnx");\n            boolean hasInt4Madlad = int4Encoder.exists() && int4Decoder.exists() && int4Cache.exists();\n\n            if(hasInt4Madlad) {\n                encoderPath = int4Encoder.getPath();\n                decoderPath = int4Decoder.getPath();\n                vocabPath = madladBasePath + "/spiece.model";\n                embedAndLmHeadPath = madladBasePath + "/madlad_embed_8bit.onnx";\n                cacheInitializerPath = int4Cache.getPath();\n                Log.i("RTranslatorLite", "Using MADLAD INT4 low-memory model from " + int4Dir.getName());\n            } else {\n                // Compatibility fallback for older/private alpha model packages.\n                encoderPath = madladBasePath + "/Int8WO/madlad_encoder_8bit.onnx";\n                decoderPath = madladBasePath + "/Int8WO/madlad_decoder_8bit.onnx";\n                vocabPath = madladBasePath + "/spiece.model";\n                embedAndLmHeadPath = madladBasePath + "/madlad_embed_8bit.onnx";\n                cacheInitializerPath = madladBasePath + "/Int8WO/madlad_cache_initializer_8bit.onnx";\n                Log.w("RTranslatorLite", "MADLAD INT4 files not found; trying legacy INT8WO fallback");\n            }\n'''
text = text[:start] + replacement + text[end:]

old_arena = '                        boolean arena = true;\n'
new_arena = ('                        // Reduce peak native memory on phones with limited RAM.\n'
             '                        // MADLAD is large enough that ORT CPU arenas/memory patterns can push it into OOM.\n'
             '                        boolean arena = !(mode == MADLAD || mode == MADLAD_CACHE);\n')
if old_arena not in text:
    raise SystemExit("Could not locate ORT arena setting; upstream changed")
text = text.replace(old_arena, new_arena, 1)
translator.write_text(text, encoding="utf-8")

# Make MADLAD cache backend the first-run/default translator so the Lite build does
# not silently start with Mozilla/Bergamot when no preference has been saved yet.
g = global_java.read_text(encoding="utf-8")
needle = 'sharedPreferences.getInt("selectedTranslationModel", Translator.MOZILLA)'
count = g.count(needle)
if count < 2:
    raise SystemExit(f"Expected at least 2 default translator selectors, found {count}")
g = g.replace(needle, 'sharedPreferences.getInt("selectedTranslationModel", Translator.MADLAD_CACHE)')
global_java.write_text(g, encoding="utf-8")

# Give the Lite package its own FileProvider authority. The original authority is
# hard-coded in the manifest and in two GalleryImageSelector call sites, so all
# three must change for stock RTranslator and Lite to coexist.
old_authority = 'com.gallery.RTranslator.2.0.provider'
new_authority = 'com.gallery.RTranslator.lite.provider'
for path in (manifest, user_data, settings_fragment):
    value = path.read_text(encoding="utf-8")
    if old_authority not in value:
        raise SystemExit(f"Expected FileProvider authority not found in {path}")
    path.write_text(value.replace(old_authority, new_authority), encoding="utf-8")

bg = build_gradle.read_text(encoding="utf-8")
bg = bg.replace('applicationId "nie.translator.rtranslator"', 'applicationId "nie.translator.rtranslator.lite"', 1)
bg = re.sub(r'versionCode\s+\d+', 'versionCode 30005', bg, count=1)
bg = re.sub(r"versionName\s+'[^']+'", "versionName '3.0.0-alpha3-lite3'", bg, count=1)
build_gradle.write_text(bg, encoding="utf-8")

print("Patched:")
print(" - official MADLAD INT4 path Int4Acc4 first")
print(" - legacy Int4_16 and Int8WO compatibility fallbacks")
print(" - MADLAD_CACHE is the first-run/default translation backend")
print(" - ORT arenas disabled for MADLAD low-memory mode")
print(" - side-by-side applicationId and unique FileProvider authority")
print(" - version 3.0.0-alpha3-lite3")
