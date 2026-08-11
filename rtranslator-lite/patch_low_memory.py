#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "RTranslator")
translator = ROOT / "app/src/main/java/nie/translator/rtranslator/voice_translation/neural_networks/translation/Translator.java"
build_gradle = ROOT / "app/build.gradle"

if not translator.exists() or not build_gradle.exists():
    raise SystemExit(f"RTranslator tree not found under {ROOT}")

text = translator.read_text(encoding="utf-8")

start_marker = '}else if(mode == MADLAD || mode == MADLAD_CACHE){  //madlad\n'
end_marker = '        }else {  //hy-mt\n'
start = text.find(start_marker)
end = text.find(end_marker, start)
if start < 0 or end < 0:
    raise SystemExit("Could not locate MADLAD initialization block; upstream changed")

replacement = '''}else if(mode == MADLAD || mode == MADLAD_CACHE){  //madlad\n            // RTranslator Lite: prefer the experimental INT4 layout when present,\n            // but keep compatibility with the alpha3 INT8WO package.\n            String madladBasePath = Environment.getExternalStorageDirectory().getPath() + "/models/Translation/Madlad";\n            File int4Encoder = new File(madladBasePath + "/Int4_16/madlad_encoder_4bit.onnx");\n            File int4Decoder = new File(madladBasePath + "/Int4_16/madlad_decoder_4bit.onnx");\n            File int4Cache = new File(madladBasePath + "/Int4_16/madlad_cache_initializer_4bit.onnx");\n            boolean hasInt4Madlad = int4Encoder.exists() && int4Decoder.exists() && int4Cache.exists();\n\n            if(hasInt4Madlad) {\n                encoderPath = int4Encoder.getPath();\n                decoderPath = int4Decoder.getPath();\n                vocabPath = madladBasePath + "/spiece.model";\n                // alpha3 uses the shared embedding model for both INT8 and INT4 decoder layouts\n                embedAndLmHeadPath = madladBasePath + "/madlad_embed_8bit.onnx";\n                cacheInitializerPath = int4Cache.getPath();\n                Log.i("RTranslatorLite", "Using MADLAD INT4 low-memory model");\n            } else {\n                encoderPath = madladBasePath + "/Int8WO/madlad_encoder_8bit.onnx";\n                decoderPath = madladBasePath + "/Int8WO/madlad_decoder_8bit.onnx";\n                vocabPath = madladBasePath + "/spiece.model";\n                embedAndLmHeadPath = madladBasePath + "/madlad_embed_8bit.onnx";\n                cacheInitializerPath = madladBasePath + "/Int8WO/madlad_cache_initializer_8bit.onnx";\n                Log.w("RTranslatorLite", "INT4 MADLAD not found; falling back to INT8WO");\n            }\n'''
text = text[:start] + replacement + text[end:]

old_arena = '                        boolean arena = true;\n'
new_arena = ('                        // Lower peak native-memory use on 8 GB devices.\n'
             '                        // MADLAD is large enough that ORT CPU arenas can trigger OOM.\n'
             '                        boolean arena = !(mode == MADLAD || mode == MADLAD_CACHE);\n')
if old_arena not in text:
    raise SystemExit("Could not locate ORT arena setting; upstream changed")
text = text.replace(old_arena, new_arena, 1)
translator.write_text(text, encoding="utf-8")

bg = build_gradle.read_text(encoding="utf-8")
bg = bg.replace('applicationId "nie.translator.rtranslator"', 'applicationId "nie.translator.rtranslator.lite"', 1)
bg = re.sub(r'versionCode\s+\d+', 'versionCode 30003', bg, count=1)
bg = re.sub(r"versionName\s+'[^']+'", "versionName '3.0.0-alpha3-lite'", bg, count=1)
build_gradle.write_text(bg, encoding="utf-8")

print("Patched:")
print(" - MADLAD INT4-first with INT8 fallback")
print(" - ORT arenas disabled for MADLAD low-memory mode")
print(" - side-by-side applicationId nie.translator.rtranslator.lite")
print(" - version 3.0.0-alpha3-lite")
