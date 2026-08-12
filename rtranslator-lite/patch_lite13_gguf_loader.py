#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "RTranslator")
engine = ROOT / "app/src/main/java/nie/translator/rtranslator/voice_translation/neural_networks/translation/LlamaCppTranslationEngine.java"
build = ROOT / "app/build.gradle"
provider_files = [
    ROOT / "app/src/main/AndroidManifest.xml",
    ROOT / "app/src/main/java/nie/translator/rtranslator/access/UserDataFragment.java",
    ROOT / "app/src/main/java/nie/translator/rtranslator/settings/SettingsFragment.java",
]
for p in [engine, build, *provider_files]:
    if not p.exists():
        raise SystemExit(f"Missing required file: {p}")

# Validate the actual GGUF magic before handing a file to native llama.cpp.
e = engine.read_text(encoding="utf-8")
needle = '''            if (!modelFile.isFile() || !modelFile.canRead()) {\n                lastLoadError = "GGUF file is not readable";\n                throw new IllegalStateException("Cannot read GGUF: " + modelFile.getAbsolutePath());\n            }\n            Log.i(TAG, "Loading GGUF model: " + modelFile.getAbsolutePath() + " (" + modelFile.length() + " bytes)");\n'''
replacement = '''            if (!modelFile.isFile() || !modelFile.canRead()) {\n                lastLoadError = "GGUF file is not readable";\n                throw new IllegalStateException("Cannot read GGUF: " + modelFile.getAbsolutePath());\n            }\n            if (modelFile.length() < 16) {\n                lastLoadError = "GGUF file is too small (" + modelFile.length() + " bytes)";\n                throw new IllegalStateException(lastLoadError);\n            }\n            try (java.io.FileInputStream in = new java.io.FileInputStream(modelFile)) {\n                byte[] magic = new byte[4];\n                if (in.read(magic) != 4 || magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F') {\n                    lastLoadError = "Invalid GGUF header; file does not start with GGUF magic";\n                    throw new IllegalStateException(lastLoadError);\n                }\n            }\n            Log.i(TAG, "Loading GGUF model: " + modelFile.getAbsolutePath() + " (" + modelFile.length() + " bytes)");\n'''
if needle not in e:
    raise SystemExit("Lite12 readable-file block missing")
e = e.replace(needle, replacement, 1)
engine.write_text(e, encoding="utf-8")

# Separate Lite13 package so Lite12 remains installable for comparison.
b = build.read_text(encoding="utf-8")
if 'applicationId "nie.translator.rtranslator.lite12"' not in b:
    raise SystemExit("Lite12 applicationId missing")
b = b.replace('applicationId "nie.translator.rtranslator.lite12"', 'applicationId "nie.translator.rtranslator.lite13"', 1)
b = b.replace('versionCode 30014', 'versionCode 30015', 1)
b = b.replace("versionName '3.0.0-alpha3-lite12'", "versionName '3.0.0-alpha3-lite13'", 1)
build.write_text(b, encoding="utf-8")

changed_labels = 0
for strings_xml in (ROOT / "app/src/main/res").glob("values*/strings.xml"):
    s = strings_xml.read_text(encoding="utf-8")
    if "RTranslator Lite12" in s:
        strings_xml.write_text(s.replace("RTranslator Lite12", "RTranslator Lite13"), encoding="utf-8")
        changed_labels += 1
if changed_labels == 0:
    raise SystemExit("Lite12 app label missing")

for p in provider_files:
    s = p.read_text(encoding="utf-8")
    if "com.gallery.RTranslator.lite12.provider" not in s:
        raise SystemExit(f"Lite12 provider missing in {p}")
    p.write_text(s.replace("com.gallery.RTranslator.lite12.provider", "com.gallery.RTranslator.lite13.provider"), encoding="utf-8")

for p in (ROOT / "app/src/main/java").rglob("*.java"):
    s = p.read_text(encoding="utf-8")
    if "RTranslatorLite12" in s:
        p.write_text(s.replace("RTranslatorLite12", "RTranslatorLite13"), encoding="utf-8")

print("Lite13 GGUF loader patch complete:")
print(" - validates GGUF magic before JNI")
print(" - retains Lite11 OCR/Documents and Lite12 runtime fixes")
print(" - applicationId nie.translator.rtranslator.lite13")
