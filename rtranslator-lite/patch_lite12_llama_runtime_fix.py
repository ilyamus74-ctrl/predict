#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "RTranslator")
engine = ROOT / "app/src/main/java/nie/translator/rtranslator/voice_translation/neural_networks/translation/LlamaCppTranslationEngine.java"
manager = ROOT / "app/src/main/java/nie/translator/rtranslator/settings/ModelManagerFragment.java"
build = ROOT / "app/build.gradle"
provider_files = [
    ROOT / "app/src/main/AndroidManifest.xml",
    ROOT / "app/src/main/java/nie/translator/rtranslator/access/UserDataFragment.java",
    ROOT / "app/src/main/java/nie/translator/rtranslator/settings/SettingsFragment.java",
]
for p in [engine, manager, build, *provider_files]:
    if not p.exists():
        raise SystemExit(f"Missing required file: {p}")

# ---------------------------------------------------------------------------
# GGUF loader: report the real failure and always clean a partial native load.
# ---------------------------------------------------------------------------
e = engine.read_text(encoding="utf-8")
field = '    private File modelFile;\n'
if field not in e:
    raise SystemExit("Lite11 modelFile field missing")
e = e.replace(field, field + '    private static volatile String lastLoadError = "";\n', 1)

old = '''    public void load() throws Exception {\n        synchronized (lock) {\n            modelFile = findModelFile(mode);\n            if (modelFile == null) {\n                throw new IllegalStateException("GGUF model not found in " + getModelDirectory(mode).getAbsolutePath());\n            }\n            Log.i(TAG, "Loading GGUF model: " + modelFile.getAbsolutePath());\n            bridge = TranslationBridge.create(global);\n            bridge.loadModel(modelFile.getAbsolutePath());\n            Log.i(TAG, "GGUF model ready: " + modelFile.getName());\n        }\n    }\n'''
new = '''    public void load() throws Exception {\n        synchronized (lock) {\n            lastLoadError = "";\n            modelFile = findModelFile(mode);\n            if (modelFile == null) {\n                lastLoadError = "GGUF file not found";\n                throw new IllegalStateException("GGUF model not found in " + getModelDirectory(mode).getAbsolutePath());\n            }\n            if (!modelFile.isFile() || !modelFile.canRead()) {\n                lastLoadError = "GGUF file is not readable";\n                throw new IllegalStateException("Cannot read GGUF: " + modelFile.getAbsolutePath());\n            }\n            Log.i(TAG, "Loading GGUF model: " + modelFile.getAbsolutePath() + " (" + modelFile.length() + " bytes)");\n            try {\n                bridge = TranslationBridge.create(global);\n                bridge.loadModel(modelFile.getAbsolutePath());\n                Log.i(TAG, "GGUF model ready: " + modelFile.getName());\n            } catch (Throwable t) {\n                String message = t.getMessage();\n                lastLoadError = t.getClass().getSimpleName() + (message == null || message.trim().isEmpty() ? "" : ": " + message);\n                Log.e(TAG, "GGUF load failed: " + lastLoadError + " | " + modelFile.getAbsolutePath(), t);\n                if (bridge != null) {\n                    try {\n                        bridge.unload();\n                    } catch (Throwable cleanup) {\n                        Log.w(TAG, "Partial GGUF cleanup failed", cleanup);\n                    }\n                    bridge = null;\n                }\n                if (t instanceof Exception) throw (Exception) t;\n                throw new RuntimeException(t);\n            }\n        }\n    }\n\n    public static String getLastLoadError() {\n        return lastLoadError == null ? "" : lastLoadError;\n    }\n'''
if old not in e:
    raise SystemExit("Lite11 GGUF load method not found")
e = e.replace(old, new, 1)
engine.write_text(e, encoding="utf-8")

# ---------------------------------------------------------------------------
# Model Manager: surface native failure instead of generic toast.
# ---------------------------------------------------------------------------
m = manager.read_text(encoding="utf-8")
old = '                Toast.makeText(activity, "Model failed to load. Restoring previous translator...", Toast.LENGTH_LONG).show();\n'
new = '''                String detail = LlamaCppTranslationEngine.getLastLoadError();\n                String message = "Model failed to load" + (detail.isEmpty() ? "" : ": " + detail) + ". Restoring previous translator...";\n                Toast.makeText(activity, message, Toast.LENGTH_LONG).show();\n'''
if old not in m:
    raise SystemExit("Lite11 generic model failure toast missing")
m = m.replace(old, new, 1)
manager.write_text(m, encoding="utf-8")

# ---------------------------------------------------------------------------
# Separate Lite12 install identity. Lite11 remains installed/testable.
# ---------------------------------------------------------------------------
b = build.read_text(encoding="utf-8")
if 'applicationId "nie.translator.rtranslator.lite11"' not in b:
    raise SystemExit("Lite11 applicationId missing")
b = b.replace('applicationId "nie.translator.rtranslator.lite11"', 'applicationId "nie.translator.rtranslator.lite12"', 1)
b = b.replace('versionCode 30013', 'versionCode 30014', 1)
b = b.replace("versionName '3.0.0-alpha3-lite11'", "versionName '3.0.0-alpha3-lite12'", 1)
build.write_text(b, encoding="utf-8")

changed_labels = 0
for strings_xml in (ROOT / "app/src/main/res").glob("values*/strings.xml"):
    s = strings_xml.read_text(encoding="utf-8")
    if "RTranslator Lite11" in s:
        strings_xml.write_text(s.replace("RTranslator Lite11", "RTranslator Lite12"), encoding="utf-8")
        changed_labels += 1
if changed_labels == 0:
    raise SystemExit("Lite11 app label missing")

for p in provider_files:
    s = p.read_text(encoding="utf-8")
    if "com.gallery.RTranslator.lite11.provider" not in s:
        raise SystemExit(f"Lite11 provider missing in {p}")
    p.write_text(s.replace("com.gallery.RTranslator.lite11.provider", "com.gallery.RTranslator.lite12.provider"), encoding="utf-8")

for p in (ROOT / "app/src/main/java").rglob("*.java"):
    s = p.read_text(encoding="utf-8")
    if "RTranslatorLite11" in s:
        p.write_text(s.replace("RTranslatorLite11", "RTranslatorLite12"), encoding="utf-8")

print("Lite12 GGUF runtime patch complete:")
print(" - partial/failed GGUF loads are explicitly cleaned")
print(" - real llama.cpp load failure is shown in Model Manager")
print(" - Lite11 OCR/Documents retained unchanged")
print(" - applicationId nie.translator.rtranslator.lite12")
