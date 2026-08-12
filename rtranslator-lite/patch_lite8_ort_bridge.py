#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "RTranslator")
build_gradle = ROOT / "app/build.gradle"

if not build_gradle.exists():
    raise SystemExit(f"Required file missing: {build_gradle}")

bg = build_gradle.read_text(encoding="utf-8")

# patch_lite8_sherpa_asr.py originally tried to consume ORT 1.27.1 from Maven.
# ORT 1.27.1 is not published as onnxruntime-android in Maven Central yet.
#
# Runtime arrangement used by Lite8:
#   sherpa AAR 1.13.5 -> libonnxruntime.so 1.27.1 + sherpa JNI/C API
#   ORT Android AAR 1.24.3 -> Java classes + libonnxruntime4j_jni.so only
#                              (its older libonnxruntime.so is stripped by CI)
#
# ONNX Runtime's C API is backward-compatible, so an older 1.24.3 Java JNI
# client can request its API from the newer 1.27.1 core. This avoids the unsafe
# opposite direction (new sherpa JNI against an older ORT core).
old_ort = "implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.27.1'"
new_ort = "implementation files('libs/onnxruntime-android-1.24.3-no-core-arm64.aar')"
if old_ort not in bg:
    raise SystemExit("Expected Lite8 Maven ORT 1.27.1 dependency not found")
bg = bg.replace(old_ort, new_ort, 1)

old_sherpa = "implementation files('libs/sherpa-onnx-1.13.5-no-ort-arm64.aar')"
new_sherpa = "implementation files('libs/sherpa-onnx-1.13.5-core-arm64.aar')"
if old_sherpa not in bg:
    raise SystemExit("Expected Lite8 stripped sherpa dependency not found")
bg = bg.replace(old_sherpa, new_sherpa, 1)

build_gradle.write_text(bg, encoding="utf-8")

print("Lite8 ORT bridge patch complete:")
print(" - sherpa AAR supplies libonnxruntime.so 1.27.1")
print(" - ORT Android 1.24.3 local AAR supplies Java + onnxruntime4j JNI")
print(" - ORT 1.24.3 core is stripped by CI")
