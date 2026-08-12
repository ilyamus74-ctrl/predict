#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "RTranslator")

df2_path = ROOT / "app/src/main/java/nie/translator/rtranslator/access/DownloadFragment2.java"
build_gradle = ROOT / "app/build.gradle"
provider_files = [
    ROOT / "app/src/main/AndroidManifest.xml",
    ROOT / "app/src/main/java/nie/translator/rtranslator/access/UserDataFragment.java",
    ROOT / "app/src/main/java/nie/translator/rtranslator/settings/SettingsFragment.java",
]

for p in [df2_path, build_gradle, *provider_files]:
    if not p.exists():
        raise SystemExit(f"Required file missing: {p}")

# IMPORTANT: v3.00 AccessActivity actually launches DownloadFragment2, not DownloadFragment.
# Replace the active downloader's model list with Whisper-only files.
df2 = df2_path.read_text(encoding="utf-8")

whisper_infos = '''DOWNLOAD_INFOS = new DownloadInfo[]{
                new DownloadInfo(
                        "Whisper_cache_initializer.onnx",
                        baseUrl + "Whisper_cache_initializer.onnx",
                        downloadFolder,
                        14000,
                        true
                ),
                new DownloadInfo(
                        "Whisper_cache_initializer_batch.onnx",
                        baseUrl + "Whisper_cache_initializer_batch.onnx",
                        downloadFolder,
                        14000,
                        true
                ),
                new DownloadInfo(
                        "Whisper_decoder.onnx",
                        baseUrl + "Whisper_decoder.onnx",
                        downloadFolder,
                        173000,
                        true
                ),
                new DownloadInfo(
                        "Whisper_detokenizer.onnx",
                        baseUrl + "Whisper_detokenizer.onnx",
                        downloadFolder,
                        461,
                        true
                ),
                new DownloadInfo(
                        "Whisper_encoder.onnx",
                        baseUrl + "Whisper_encoder.onnx",
                        downloadFolder,
                        88000,
                        true
                ),
                new DownloadInfo(
                        "Whisper_initializer.onnx",
                        baseUrl + "Whisper_initializer.onnx",
                        downloadFolder,
                        69,
                        true
                ),
        };'''

df2, n = re.subn(
    r'DOWNLOAD_INFOS\s*=\s*new DownloadInfo\[\]\{.*?\n\s*\};',
    whisper_infos,
    df2,
    count=1,
    flags=re.S,
)
if n != 1:
    raise SystemExit(f"Could not replace active DownloadFragment2 DOWNLOAD_INFOS block: {n}")

old_progress = 'progressNumbersText.setText(decimalFormat.format(downloadedGb)+" / "+decimalFormat.format(totalSize)+" GB");'
new_progress = 'progressNumbersText.setText("LITE6 | ACTIVE Downloader2 | Whisper only | "+decimalFormat.format(downloadedGb)+" / "+decimalFormat.format(totalSize)+" GB");'
if old_progress not in df2:
    raise SystemExit("Could not locate DownloadFragment2 progress text")
df2 = df2.replace(old_progress, new_progress, 1)

df2_path.write_text(df2, encoding="utf-8")

# Make launcher identity unambiguous.
changed_labels = 0
for strings_xml in (ROOT / "app/src/main/res").glob("values*/strings.xml"):
    text = strings_xml.read_text(encoding="utf-8")
    new_text, count = re.subn(
        r'(<string\s+name="app_name"[^>]*>).*?(</string>)',
        r'\1RTranslator Lite6\2',
        text,
        count=1,
        flags=re.S,
    )
    if count:
        strings_xml.write_text(new_text, encoding="utf-8")
        changed_labels += 1
if changed_labels == 0:
    raise SystemExit("No app_name resources updated")

# Move from the Lite5 identity produced by the preceding patch.
bg = build_gradle.read_text(encoding="utf-8")
if 'applicationId "nie.translator.rtranslator.lite5"' not in bg:
    raise SystemExit("Expected Lite5 applicationId before Lite6 patch")
bg = bg.replace('applicationId "nie.translator.rtranslator.lite5"', 'applicationId "nie.translator.rtranslator.lite6"', 1)
bg = re.sub(r'versionCode\s+\d+', 'versionCode 30008', bg, count=1)
bg = re.sub(r"versionName\s+'[^']+'", "versionName '3.0.0-alpha3-lite6'", bg, count=1)
build_gradle.write_text(bg, encoding="utf-8")

old_authority = 'com.gallery.RTranslator.lite5.provider'
new_authority = 'com.gallery.RTranslator.lite6.provider'
for path in provider_files:
    value = path.read_text(encoding="utf-8")
    if old_authority not in value:
        raise SystemExit(f"Expected Lite5 FileProvider authority not found in {path}")
    path.write_text(value.replace(old_authority, new_authority), encoding="utf-8")

print("Lite6 active downloader patch complete:")
print(" - DownloadFragment2 now has exactly 6 Whisper models")
print(" - all NLLB entries removed from active downloader")
print(" - visible ACTIVE Downloader2 marker added")
print(" - applicationId nie.translator.rtranslator.lite6")
print(" - version 3.0.0-alpha3-lite6")
print(f" - updated app_name in {changed_labels} resource file(s)")
