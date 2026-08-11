#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "RTranslator")

# 1) Make the launcher/app label unambiguous in every localized strings.xml that defines app_name.
changed_labels = 0
for strings_xml in (ROOT / "app/src/main/res").glob("values*/strings.xml"):
    text = strings_xml.read_text(encoding="utf-8")
    new_text, n = re.subn(
        r'(<string\s+name="app_name"[^>]*>).*?(</string>)',
        r'\1RTranslator Lite5\2',
        text,
        count=1,
        flags=re.S,
    )
    if n:
        strings_xml.write_text(new_text, encoding="utf-8")
        changed_labels += 1

if changed_labels == 0:
    raise SystemExit("No app_name string resource found")

# 2) Put a visible build marker directly into the download progress text.
download_fragment = ROOT / "app/src/main/java/nie/translator/rtranslator/access/DownloadFragment.java"
df = download_fragment.read_text(encoding="utf-8")
old_progress = 'progressNumbersText.setText(decimalFormat.format(downloadedGb)+" / "+decimalFormat.format(totalSize)+" GB");'
new_progress = 'progressNumbersText.setText("LITE5 | Whisper only | "+decimalFormat.format(downloadedGb)+" / "+decimalFormat.format(totalSize)+" GB");'
if old_progress not in df:
    raise SystemExit("Could not locate progressNumbersText line")
df = df.replace(old_progress, new_progress, 1)
download_fragment.write_text(df, encoding="utf-8")

# 3) Give Lite5 a completely separate package so it cannot be confused with stock/Lite3/Lite4.
build_gradle = ROOT / "app/build.gradle"
bg = build_gradle.read_text(encoding="utf-8")
if 'applicationId "nie.translator.rtranslator.lite"' not in bg:
    raise SystemExit("Expected Lite4 applicationId not found")
bg = bg.replace('applicationId "nie.translator.rtranslator.lite"', 'applicationId "nie.translator.rtranslator.lite5"', 1)
bg = re.sub(r'versionCode\s+\d+', 'versionCode 30007', bg, count=1)
bg = re.sub(r"versionName\s+'[^']+'", "versionName '3.0.0-alpha3-lite5'", bg, count=1)
build_gradle.write_text(bg, encoding="utf-8")

# 4) Unique FileProvider authority for the separate Lite5 package.
old_authority = 'com.gallery.RTranslator.lite.provider'
new_authority = 'com.gallery.RTranslator.lite5.provider'
provider_files = [
    ROOT / "app/src/main/AndroidManifest.xml",
    ROOT / "app/src/main/java/nie/translator/rtranslator/access/UserDataFragment.java",
    ROOT / "app/src/main/java/nie/translator/rtranslator/settings/SettingsFragment.java",
]
for path in provider_files:
    value = path.read_text(encoding="utf-8")
    if old_authority not in value:
        raise SystemExit(f"Expected Lite4 FileProvider authority not found in {path}")
    path.write_text(value.replace(old_authority, new_authority), encoding="utf-8")

print(f"Updated app_name in {changed_labels} localized resource file(s)")
print("Added visible LITE5 | Whisper only marker to download progress")
print("applicationId = nie.translator.rtranslator.lite5")
print("versionName = 3.0.0-alpha3-lite5")
print("FileProvider authority = com.gallery.RTranslator.lite5.provider")
