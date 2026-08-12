#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "RTranslator")
activity_path = ROOT / "app/src/main/java/nie/translator/rtranslator/voice_translation/VoiceTranslationActivity.java"
df2_path = ROOT / "app/src/main/java/nie/translator/rtranslator/access/DownloadFragment2.java"
build_gradle = ROOT / "app/build.gradle"
provider_files = [
    ROOT / "app/src/main/AndroidManifest.xml",
    ROOT / "app/src/main/java/nie/translator/rtranslator/access/UserDataFragment.java",
    ROOT / "app/src/main/java/nie/translator/rtranslator/settings/SettingsFragment.java",
]

for p in [activity_path, df2_path, build_gradle, *provider_files]:
    if not p.exists():
        raise SystemExit(f"Required file missing: {p}")

activity = activity_path.read_text(encoding="utf-8")

# 1) Dedicated runtime permission request for WalkieTalkie microphone service.
needle = "private static final int REQUEST_CODE_REQUIRED_PERMISSIONS = 2;"
replacement = needle + "\n    private static final int REQUEST_CODE_WALKIE_TALKIE_MIC_PERMISSION = 6;"
if needle not in activity:
    raise SystemExit("Could not find Activity permission request constant")
activity = activity.replace(needle, replacement, 1)

# 2) Never restore WalkieTalkie automatically after a process crash/restart.
old_onstart = '''SharedPreferences sharedPreferences = this.getSharedPreferences("default", Context.MODE_PRIVATE);\n        setFragment(sharedPreferences.getInt("fragment", DEFAULT_FRAGMENT));'''
new_onstart = '''SharedPreferences sharedPreferences = this.getSharedPreferences("default", Context.MODE_PRIVATE);\n        int savedFragment = sharedPreferences.getInt("fragment", DEFAULT_FRAGMENT);\n        // Lite7 safety: a crashed WalkieTalkie foreground service must never create a startup crash-loop.\n        if (savedFragment == WALKIE_TALKIE_FRAGMENT) {\n            sharedPreferences.edit().putInt("fragment", DEFAULT_FRAGMENT).apply();\n            savedFragment = DEFAULT_FRAGMENT;\n        }\n        setFragment(savedFragment);'''
if old_onstart not in activity:
    raise SystemExit("Could not find VoiceTranslationActivity.onStart fragment restore")
activity = activity.replace(old_onstart, new_onstart, 1)

# 3) Request RECORD_AUDIO before the microphone foreground service can be created.
old_walkie_case = '''case WALKIE_TALKIE_FRAGMENT: {\n                // possible setting of the fragment\n                if (getCurrentFragment() != WALKIE_TALKIE_FRAGMENT) {'''
new_walkie_case = '''case WALKIE_TALKIE_FRAGMENT: {\n                // Android 14+ checks RECORD_AUDIO when a microphone foreground service is created.\n                // Request it while this Activity is visible, before creating/starting WalkieTalkie.\n                if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {\n                    requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, REQUEST_CODE_WALKIE_TALKIE_MIC_PERMISSION);\n                    return;\n                }\n                // possible setting of the fragment\n                if (getCurrentFragment() != WALKIE_TALKIE_FRAGMENT) {'''
if old_walkie_case not in activity:
    raise SystemExit("Could not find WALKIE_TALKIE_FRAGMENT case")
activity = activity.replace(old_walkie_case, new_walkie_case, 1)

# 4) Do not persist WalkieTalkie as the fragment to restore after process death.
old_save = '''SharedPreferences.Editor editor = sharedPreferences.edit();\n                editor.putInt("fragment", getCurrentFragment());\n                editor.apply();'''
new_save = '''SharedPreferences.Editor editor = sharedPreferences.edit();\n                int fragmentToPersist = getCurrentFragment();\n                if (fragmentToPersist == WALKIE_TALKIE_FRAGMENT) {\n                    fragmentToPersist = DEFAULT_FRAGMENT;\n                }\n                editor.putInt("fragment", fragmentToPersist);\n                editor.apply();'''
if old_save not in activity:
    raise SystemExit("Could not find saveFragment body")
activity = activity.replace(old_save, new_save, 1)

# 5) Handle the new microphone permission result before the existing Bluetooth permission handler.
old_perm = '''super.onRequestPermissionsResult(requestCode, permissions, grantResults);\n\n        if (requestCode != REQUEST_CODE_REQUIRED_PERMISSIONS) {\n            return;\n        }'''
new_perm = '''super.onRequestPermissionsResult(requestCode, permissions, grantResults);\n\n        if (requestCode == REQUEST_CODE_WALKIE_TALKIE_MIC_PERMISSION) {\n            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {\n                setFragment(WALKIE_TALKIE_FRAGMENT);\n            } else {\n                getSharedPreferences("default", Context.MODE_PRIVATE).edit()\n                        .putInt("fragment", DEFAULT_FRAGMENT).apply();\n                Toast.makeText(global, R.string.error_missing_mic_permissions, Toast.LENGTH_LONG).show();\n                setFragment(DEFAULT_FRAGMENT);\n            }\n            return;\n        }\n\n        if (requestCode != REQUEST_CODE_REQUIRED_PERMISSIONS) {\n            return;\n        }'''
if old_perm not in activity:
    raise SystemExit("Could not find onRequestPermissionsResult entry")
activity = activity.replace(old_perm, new_perm, 1)

# 6) Modern foreground-service launch: always supply a notification, even when POST_NOTIFICATIONS is denied.
old_start_service = '''if(NotificationManagerCompat.from(VoiceTranslationActivity.this).areNotificationsEnabled()) {\n            intent.putExtra("notification", notification);\n        }else{\n            //Toast.makeText(VoiceTranslationActivity.this, getResources().getString(R.string.toast_missing_notification_permission), Toast.LENGTH_LONG).show();\n        }\n        startService(intent);\n        responseListener.onSuccess();'''
new_start_service = '''// A foreground service always needs its Notification object. POST_NOTIFICATIONS is not a prerequisite\n        // for starting the service, so do not omit the notification when that permission is denied.\n        intent.putExtra("notification", notification);\n        try {\n            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {\n                startForegroundService(intent);\n            } else {\n                startService(intent);\n            }\n            responseListener.onSuccess();\n        } catch (SecurityException e) {\n            android.util.Log.e("RTranslatorLite7", "Cannot start microphone foreground service", e);\n            getSharedPreferences("default", Context.MODE_PRIVATE).edit()\n                    .putInt("fragment", DEFAULT_FRAGMENT).apply();\n            Toast.makeText(this, R.string.error_missing_mic_permissions, Toast.LENGTH_LONG).show();\n            setFragment(DEFAULT_FRAGMENT);\n        } catch (RuntimeException e) {\n            android.util.Log.e("RTranslatorLite7", "WalkieTalkie service startup failed", e);\n            getSharedPreferences("default", Context.MODE_PRIVATE).edit()\n                    .putInt("fragment", DEFAULT_FRAGMENT).apply();\n            Toast.makeText(this, "WalkieTalkie service failed to start", Toast.LENGTH_LONG).show();\n            setFragment(DEFAULT_FRAGMENT);\n        }'''
# There are two identical blocks (Conversation and WalkieTalkie); replace only the one after secondLanguage.
walkie_method_start = activity.find("public void startWalkieTalkieService")
if walkie_method_start < 0:
    raise SystemExit("Could not find startWalkieTalkieService")
walkie_tail = activity[walkie_method_start:]
if old_start_service not in walkie_tail:
    raise SystemExit("Could not find legacy WalkieTalkie service launch block")
walkie_tail = walkie_tail.replace(old_start_service, new_start_service, 1)
activity = activity[:walkie_method_start] + walkie_tail

activity_path.write_text(activity, encoding="utf-8")

# 7) Visible Lite7 identity and downloader marker.
df2 = df2_path.read_text(encoding="utf-8")
if "LITE6 | ACTIVE Downloader2 | Whisper only" not in df2:
    raise SystemExit("Expected Lite6 active downloader marker not found")
df2_path.write_text(df2.replace(
    "LITE6 | ACTIVE Downloader2 | Whisper only",
    "LITE7 | ACTIVE Downloader2 | Whisper only",
    1,
), encoding="utf-8")

changed_labels = 0
for strings_xml in (ROOT / "app/src/main/res").glob("values*/strings.xml"):
    text = strings_xml.read_text(encoding="utf-8")
    if "RTranslator Lite6" in text:
        strings_xml.write_text(text.replace("RTranslator Lite6", "RTranslator Lite7"), encoding="utf-8")
        changed_labels += 1
if changed_labels == 0:
    raise SystemExit("No RTranslator Lite6 app_name resource found for Lite7")

bg = build_gradle.read_text(encoding="utf-8")
required_gradle_strings = [
    'applicationId "nie.translator.rtranslator.lite6"',
    'versionCode 30008',
    "versionName '3.0.0-alpha3-lite6'",
]
for value in required_gradle_strings:
    if value not in bg:
        raise SystemExit(f"Expected Lite6 Gradle value not found: {value}")
bg = bg.replace('applicationId "nie.translator.rtranslator.lite6"', 'applicationId "nie.translator.rtranslator.lite7"', 1)
bg = bg.replace('versionCode 30008', 'versionCode 30009', 1)
bg = bg.replace("versionName '3.0.0-alpha3-lite6'", "versionName '3.0.0-alpha3-lite7'", 1)
build_gradle.write_text(bg, encoding="utf-8")

old_authority = 'com.gallery.RTranslator.lite6.provider'
new_authority = 'com.gallery.RTranslator.lite7.provider'
for path in provider_files:
    value = path.read_text(encoding="utf-8")
    if old_authority not in value:
        raise SystemExit(f"Expected Lite6 FileProvider authority not found in {path}")
    path.write_text(value.replace(old_authority, new_authority), encoding="utf-8")

print("Lite7 WalkieTalkie safety patch complete:")
print(" - RECORD_AUDIO requested before WalkieTalkie microphone FGS")
print(" - WalkieTalkie is never restored after process death")
print(" - WalkieTalkie is not persisted as startup fragment")
print(" - FGS always receives notification even if POST_NOTIFICATIONS is denied")
print(" - startForegroundService used on Android O+")
print(" - SecurityException/RuntimeException falls back to text mode")
print(" - applicationId nie.translator.rtranslator.lite7")
print(" - version 3.0.0-alpha3-lite7")
