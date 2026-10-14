# Pixel 8 RU-DE speech recognition diagnostic

This is a diagnostic only: it does not translate, record audio to a file, or upload audio to the app developer.

Build locally (JDK 17, Android SDK 35 and Gradle 8.7):

    gradle :app:assembleDebug

GitHub Actions creates an APK under the artifact **Pixel8-RU-DE-ASR-Quick-Balanced-APK**.

On a Pixel 8 / Android 17:
1. Install the debug APK and grant microphone permission.
2. Select on-device or default SpeechRecognizer and auto RU-DE.
3. Run support check; try Russian then German phrases alternately.
4. Export a TXT event log and compare with airplane mode.

Language switch support is a capability of the installed speech recognition service, not guaranteed by Android version alone. Recognition APIs do not guarantee uninterrupted two-speaker transcription.

## Version 0.2

- Select QUICK_RESPONSE or BALANCED with the dedicated sensitivity spinner.
- The UI separately shows language detection, model switching outcome and downloaded language availability.
- Download buttons use ModelDownloadListener and support refresh after onSuccess.
- Exports SUMMARY counts for switch succeeded/failed/skipped/no-attempt.
- Changed applicationId so this build installs beside v0.1; old test remains installed.

If a model is DOWNLOADABLE instead of INSTALLED, request its download and wait for success, then recheck. Comparing modes requires running each mode with the same RU and DE sentences, exporting separate TXT logs. The switch event 0 is NOT_ATTEMPTED, not success.
