# Pixel 8 RU-DE speech recognition diagnostic

This is a diagnostic only: it does not translate, record audio to a file, or upload audio to the app developer.

Build locally (JDK 17, Android SDK 35 and Gradle 8.7):

    gradle :app:assembleDebug

GitHub Actions creates an APK under the artifact **Pixel8-RU-DE-ASR-Diagnostic-APK**.

On a Pixel 8 / Android 17:
1. Install the debug APK and grant microphone permission.
2. Select on-device or default SpeechRecognizer and auto RU-DE.
3. Run support check; try Russian then German phrases alternately.
4. Export a TXT event log and compare with airplane mode.

Language switch support is a capability of the installed speech recognition service, not guaranteed by Android version alone. Recognition APIs do not guarantee uninterrupted two-speaker transcription.
