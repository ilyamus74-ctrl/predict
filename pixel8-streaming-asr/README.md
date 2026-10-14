# Pixel 8: REAL streaming Nemotron 3.5 ASR (RU/DE)

Separate **diagnostic APK**, not a patch to RTranslator and not a translator.

- **Runtime:** official \`sherpa-onnx v1.13.8\` Android AAR, arm64, CPU.
- **Models:** NVIDIA Nemotron 3.5 Streaming 0.6B INT8 **160 ms** and **560 ms** (optional separate downloads).
- **Network:** only first model download; ASR is fully offline afterwards.
- **Audio path:** AudioRecord 16kHz/mono/PCM16; independent capture and inference threads; bounded queue; incremental transducer decoding with model state; no reprocessing full recordings.
- **Language prompt:** auto / ru / de, selected at stream creation, not mid-utterance.
- **Output:** PARTIAL displayed while talking; FINAL on model endpoint; decode-only RTF; unprocessed queue duration; packet drops; initial partial latency.
- **Scope:** no translation or TTS yet, no accurate per-word glass-to-glass latency measurement; RU/DE display label is a heuristic from text script, **not native model language metadata**.

## Install
1. Go to [Actions](https://github.com/ilyamus74-ctrl/predict/actions/workflows/build-pixel8-streaming-asr.yml) and open the successful workflow.
2. Download \`Pixel8-Nemotron35-True-Streaming-v01-APK\`, unzip it, and install \`app-debug.apk\` on the Pixel 8.
3. Select \`160 ms\`, tap **Скачать модель**, keep app visible during download (~500 MB compressed; allow at least 1.5 GB of free storage).
4. Grant RECORD_AUDIO; select AUTO, RU or DE; tap **Запустить streaming**.
5. Talk continuously 20-60 seconds in RU and DE; monitor the live text, RTF, queue and dropped audio chunks.
6. Optional: download 560 ms model and test same phrases.
7. Stop capture. The model stays on device offline.

## Technical distinction
AUDIO is continuously recorded **while decoding proceeds on another worker**; the same online stream is reused for new samples. Silences cause **endpointing/reset**, not the initial invocation of recognizer. This is true streaming ASR and not a sliding-window batch simulation.

## Model source
- Official model downloads: https://k2-fsa.github.io/sherpa/onnx/nemo/nemotron-streaming.html
- sherpa-onnx Android library: https://github.com/k2-fsa/sherpa-onnx/releases/tag/v1.13.8
- Model weights subject to NVIDIA OpenMDW-1.1; runtime Apache-2.0.

## Accuracy and speed
**Not yet benchmarked on Pixel 8 / Tensor G3.** RTF target < 1; lower is better. Queue metric represents *pending audio* in the app, not end-to-end latency. If RTF is above 1 for extended speech, the backlog will grow until drops occur.

## Building
Run the GitHub Actions workflow. It pulls the official AAR as an uncommitted build-time dependency. Android Studio users need to place \`sherpa-onnx-1.13.8.aar\` at \`app/libs/sherpa-onnx.aar\`.
