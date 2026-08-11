# RTranslator 3 Lite builder

Temporary CI builder for an 8 GB Android target. It does **not** modify the repository's `main` branch.

The workflow clones `niedev/RTranslator` branch `v3.00`, then applies `patch_low_memory.py` and builds a side-by-side APK (`nie.translator.rtranslator.lite`).

Changes in this first build:

- keeps the RTranslator 3 alpha UI, ASR, Conversation, WalkieTalkie and TTS code;
- MADLAD checks for `Madlad/Int4_16/*_4bit.onnx` first;
- falls back to the alpha3 `Madlad/Int8WO/*_8bit.onnx` layout if INT4 files are absent;
- disables ONNX Runtime CPU arena/memory-pattern pooling for MADLAD to reduce peak native RAM;
- leaves beam size default at upstream value `1`;
- changes the app id so it can be installed next to stock RTranslator.

Expected on-device model layout:

```text
/storage/emulated/0/models/Translation/Madlad/
├── spiece.model
├── madlad_embed_8bit.onnx
├── Int4_16/
│   ├── madlad_encoder_4bit.onnx
│   ├── madlad_decoder_4bit.onnx
│   └── madlad_cache_initializer_4bit.onnx
└── Int8WO/                 # fallback only
    ├── madlad_encoder_8bit.onnx
    ├── madlad_decoder_8bit.onnx
    └── madlad_cache_initializer_8bit.onnx
```

This is an experimental alpha build. Compile success verifies the Android integration, not translation quality or device RAM behavior. Those require installation and an on-device test.
