# RTranslator 3 Lite builder

Temporary CI builder for an 8 GB Android target. It does **not** modify the repository's `main` branch.

The workflow clones `niedev/RTranslator` branch `v3.00`, applies `patch_low_memory.py`, and builds a side-by-side APK (`nie.translator.rtranslator.lite`).

Current Lite2 changes:

- keeps the RTranslator 3 alpha UI, ASR, Conversation, WalkieTalkie and TTS code;
- makes `MADLAD_CACHE` the first-run/default translation backend;
- uses the actual official `Madlad.zip` INT4 directory, `Madlad/Int4Acc4/`;
- also recognizes the older commented `Int4_16` path and retains `Int8WO` as a compatibility fallback;
- disables ONNX Runtime CPU arena/memory-pattern pooling for MADLAD to reduce peak native RAM;
- leaves beam size at upstream default `1`;
- changes the app id so it can be installed next to stock RTranslator.

Verified official model package layout:

```text
/storage/emulated/0/models/Translation/Madlad/
├── spiece.model
├── madlad_embed_8bit.onnx
└── Int4Acc4/
    ├── madlad_encoder_4bit.onnx
    ├── madlad_decoder_4bit.onnx
    └── madlad_cache_initializer_4bit.onnx
```

The official ZIP contains about 1.70 GB uncompressed model data. The three main encoder/decoder/cache models are INT4; the separated embedding model remains INT8.

This is an experimental alpha build. CI compilation verifies the Android integration. Actual peak RAM, speed and translation quality still require an on-device run.
