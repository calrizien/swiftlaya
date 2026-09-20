# Initial validation record

## Mac continuation — 2026-09-20

One pinned English checkpoint now passes the correctness gate on Apple silicon. This validates the conversion and native inference path; it does not establish production readiness.

- Host: Apple M1 Max, arm64, macOS 27.0 (26A428), Xcode 27.0 (27A266a), Swift 6.4.
- Checkpoint: `convaiinnovations/laya` at immutable revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`; source `model.safetensors` SHA-256 `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`.
- Conversion: Python 3.11, torch 2.7.0, NumPy 2.1.3, coremltools 9.0, and `laya-coreml` 0.1.0. The ignored local export was 1.6 GB; its Core ML weight file SHA-256 was `13e70e9a580c0960a4a53088ead23b57c091db122f028b8fe9f213ee956e5eea`.
- Export verification: all nine fixtures matched the upstream PyTorch model and CPU-only Core ML with absolute tolerance 0.001 plus relative tolerance 0.000001. The artifact records `coreMLVerified: true`.
- Native verification: `laya-predict verify` passed all nine fixtures, including exact tokenizer inputs and final typed answers.
- Native smoke test: `laya-predict predict` completed `Examples/triage.json` with the local Core ML artifact.
- Swift checks: root `swift build` passed, all 47 root tests passed, and `swift build --package-path Integrations/HuggingFace` passed. The resolved tokenizer dependencies are pinned in `Integrations/HuggingFace/Package.resolved`.
- CI: the PR's Swift 6.4 macOS job passed. Its Ubuntu job failed while setting up Swift, before repository code ran.

The validated Core ML artifact uses one model row, sixteen option columns, and enumerated sequence lengths of 16, 32, 64, 96, 128, 192, 256, 384, and 512. `CoreMLBackend` pads to the next supported length and makes one Core ML call per question. This is a deliberate correctness-first limit; measure before adding fixed-size batch exports.

## Actually executed during authoring

- Compiler: Swift 6.2.1, x86_64 Linux. The requested Swift 6.4 toolchain and Apple SDKs were not installed here.
- A separate local compatibility copy changed **only the first manifest line** from tools-version 6.4 to 6.2. Source files remained the same. Delivered manifests require 6.4 and Swift 6 language mode.
- `swift test -j 4`: **47 Swift Testing tests passed**, including parameterized cases. The portable library and `laya-route` executable compiled. Root tests require no third-party packages, weights, or network.
- Coverage: ordered criteria, structured JSON and false/zero values, all three outputs, temperature overrides, entropy vs noul confidence, sequence packing, mask removal, header rejection, zero-room truncation, padding, nonfinite output rejection, routing precedence, script/language detection, concurrent single-flight loads, LRU, attach/preload union, unload races, cancellation, presets, email cleanup, export validation, and parity-verifier failure paths.
- `swift package dump-package` on the tokenizer integration's analogous 6.2 compatibility manifest succeeded; this validates the manifest, **not** dependency resolution or integration compilation.
- Python `py_compile` and `--help` succeeded for `Tools/export_coreml.py`. All **4 exporter utility tests passed**, including a check that staging leaves source tokenizer metadata unchanged; these tests use no model dependencies.

## Not established

The separate [Core AI probe](COREAI_TRIAL.md) now has bounded source-runtime
evidence. It does not extend the Core ML validation claims below.

- Held-out checkpoint accuracy, calibration for Tvaily's labels, or behavioral parity with its current Jev selector.
- GPU or Neural Engine parity, memory use, cold/warm latency, and cancellation responsiveness.
- iOS or visionOS build and runtime execution. The validated artifact uses the iOS 18 Core ML feature level because multiple enumerated input shapes require it.
- Multilingual or typed-decisions checkpoint conversion and tokenizer parity.
- Tvaily application integration, model packaging/download integrity, or a Swift training implementation.

Synthetic tokenizers/backends are confined to tests. Passing those tests is SDK contract evidence, not evidence of neural-model inference or accuracy. Follow `PORTING.md` before changing the status to production-ready.
