# Initial validation record

## Actually executed during authoring

- Compiler: Swift 6.2.1, x86_64 Linux. The requested Swift 6.4 toolchain and Apple SDKs were not installed here.
- A separate local compatibility copy changed **only the first manifest line** from tools-version 6.4 to 6.2. Source files remained the same. Delivered manifests require 6.4 and Swift 6 language mode.
- `swift test -j 4`: **47 Swift Testing tests passed**, including parameterized cases. The portable library and `laya-route` executable compiled. Root tests require no third-party packages, weights, or network.
- Coverage: ordered criteria, structured JSON and false/zero values, all three outputs, temperature overrides, entropy vs noul confidence, sequence packing, mask removal, header rejection, zero-room truncation, padding, nonfinite output rejection, routing precedence, script/language detection, concurrent single-flight loads, LRU, attach/preload union, unload races, cancellation, presets, email cleanup, export validation, and parity-verifier failure paths.
- `swift package dump-package` on the tokenizer integration's analogous 6.2 compatibility manifest succeeded; this validates the manifest, **not** dependency resolution or integration compilation.
- Python `py_compile` and `--help` succeeded for `Tools/export_coreml.py`. All **4 exporter utility tests passed**, including a check that staging leaves source tokenizer metadata unchanged; these tests use no model dependencies.

## Not established

- Swift 6.4 compilation or CI success until the corresponding workflow actually completes.
- Type-checking or running the Apple-only Core ML implementation; Linux conditionally excludes it.
- Resolving/building the optional tokenizer dependency in the network-restricted authoring environment.
- Converting an actual Laya checkpoint, native tokenizer/logit parity, checkpoint accuracy, GPU/Neural Engine execution, memory usage, or latency.
- iOS/visionOS application integration or a Swift training implementation.

Synthetic tokenizers/backends are confined to tests. Passing those tests is SDK contract evidence, not evidence of neural-model inference or accuracy. Follow `PORTING.md` before changing the status to production-ready.
