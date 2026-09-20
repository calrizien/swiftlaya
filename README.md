# SwiftLaya

A **Swift 6.4 inference-runtime port of [Laya](https://github.com/NandhaKishorM/laya)**: typed `choice`, ordinal `score`, and Boolean-probability `noul` decisions. This fork preserves the Python reference, training notebooks, and Apache-2.0 license.

**Status: one English checkpoint passes native parity, but this is not yet a production replacement.** On macOS 27 with Swift 6.4, the pinned English checkpoint passed full-graph export, CPU-only Core ML verification, and nine Python-to-Swift fixtures. Tvaily integration, held-out accuracy, performance, iOS/visionOS execution, multilingual checkpoints, and typed-decisions checkpoints remain unvalidated. Converted weights are not included. Training has not been rewritten in Swift.

## What is native

| Product | Responsibility | Platforms |
| --- | --- | --- |
| `Laya` | Ordered questions, structured criteria, sequence construction, batching, calibration/postprocessing, presets, email cleanup, language routing, actor-isolated model cache | Swift-supported platforms, including Linux |
| `LayaCoreML` | Full decision-model inference using local `.mlpackage` / `.mlmodelc` artifacts | macOS 15+, iOS 18+, visionOS 2+ |
| `laya-route` | Inspect routing without downloading or running a model | macOS / Linux |
| `LayaHuggingFace` | Optional exact-checkpoint tokenizer adapter and local checkpoint loader | Separate package in `Integrations/HuggingFace` |
| `laya-predict` | Native prediction and Python-to-Swift parity verification | macOS CLI |

The root package has no third-party dependencies. The optional tokenizer integration pins `swift-transformers` to a reviewed commit. There is no Python bridge, subprocess inference, HTTP service, or implicit model download in the Swift runtime. Python is used **offline** to convert and validate the existing trained graph.

## Build and route — no weights needed

Install Swift 6.4, then from the repository root:

```sh
swift --version
swift build
swift test
swift run laya-route 'Please help with the bill'
swift run laya-route '{"message":"मुझे मदद चाहिए"}'
swift run laya-route --model typed 'Inspect this request'
```

The routing CLI does **not** produce model answers. It emits checkpoint selection and an explanation. Language detection is the upstream-style script/stopword heuristic, not a learned language classifier; short or mixed-language inputs can be ambiguous. Pass a language or checkpoint explicitly when known.

## Prepare a real checkpoint

Use a local Laya checkpoint that includes `model.safetensors`, `rl_agent_config.json`, `encoder/config.json`, and `tokenizer/`.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[coreml]'
python Tools/export_coreml.py \
  --checkpoint /absolute/path/to/laya-checkpoint \
  --output /absolute/path/to/exports/laya-english
```

Run this on macOS. The exporter loads the Apache-2.0 `laya-coreml` export graph, strictly loads the original weights, and checks its encoder, type embedding, Transformer decision head, scorer, and action-head outputs against the upstream model before conversion. It then checks several batch/sequence/option shapes against Core ML before setting `coreMLVerified`. It does not silently substitute an embedding model or synthetic output when conversion fails.

For an explicit Hub download, pass a repository id and `--revision` with an immutable model commit SHA. `--subfolder multilingual` or `--subfolder typed-decisions` selects the bundled variants. Model weights must be pinned independently of this code repository. The validated export accepts one model row, sixteen options, and enumerated sequence lengths up to 512 tokens. The public SDK still accepts question batches; the backend runs them one row at a time and returns only the requested option columns. This correctness-first ceiling is recorded in the artifact.

ModernBERT/mmBERT attention, rotary embeddings, dynamic shapes, and tokenizer behavior are conversion risks. A successful source build is not evidence that these operations convert correctly. See [the porting gates](docs/PORTING.md).

## Verify native behavior, then predict

```sh
swift build --package-path Integrations/HuggingFace
swift run --package-path Integrations/HuggingFace laya-predict verify \
  /absolute/path/to/exports/laya-english \
  /absolute/path/to/exports/laya-english/parity.json
swift run --package-path Integrations/HuggingFace laya-predict predict \
  /absolute/path/to/exports/laya-english Examples/triage.json
```

The verification command checks exact token IDs, option positions, padding, question types, raw logits within the recorded absolute tolerance plus one part per million relative tolerance, and final typed answers. A failure is a release blocker, not a reason to loosen thresholds without investigating. The loader rejects exports marked unverified unless a caller explicitly opts into experimental loading.

The native question JSON uses **ordered arrays**, including choice criteria, rather than unordered objects. See `Examples/triage.json`. This is intentionally not a drop-in JSON clone of the Python SDK: option order is part of the model contract.

## Embed in an application

Add the `Integrations/HuggingFace` package and its `LayaHuggingFace` product (plus `Laya` for direct imports):

```swift
import Foundation
import Laya
import LayaHuggingFace

let local = try await LocalCheckpoint.load(
    from: URL(fileURLWithPath: "/absolute/path/to/exports/laya-english")
)
let result = try await local.agent.predict(
    state: .object(["message": .string("Please refund the duplicate charge.")]),
    questions: Presets.triage()
)
if case .choice(let label) = result.answers["intent"]?.value {
    print(label)
}
```

For multiple checkpoints, provide a local loader to `Router`:

```swift
let root = URL(fileURLWithPath: "/absolute/path/to/exports")
let router = try Router(maxLoaded: 2) { checkpoint in
    let local = try await LocalCheckpoint.load(
        from: root.appendingPathComponent(checkpoint.rawValue)
    )
    return local.agent
}
try await router.preload([.english, .multilingual])
let result = try await router.predict(
    state: .string("Please help with the bill"),
    questions: Presets.triage()
)
```

Arrange exported directories as `english`, `multilingual`, and optionally `typed-decisions` for that example. `preload` avoids language-switch reloads, but resident models consume memory. Cache limits constrain cached agents, not memory retained by in-flight calls or externally held agents. Do not preload everything blindly in a memory-constrained app.

`LocalCheckpoint` defaults to CPU-only execution for initial parity. Applications may explicitly select CPU/GPU or all compute units after rerunning parity and measuring real hardware. No Neural Engine placement or latency is promised.

## Compatibility and safety

Choice/score confidence preserves Laya's normalized-entropy definition; it is **not** the top-label probability. Noul confidence is the larger of `P(true)` and `P(false)`. Action probability is produced by the separate learned action head. Proper-score training does not guarantee calibration on your application's distribution. Guardrail/moderation presets are model inputs, not a replacement for application safety policy.

Structured JSON state uses sorted keys and spaced separators for deterministic Swift serialization; upstream Python retains insertion order. Numeric JSON representations can also differ. Pass an already serialized state as `.string(originalJSON)` when exact original bytes matter. Choice options and ordinal levels always retain explicit array order.

The port rejects malformed schemas, nonfinite outputs, dropped option markers, and over-budget headers. It fixes the Python zero-room left-truncation edge case; empty questions return an empty response without inference. `truncatedStateTokens` makes state truncation observable. These are deliberate compatibility differences.

## Validation and upstream

[Validation record](docs/VALIDATION.md) · [remaining work and acceptance gates](docs/PORTING.md) · [upstream README and benchmark claims](docs/UPSTREAM_README.md)

Upstream's published GPU measurements are not SwiftLaya benchmarks. The original Python implementation remains in `laya/`; it is the comparison reference, not a runtime dependency of the Swift products. Existing Python CI/release workflows are left unchanged. Do not publish upstream's PyPI package from this fork while establishing a separate Swift release process.
