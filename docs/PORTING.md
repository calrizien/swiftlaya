# Porting decisions and completion gates

## Scope

Base: `calrizien/swiftlaya` main at `d113dca2512fb3eaca313534bc54c7162d87c1d4` (upstream Laya 0.3.4). This is an **inference-runtime rewrite**, not a Swift implementation of PyTorch, Hugging Face training, RLCD, or the notebook workflows. The trained architecture executes through Core ML. The current implementation is not a certified replacement until all gates below pass.

Preserve `laya/`, notebooks, original tests, licenses, and the upstream README during migration. They are reference material and independent tooling. Do not delete them merely to improve GitHub's Swift language percentage.

## Why this backend

The checkpoint is not just ModernBERT embeddings. `laya/common.py::DecisionModel.forward` includes:

1. Bidirectional encoder output, then the question-type embedding added to all positions.
2. Optional pre-norm TransformerEncoder layers. Their default feed-forward activation is ReLU; the scorer and action head use GELU. Converting the full graph avoids silently replacing one with the other.
3. Gather at option markers; scalar scorer; inactive option masking to `-1e4`.
4. Action-head features from the **unscaled** option softmax: top probability, top-two margin, normalized entropy, and option count / 255, concatenated with CLS hidden state.
5. Independent action logits. Per-type/per-option temperature calibration belongs in postprocessing, not before the action head.

Replacing this with a pooled embedding + cosine similarity would not be a port. Core ML allows native application execution while keeping these trained operations together. `DecisionBackend` leaves room for an MLX or ONNX implementation later without changing schemas or routing. No unverified architecture is used as a production fallback.

## Native contract

Inputs: `input_ids`, `attention_mask` shaped `[B,L]`; `marker_pos`, `marker_mask` shaped `[B,K]`; `qtype` shaped `[B]`. All external tensors are int32. The exported wrapper converts IDs to long and the marker mask to Boolean internally. K is at least two. All shapes have finite exported upper bounds.

Outputs: `logits [B,K]`, `act_logits [B,A]`, both **unscaled logits**. Swift validates shapes, finiteness, vocabulary bounds, and marker indices. Core ML arrays are read with stride-aware indices. Returning probabilities through this protocol would apply softmax twice and is incorrect.

Option order is explicit. The native schema is an array of questions, with choice criteria an array of `{label, criterion}`. Python-style dictionaries are deliberately not decoded as ordered choices. Score legends retain structured JSON. State serialization and structured criterion key order are deterministic, but are not byte-identical to every insertion-ordered Python object. Raw string state is the exact-format escape hatch.

## Concurrency and lifecycle

`Agent` and `CoreMLBackend` isolate state. Core ML reference objects stay inside their backend actor; no `@unchecked Sendable` is added by this port. A prediction batches all questions once. Cancellation is checked before work and before publishing results; an already submitted Core ML kernel may finish after cancellation.

`Router` deduplicates concurrent loads using a shared task, maintains LRU order, and uses generation tokens so unloading during an in-flight load cannot resurrect a checkpoint. A cancelled waiter does not cancel another caller's shared load. Attaching/preloading reserves the union of requested and attached models. The cache capacity is not a bound on all live process allocations: clients and in-flight calls may retain evicted agents.

## Required gates before calling the rewrite complete

### Gate 1 — Swift 6.4 / Apple SDK

Run `swift --version`, `swift build`, `swift test`, and build `Integrations/HuggingFace` on macOS with Swift 6.4. The new CI workflow targets exactly that compiler on Linux and macOS; it does not edit tools-version to make a green check. Ensure the Core ML conditional branch really compiles. Build an application importing the package for iOS and visionOS before advertising tested support on those platforms. Pin the integration's transitive `Package.resolved` after a successful resolution.

### Gate 2 — Full-graph conversion

Start with the English checkpoint. Run `Tools/export_coreml.py` on macOS in a reproducible virtual environment. Record Python, torch, transformers, coremltools, OS, hardware, immutable model revision, and all source hashes. The script records a subset automatically; retain `pip freeze` alongside results.

Do not weaken masks, remove the action head, change activations, replace attention with a causal implementation, or silently alter rotary/sliding-window behavior to make conversion pass. The exporter reuses the Apache-2.0 `laya-coreml` export graph because the Transformers graph contains unsupported tracing operators. It strictly loads the original state dictionary and compares raw outputs against upstream PyTorch across every fixture before Core ML conversion. The exporter is experimental until a full checkpoint passes both Core ML and native Swift parity.

The exporter first checks eager-vs-traced tensors across multiple shapes, then PyTorch-vs-Core ML tensors. Its `coreMLVerified` marker means those exported test cases passed on CPU. It does not certify native tokenization, task accuracy, arbitrary shapes, or other compute policies.

### Gate 3 — Native parity

Run `laya-predict verify CHECKPOINT_DIR CHECKPOINT_DIR/parity.json`. This checks tokenizer parity before model output. Fix tokenizer normalization, special-token configuration, padding, and option order at the source rather than accepting mismatched tokens. The initial nine fixtures cover all primitives, ragged batching, non-Latin content, mask sanitization, long-state truncation, maximum configured options, and maximum configured batch count. Expand coverage to structured criteria, Unicode normalization, emoji, empty/whitespace states, all calibration buckets, and randomized valid shapes before release.

Repeat for multilingual and typed-decisions checkpoints independently. A tokenizer working for English does not establish Gemma/mmBERT-tokenizer compatibility. Review real model configuration rather than inferring dimensions from a model's name.

### Gate 4 — Application relevance and performance

Only after correctness: measure cold load, warm batch p50/p95, peak resident memory, checkpoint-switch cost, and cancellation responsiveness on target Apple hardware. Measure CPU-only, CPU/GPU, and all-compute policies separately; rerun parity for each. Consider float16/quantization only after a float32 baseline. Export dimensions and option count can materially affect memory. Upstream T4 figures must never be copied into native performance claims.

Use held-out, application-specific labels to assess accuracy, false positives/negatives, calibration, and abstention. Confidence is not an application safety policy. Application integration belongs in a separate PR and must preserve existing safety and deterministic fallback policies.

### Gate 5 — Packaging

Keep code and model licenses/provenance, establish a distinct Swift release process, package converted assets separately, and make downloads explicit and integrity-checked. Runtime loading currently accepts local artifacts and does not verify a downloaded artifact signature. The source weight hash is provenance, not a signature over the converted package. Training and benchmark notebooks remain Python and are not claimed as rewritten.

## Source references

- Upstream implementation: `laya/common.py`, `agent.py`, `lang.py`, `router.py`, `presets.py`, `email.py` at the base commit above.
- Swift 6.4 announcement: https://www.swift.org/blog/swift-6.4-released/ (September 15, 2026).
- Core ML conversion: https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html
- Flexible shapes: https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html
- Core ML array shape constraints: https://developer.apple.com/documentation/CoreML/MLMultiArrayConstraint#Accessing-the-Constraints
- Export graph: https://github.com/mizorewww/laya-coreml/blob/main/laya_coreml/torch_model.py
- Swift tokenizer source: https://github.com/huggingface/swift-transformers/tree/9088d55148b799e853cf4e039b0f0a1e3efe034c
- Original model card: https://huggingface.co/convaiinnovations/laya
