# Core AI probe — 2026-09-20

This is a separate experiment, not a replacement for the verified Core ML backend.
The existing Core ML artifact and environment were not changed.

## Evidence

- Host: Apple M1 Max (`h13c`), macOS 27, Xcode 27.
- Environment: Python 3.11.10, torch 2.8.0, transformers 4.57.3, NumPy 2.4.6,
  coreai-torch 0.4.1, coreai-core 1.0.0b2, in a separate `.venv-coreai`.
- Source: the same English checkpoint pinned in `VALIDATION.md`.
- The complete upstream Laya decision model exported through `torch.export`,
  Core AI decompositions, and `TorchConverter`. It retains both output heads.
- Fixed shape: one row, 128 sequence positions, 16 option columns.
- Source `.aimodel` runtime checks matched stored option and action logits with
  `atol=0.001, rtol=1e-6`, and preserved the highest-logit option for six fixtures.
  Three cases do not fit this shape: mixed-ragged, long-state, and batch-boundary.
- CPU-only and GPU-preferred specialization worked. GPU preference still permits
  CPU and Neural Engine work; this is not instrumented GPU-placement proof.
- Host-only GPU AOT compilation succeeded and the resulting `.aimodelc` passed
  `coreai-build inspect`. Python did not load that compiled path directly, so
  compiled-asset inference parity is not established.
- An all-architecture AOT attempt failed with float32 Neural Engine I/O errors.
  Its temporary output was removed. Source and host-only compiled assets are
  approximately 1.6 GB each and remain ignored local files.

## Reproduce the bounded checks

Use a separate environment with the versions above; do not upgrade the verified
Core ML environment in place. Reuse the cached pinned checkpoint, not new weights.

```sh
.venv-coreai/bin/python Tools/coreai_probe.py --checkpoint /path/to/pinned/checkpoint
.venv-coreai/bin/python Tools/coreai_probe.py --verify-only --compute cpu
.venv-coreai/bin/python Tools/coreai_probe.py --verify-only --compute gpu
xcrun swift -e 'import CoreAI; print(AIModel.deviceArchitectureName)'
xcrun coreai-build compile exports/coreai-probe/laya-fixed-b1-l128-k16.aimodel \
  --platform macOS --architecture h13c --preferred-compute gpu \
  --output exports/coreai-probe/gpu-h13c
```

`h13c` is this host, not a universal target. Restrict the architecture explicitly:
the compiler otherwise attempts every supported architecture.

## Remaining gates

This probe does not verify Swift Core AI inference, every fixture, final calibrated
numeric answers, dynamic shapes, batching, ANE execution, app-specific quality, or
whole-app performance. The current trial remains on CPU-only Core ML.

The next useful experiment is a bounded longer-sequence/batched Core AI export,
followed by all existing parity checks and end-to-end request timing. Keep the
same tolerance and reject cancelled, stale, or truncated requests.

Sources: [Apple's RoBERTa export recipe](https://github.com/apple/coreai-models/blob/main/models/roberta/export.py),
[Core AI runtime](https://developer.apple.com/documentation/coreai), and
[AOT compilation](https://developer.apple.com/documentation/coreai/compiling-core-ai-models-ahead-of-time).
