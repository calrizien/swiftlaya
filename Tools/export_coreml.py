#!/usr/bin/env python3
"""Experimental full-graph exporter. Python is build-time only; never used by the Swift runtime.

Does not overwrite an output directory or mark an unverified conversion as verified.
ModernBERT/mmBERT conversion remains a release gate until this command AND native parity pass.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_COMMIT = "d113dca2512fb3eaca313534bc54c7162d87c1d4"


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_checkpoint(source: Path, destination: Path) -> Path:
    """Copy mutable metadata; reference immutable weights without copying gigabytes.

    The upstream Agent may normalize tokenizer_config.json in-place. Never point it
    at the original checkpoint or a shared Hugging Face cache during conversion.
    """
    destination.mkdir()
    for name in ("encoder", "tokenizer"):
        shutil.copytree(source / name, destination / name)
    shutil.copyfile(source / "rl_agent_config.json", destination / "rl_agent_config.json")
    (destination / "model.safetensors").symlink_to((source / "model.safetensors").resolve(strict=True))
    return destination


def native_question(qid, q):
    result = {"id": qid, **q}
    if q["type"] == "choice":
        result["criteria"] = [{"label": k, "criterion": v} for k, v in q["criteria"].items()]
    return result


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Local checkpoint directory or Hugging Face repo id")
    p.add_argument("--subfolder", choices=["multilingual", "typed-decisions"])
    p.add_argument("--revision", help="Required immutable 40-hex Hub commit for non-local checkpoints")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--max-batch", type=int, default=8)
    p.add_argument("--max-options", type=int, default=16)
    p.add_argument("--atol", type=float, default=0.001)
    p.add_argument("--skip-coreml-verification", action="store_true", help="Explicitly produce an UNVERIFIED artifact; native loader rejects it by default")
    a = p.parse_args()
    if not (3 <= a.max_batch <= 32 and 5 <= a.max_options <= 64):
        p.error("Require 3 <= max-batch <= 32 and 5 <= max-options <= 64 for this experimental exporter.")
    if not (0 < a.atol <= 0.01):
        p.error("atol must be positive and at most 0.01; do not mask conversion errors with a loose tolerance.")
    if a.output.exists():
        p.error("Output already exists; choose a new directory. No overwrite is performed.")
    if platform.system() != "Darwin" and not a.skip_coreml_verification:
        p.error("Final Core ML verification requires macOS. Use --skip-coreml-verification only for exploratory conversion.")
    return a


def main() -> None:
    a = arguments()
    # Heavy optional dependencies are imported only for a real export, not --help or utility tests.
    import numpy as np
    import torch
    import coremltools as ct
    sys.path.insert(0, str(ROOT))
    from laya.agent import Agent
    from laya.common import QTYPES, build_sequence, collate_items, render_options

    checkpoint = Path(a.checkpoint).expanduser()
    if not checkpoint.is_dir():
        if not a.revision or len(a.revision) != 40 or any(c not in "0123456789abcdefABCDEF" for c in a.revision):
            raise ValueError("A Hub download requires --revision with an immutable 40-hex commit SHA.")
        from huggingface_hub import snapshot_download
        patterns = [a.subfolder + "/*"] if a.subfolder else ["model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*"]
        checkpoint = Path(snapshot_download(repo_id=a.checkpoint, revision=a.revision, allow_patterns=patterns))
    checkpoint = checkpoint / a.subfolder if a.subfolder else checkpoint
    if not (checkpoint / "encoder" / "config.json").is_file():
        raise ValueError("The checkpoint must include encoder/config.json so the model is built from local architecture metadata.")
    if not (checkpoint / "tokenizer").is_dir():
        raise ValueError("The checkpoint must include its exact tokenizer; encoder-name fallback is not permitted for export.")
    with tempfile.TemporaryDirectory(prefix="swiftlaya-source-") as source_temp:
        source = stage_checkpoint(checkpoint, Path(source_temp) / "checkpoint")
        agent = Agent(str(source), device="cpu")
        agent.model.eval()
        if hasattr(torch.backends, "mha"):
            torch.backends.mha.set_fastpath_enabled(False)
        cfg = dict(agent.cfg)
        cfg.update(max_questions=a.max_batch, max_options=a.max_options)
        max_len, head_len = cfg.get("max_len", 512), cfg.get("head_max_len", 192)
        names = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]

        class ExportGraph(torch.nn.Module):
            def __init__(self, model):
                super().__init__(); self.model = model
            def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
                return self.model(input_ids.long(), attention_mask.long(), marker_pos.long(), marker_mask.bool(), qtype.long())

        def choice(count=3):
            return {"type": "choice", "instructions": "Which option best matches the state?",
                    "criteria": {f"option{i}": f"description {i}" for i in range(count)}}
        noul = {"type": "noul", "instructions": "Does the customer request a refund?"}
        score = {"type": "score", "instructions": "How urgent is this?", "criteria": ["low", "medium", "high"]}
        cases = [
            ("choice", "Please refund the duplicate charge.", {"category": choice()}),
            ("noul", "Please refund the duplicate charge.", {"refund": noul}),
            ("score", "This is blocking our work today.", {"urgency": score}),
            ("mixed-ragged", {"message": "Please help; this is urgent.", "language": "en"}, {"category": choice(5), "refund": noul, "urgency": score}),
            ("unicode", {"message": "मुझे मदद चाहिए — 退款 — München"}, {"refund": noul}),
            ("mask-sanitization", "Please " + agent.tok.mask_token + " refund me", {"refund": noul}),
            ("long-state", "Please review this request. " * max_len, {"refund": noul}),
            ("option-boundary", "Choose an option.", {"category": choice(a.max_options)}),
            ("batch-boundary", "Refund requested.", {f"q{i}": noul for i in range(a.max_batch)}),
        ]
        fixtures, inputs = [], []
        with torch.inference_mode():
            for name, state, questions in cases:
                # Swift canonicalizes JSON object keys. Canonicalize the reference identically.
                state = json.loads(json.dumps(state, ensure_ascii=False, sort_keys=True))
                items, sequences = [], []
                for q in questions.values():
                    internal = agent._to_internal(q)
                    ids, markers = build_sequence(agent.tok, state, internal, max_len, head_len)
                    if len(markers) != len(render_options(internal)) or ids[-1] != agent.tok.sep_token_id:
                        raise ValueError(f"{name}: option/header truncation; lower --max-options.")
                    qt = QTYPES[q["type"]]
                    items.append({"ids": ids, "markers": markers, "qtype": qt})
                    sequences.append({"ids": ids, "markers": markers, "questionType": qt, "truncatedStateTokens": 0})
                b = collate_items([items], agent.tok.pad_token_id)
                example = tuple(b[n].to(torch.int32) for n in names)
                logits, acts = ExportGraph(agent.model)(*example)
                inputs.append(example)
                fixture = {"name": name, "request": {"state": state, "questions": [native_question(k, v) for k, v in questions.items()]},
                           "sequences": sequences,
                           "batch": {"batchSize": len(items), "sequenceLength": b["input_ids"].shape[1], "optionCount": b["marker_pos"].shape[1],
                                     "inputIDs": b["input_ids"].flatten().tolist(), "attentionMask": b["attention_mask"].flatten().tolist(),
                                     "markerPositions": b["marker_pos"].flatten().tolist(), "markerMask": b["marker_mask"].int().flatten().tolist(),
                                     "questionTypes": b["qtype"].tolist()},
                           "output": {"logits": logits.tolist(), "actionLogits": acts.tolist()},
                           "answers": agent.predict(state, questions)["answers"]}
                fixtures.append(fixture)

            graph = ExportGraph(agent.model).eval()
            traced = torch.jit.trace(graph, inputs[3], check_trace=True, check_inputs=[inputs[1], inputs[-1]])
            # A successful trace alone is not dynamic-shape validation.
            for example, fixture in zip(inputs, fixtures):
                actual = traced(*example)
                for tensor, expected in zip(actual, [fixture["output"]["logits"], fixture["output"]["actionLogits"]]):
                    np.testing.assert_allclose(tensor.numpy(), np.asarray(expected), atol=a.atol, rtol=0)

        B = ct.RangeDim(lower_bound=1, upper_bound=a.max_batch, default=inputs[3][0].shape[0], symbol="batch")
        L = ct.RangeDim(lower_bound=1, upper_bound=max_len, default=inputs[3][0].shape[1], symbol="sequence")
        K = ct.RangeDim(lower_bound=2, upper_bound=a.max_options, default=inputs[3][2].shape[1], symbol="options")
        shapes = [(B, L), (B, L), (B, K), (B, K), (B,)]
        converted = ct.convert(traced, convert_to="mlprogram",
                               inputs=[ct.TensorType(name=n, shape=s, dtype=np.int32) for n, s in zip(names, shapes)],
                               outputs=[ct.TensorType(name="logits"), ct.TensorType(name="act_logits")],
                               minimum_deployment_target=ct.target.iOS17,
                               compute_precision=ct.precision.FLOAT32, compute_units=ct.ComputeUnit.CPU_ONLY,
                               skip_model_load=True)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="swiftlaya-export-", dir=a.output.parent) as temp:
            destination = Path(temp) / "artifact"
            destination.mkdir()
            converted.save(str(destination / "model.mlpackage"))
            verified = False
            if not a.skip_coreml_verification:
                native = ct.models.MLModel(str(destination / "model.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
                for example, fixture in zip(inputs, fixtures):
                    actual = native.predict({n: t.numpy() for n, t in zip(names, example)})
                    for key, expected in [("logits", fixture["output"]["logits"]), ("act_logits", fixture["output"]["actionLogits"])]:
                        np.testing.assert_allclose(actual[key], np.asarray(expected), atol=a.atol, rtol=0, err_msg=fixture["name"])
                verified = True
            agent.tok.save_pretrained(destination / "tokenizer")
            dump(destination / "rl_agent_config.json", cfg)
            dump(destination / "special_tokens.json", {"cls": agent.tok.cls_token_id, "sep": agent.tok.sep_token_id,
                  "mask": agent.tok.mask_token_id, "pad": agent.tok.pad_token_id, "maskToken": agent.tok.mask_token})
            dump(destination / "parity.json", fixtures)
            dump(destination / "swiftlaya.json", {"formatVersion": 1, "coreMLVerified": verified,
                 "vocabularySize": agent.model.encoder.config.vocab_size, "actionCount": len(fixtures[0]["output"]["actionLogits"][0]),
                 "checkpoint": a.checkpoint + ("/" + a.subfolder if a.subfolder else ""),
                 "sourceRevision": a.revision or "local-checkpoint", "sourceWeightSHA256": file_sha256(checkpoint / "model.safetensors"),
                 "referenceCodeCommit": REFERENCE_COMMIT,
                 "referenceSourceSHA256": {name: file_sha256(ROOT / "laya" / name) for name in ["agent.py", "common.py"]},
                 "torchVersion": torch.__version__, "coremltoolsVersion": ct.__version__,
                 "computePrecision": "float32", "verificationComputeUnits": "cpuOnly", "absoluteTolerance": a.atol,
                 "nativeSwiftParityVerified": False})
            shutil.copyfile(ROOT / "LICENSE", destination / "LICENSE")
            if a.output.exists():
                raise FileExistsError("Output appeared during conversion; refusing to replace it.")
            os.rename(destination, a.output)
        print(f"Exported {a.output}; Core ML/PyTorch verified={verified}. Native Swift parity is still required.")


if __name__ == "__main__":
    main()
