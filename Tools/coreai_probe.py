#!/usr/bin/env python3
"""Bounded Core AI export probe for the pinned English Laya checkpoint.

This is an experiment only. It exports one fixed (1, 128, 16) input shape,
then verifies the stored fixture rows that fit that shape. No Swift backend yet.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def stage(source: Path, destination: Path) -> Path:
    destination.mkdir()
    for name in ("encoder", "tokenizer"):
        shutil.copytree(source / name, destination / name)
    shutil.copyfile(source / "rl_agent_config.json", destination / "rl_agent_config.json")
    (destination / "model.safetensors").symlink_to((source / "model.safetensors").resolve(strict=True))
    return destination


async def verify_asset(asset: Path, fixture_path: Path, compute: str) -> None:
    from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions

    fixtures = json.loads(fixture_path.read_text(encoding="utf-8"))
    options = SpecializationOptions.cpu_only()
    if compute == "gpu":
        gpu = next((unit for unit in ComputeUnitKind.available_kinds() if str(unit) == "GPU"), None)
        if gpu is None:
            raise RuntimeError("Core AI reports no GPU compute unit.")
        options = SpecializationOptions.from_preferred_compute_unit_kind(gpu)
    print(f"Specialization: {compute} ({options})")
    model = await AIModel.load(asset, specialization_options=options)
    function = model.load_function("main")
    checked = skipped = 0
    for fixture in fixtures:
        batch = fixture["batch"]
        length, option_count, rows = batch["sequenceLength"], batch["optionCount"], batch["batchSize"]
        if rows != 1 or length > 128 or option_count > 16:
            print(f"skipped {fixture['name']}: B={rows}, L={length}, K={option_count}")
            skipped += 1
            continue
        values = {
            "input_ids": np.asarray(batch["inputIDs"], dtype=np.int32).reshape(1, length),
            "attention_mask": np.asarray(batch["attentionMask"], dtype=np.int32).reshape(1, length),
            "marker_pos": np.asarray(batch["markerPositions"], dtype=np.int32).reshape(1, option_count),
            "marker_mask": np.asarray(batch["markerMask"], dtype=np.bool_).reshape(1, option_count),
            "qtype": np.asarray(batch["questionTypes"], dtype=np.int32).reshape(1),
        }
        inputs = {
            "input_ids": np.pad(values["input_ids"], ((0, 0), (0, 128 - length))),
            "attention_mask": np.pad(values["attention_mask"], ((0, 0), (0, 128 - length))),
            "marker_pos": np.pad(values["marker_pos"], ((0, 0), (0, 16 - option_count))),
            "marker_mask": np.pad(values["marker_mask"], ((0, 0), (0, 16 - option_count))),
            "qtype": values["qtype"],
        }
        outputs = await function({name: NDArray(data=value) for name, value in inputs.items()})
        got_logits = np.asarray(outputs["logits"].numpy())[:, :option_count]
        got_actions = np.asarray(outputs["act_logits"].numpy())
        expected_logits = np.asarray(fixture["output"]["logits"], dtype=np.float32)
        expected_actions = np.asarray(fixture["output"]["actionLogits"], dtype=np.float32)
        for actual in (got_logits, got_actions):
            if not np.isfinite(actual).all():
                raise AssertionError(f"{fixture['name']}: nonfinite output")
        np.testing.assert_allclose(got_logits, expected_logits, atol=1e-3, rtol=1e-6, equal_nan=False, err_msg=fixture["name"])
        np.testing.assert_allclose(got_actions, expected_actions, atol=1e-3, rtol=1e-6, equal_nan=False, err_msg=fixture["name"])
        if int(np.argmax(got_logits[0])) != int(np.argmax(expected_logits[0])):
            raise AssertionError(f"{fixture['name']}: final winner changed")
        checked += 1
        print(f"verified {fixture['name']}: winner={int(np.argmax(got_logits[0]))}")
    if checked == 0:
        raise ValueError("No fixture rows fit this asset; zero tests is not a parity pass.")
    print(f"Verified {checked} Core AI fixture rows; skipped {skipped} outside B=1, L<=128, K<=16.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fixture", type=Path, default=ROOT / "exports/laya-english-1c5edc17/parity.json")
    parser.add_argument("--output", type=Path, default=ROOT / "exports/coreai-probe/laya-fixed-b1-l128-k16.aimodel")
    parser.add_argument("--verify-only", action="store_true", help="Verify an existing Core AI asset against supported stored fixtures.")
    parser.add_argument("--compute", choices=["cpu", "gpu"], default="cpu", help="Verification specialization. GPU is a preference, not placement proof.")
    args = parser.parse_args()

    if args.verify_only:
        asyncio.run(verify_asset(args.output, args.fixture, args.compute))
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required unless --verify-only is used")

    import torch
    from coreai_torch import TorchConverter, get_decomp_table
    from laya.agent import Agent

    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))[0]
    batch = fixture["batch"]
    values = {
        "input_ids": np.asarray(batch["inputIDs"], dtype=np.int32).reshape(batch["batchSize"], batch["sequenceLength"]),
        "attention_mask": np.asarray(batch["attentionMask"], dtype=np.int32).reshape(batch["batchSize"], batch["sequenceLength"]),
        "marker_pos": np.asarray(batch["markerPositions"], dtype=np.int32).reshape(batch["batchSize"], batch["optionCount"]),
        "marker_mask": np.asarray(batch["markerMask"], dtype=np.bool_).reshape(batch["batchSize"], batch["optionCount"]),
        "qtype": np.asarray(batch["questionTypes"], dtype=np.int32).reshape(batch["batchSize"]),
    }
    # Core AI's first experiment is intentionally fixed-size and single-row.
    if values["input_ids"].shape[0] != 1:
        raise ValueError("the first fixture must have batch size 1")
    padded = {
        "input_ids": np.full((1, 128), 0, dtype=np.int32),
        "attention_mask": np.zeros((1, 128), dtype=np.int32),
        "marker_pos": np.zeros((1, 16), dtype=np.int32),
        "marker_mask": np.zeros((1, 16), dtype=np.bool_),
        "qtype": values["qtype"],
    }
    for name in ("input_ids", "attention_mask"):
        padded[name][0, : values[name].shape[1]] = values[name][0]
    for name in ("marker_pos", "marker_mask"):
        padded[name][0, : values[name].shape[1]] = values[name][0]

    with tempfile.TemporaryDirectory(prefix="coreai-laya-source-") as temp:
        checkpoint = stage(args.checkpoint, Path(temp) / "checkpoint")
        agent = Agent(str(checkpoint), device="cpu")
        agent.model.eval()

        class ExportGraph(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
                return self.model(input_ids.long(), attention_mask.long(), marker_pos.long(), marker_mask.bool(), qtype.long())

        graph = ExportGraph(agent.model).eval()
        inputs = tuple(torch.from_numpy(padded[name]) for name in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"))
        with torch.inference_mode():
            reference = graph(*inputs)
        print("torch reference: logits", tuple(reference[0].shape), "actions", tuple(reference[1].shape))

        print("torch.export: begin")
        exported = torch.export.export(graph, args=inputs)
        exported = exported.run_decompositions(get_decomp_table())
        print("torch.export: success; converting")
        program = TorchConverter().add_exported_program(
            exported_program=exported,
            input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
            output_names=["logits", "act_logits"],
        ).to_coreai()
        program.optimize()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        program.save_asset(args.output)
        print("Core AI asset:", args.output)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
