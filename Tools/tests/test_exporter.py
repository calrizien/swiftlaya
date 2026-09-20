import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("export_coreml", Path(__file__).parents[1] / "export_coreml.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExporterUtilityTests(unittest.TestCase):
    def test_choice_order_and_false_are_preserved(self):
        q = {"type": "choice", "instructions": "x", "criteria": {"z": False, "a": 0}}
        result = module.native_question("q", q)
        self.assertEqual(result["criteria"], [{"label": "z", "criterion": False}, {"label": "a", "criterion": 0}])
        self.assertEqual(list(q["criteria"]), ["z", "a"])

    def test_json_and_source_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / "x.json"
            module.dump(file, {"text": "雪", "false": False})
            self.assertEqual(json.loads(file.read_text()), {"text": "雪", "false": False})
            self.assertEqual(len(module.file_sha256(file)), 64)
            original = module.file_sha256(file)
            module.dump(file, {"text": "changed"})
            self.assertNotEqual(original, module.file_sha256(file))

    def test_staging_does_not_mutate_original_tokenizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "source"
            (original / "encoder").mkdir(parents=True)
            (original / "tokenizer").mkdir()
            module.dump(original / "tokenizer" / "tokenizer_config.json", {"tokenizer_class": "TokenizersBackend"})
            module.dump(original / "encoder" / "config.json", {})
            module.dump(original / "rl_agent_config.json", {})
            (original / "model.safetensors").write_bytes(b"test-only-placeholder")
            staged = module.stage_checkpoint(original, root / "staged")
            module.dump(staged / "tokenizer" / "tokenizer_config.json", {"tokenizer_class": "changed"})
            self.assertEqual(json.loads((original / "tokenizer" / "tokenizer_config.json").read_text())["tokenizer_class"], "TokenizersBackend")
            self.assertTrue((staged / "model.safetensors").is_symlink())
            self.assertEqual((staged / "model.safetensors").read_bytes(), b"test-only-placeholder")

    def test_nonfinite_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                module.dump(Path(temporary) / "x.json", {"bad": float("nan")})


if __name__ == "__main__":
    unittest.main()
