import unittest

from dataset.empathy_dataset import build_split_manifest, validate_split_manifest


def _raw_data(count=100):
    return [{"conv_id": f"hit:{index}_conv:{index}"} for index in range(count)]


class SplitManifestTest(unittest.TestCase):
    def test_group_split_is_deterministic_and_disjoint(self):
        raw = _raw_data()
        first = build_split_manifest(raw, val_ratio=0.1, test_ratio=0.2, seed=7)
        second = build_split_manifest(raw, val_ratio=0.1, test_ratio=0.2, seed=7)
        self.assertEqual(first, second)

        splits = {name: set(ids) for name, ids in first["splits"].items()}
        self.assertEqual(len(splits["train"]), 70)
        self.assertEqual(len(splits["val"]), 10)
        self.assertEqual(len(splits["test"]), 20)
        self.assertFalse(splits["train"] & splits["val"])
        self.assertFalse(splits["train"] & splits["test"])
        self.assertFalse(splits["val"] & splits["test"])

    def test_validator_rejects_leakage(self):
        raw = _raw_data(3)
        manifest = build_split_manifest(raw, val_ratio=0.0, test_ratio=0.0, seed=1)
        leaked = raw[0]["conv_id"]
        manifest["splits"]["test"].append(leaked)
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_split_manifest(manifest, raw)


if __name__ == "__main__":
    unittest.main()
