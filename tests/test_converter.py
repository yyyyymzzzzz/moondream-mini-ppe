import json
import random
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.convert_ppe_yolo import BoxRecord, Counts, build_easy_samples, process_split


class ConverterTests(unittest.TestCase):
    def test_location_uses_dataset_class_order(self):
        class_names = ["vest", "person", "helmet"]
        records = [
            BoxRecord(0, 0.9, 0.5, 0.1, 0.1),
            BoxRecord(2, 0.1, 0.5, 0.1, 0.1),
        ]
        counts = Counts(person=0, helmet=1, vest=1, no_helmet=0, no_vest=0)
        location_rows = [row for row in build_easy_samples(records, counts, class_names) if row[2] == "location"]
        self.assertIn(("Where is the helmet-wearing worker?", "left", "location", "location_3"), location_rows)
        self.assertIn(("Where is the vest-wearing worker?", "right", "location", "location_3"), location_rows)

    def test_process_split_uses_source_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset-a"
            image_dir = dataset / "valid" / "images"
            label_dir = dataset / "valid" / "labels"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(image_dir / "sample.jpg")
            (label_dir / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            output = root / "output"
            stats = process_split(
                split_name="test",
                source_split="valid",
                dataset_roots=[dataset],
                class_names=["helmet"],
                output_dir=output,
                rng=random.Random(42),
                samples_per_image=1,
                copy_images=True,
                absolute_image_paths=False,
            )

            self.assertEqual(stats["images"], 1)
            row = json.loads((output / "test.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(row["split"], "test")
            self.assertTrue((output / row["image"]).is_file())


if __name__ == "__main__":
    unittest.main()
