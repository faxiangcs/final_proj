"""Guard the ms-swift 4.x multimodal row schema.

ms-swift >= 4.x reads images from the row-level ``images`` column. An ``image``
key inside a message is silently dropped by RowPreprocessor, which turns every
training step into text-only training without any error.
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.build_dataset import write_jsonl


RECORDS = [
    {
        "image": "/tmp/example_000.png",
        "prompt": "prompt",
        "label": "label",
    }
]


class DatasetFormatTest(unittest.TestCase):
    def _write(self, with_solution: bool) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            write_jsonl(str(path), RECORDS, with_solution=with_solution)
            return json.loads(path.read_text().splitlines()[0])

    def test_sft_rows_put_images_at_row_level(self):
        row = self._write(with_solution=False)
        self.assertEqual(row["images"], ["/tmp/example_000.png"])
        self.assertNotIn("image", row["messages"][0])
        self.assertEqual(row["messages"][0]["content"], "prompt")
        self.assertEqual(
            row["messages"][1], {"role": "assistant", "content": "label"}
        )

    def test_grpo_rows_put_images_at_row_level(self):
        row = self._write(with_solution=True)
        self.assertEqual(row["images"], ["/tmp/example_000.png"])
        self.assertNotIn("image", row["messages"][0])
        self.assertEqual(row["solution"], "label")
        self.assertEqual(len(row["messages"]), 1)


if __name__ == "__main__":
    unittest.main()
