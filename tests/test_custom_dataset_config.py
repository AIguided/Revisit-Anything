import os
import tempfile
import unittest
from unittest.mock import patch

from place_rec_global_config import datasets, list_image_names, resolve_dataset_paths


class CustomDatasetConfigTest(unittest.TestCase):
    def test_custom_paths_use_cli_environment(self):
        environment = {
            "REVISIT_CUSTOM_DATASET_DIR": "/datasets/example",
            "REVISIT_CUSTOM_REFERENCE_DIR": "/images/reference",
            "REVISIT_CUSTOM_QUERY_DIR": "/images/query",
            "REVISIT_CUSTOM_OUTPUT_DIR": "/outputs/example",
        }
        with patch.dict(os.environ, environment, clear=False):
            root, output, reference, query = resolve_dataset_paths(
                "custom", datasets["custom"]
            )

        self.assertEqual(root, "/datasets/example")
        self.assertEqual(output, "/outputs/example")
        self.assertEqual(reference, "/images/reference/")
        self.assertEqual(query, "/images/query/")

    def test_custom_image_listing_ignores_unrelated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("one.JPG", "two.png", "notes.txt"):
                with open(os.path.join(directory, name), "wb") as output_file:
                    output_file.write(b"test")

            self.assertCountEqual(list_image_names(directory), ["one.JPG", "two.png"])


if __name__ == "__main__":
    unittest.main()
