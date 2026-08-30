import json
import os
import tempfile
import unittest

import faiss
import numpy as np

from segvlad_vector_db import (
    artifact_paths,
    benchmark_index,
    build_flat_l2_index,
    build_image_offsets,
    load_database,
    load_queries,
    prepare_vectors,
    resolve_source_paths,
    save_database,
    save_queries,
)


class SegvladVectorDbTest(unittest.TestCase):
    def test_persistent_index_metadata_and_benchmark(self):
        raw_database = np.asarray(
            [[2.0, 0.0], [1.0, 1.0], [0.0, 3.0]], dtype=np.float32
        )
        database = prepare_vectors(raw_database, normalize=True)
        index = build_flat_l2_index(database)

        with tempfile.TemporaryDirectory() as output_dir:
            paths = save_database(
                index,
                database,
                segment_to_image=[0, 0, 1],
                reference_paths=["/images/a.jpg", "/images/b.jpg"],
                output_dir=output_dir,
                name="test_db",
                normalize=True,
                extra_metadata={"dataset": "17places"},
            )

            self.assertEqual(paths, artifact_paths(output_dir, "test_db"))
            self.assertTrue(os.path.isfile(paths["index"]))
            self.assertTrue(os.path.isfile(paths["metadata"]))

            restored_index = faiss.read_index(paths["index"])
            distances, indices = restored_index.search(database[:1], 1)
            self.assertEqual(indices.tolist(), [[0]])
            self.assertAlmostEqual(float(distances[0, 0]), 0.0)

            loaded_index, segment_map, reference_paths, description = load_database(
                output_dir, "test_db"
            )
            self.assertEqual(loaded_index.ntotal, 3)
            self.assertEqual(description["dataset"], "17places")
            resolved = resolve_source_paths(
                np.asarray([[0, 2]]), segment_map, reference_paths
            )
            self.assertEqual(resolved.tolist(), [["/images/a.jpg", "/images/b.jpg"]])

            with np.load(paths["metadata"], allow_pickle=False) as metadata:
                self.assertEqual(metadata["segment_to_image"].tolist(), [0, 0, 1])
                self.assertEqual(
                    metadata["reference_paths"].tolist(),
                    ["/images/a.jpg", "/images/b.jpg"],
                )
                description = json.loads(str(metadata["metadata_json"]))
                self.assertEqual(description["dataset"], "17places")
                self.assertEqual(description["dimension"], 2)

            query_vectors = prepare_vectors(
                np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
                normalize=True,
            )
            save_queries(
                query_vectors,
                segment_to_image=[0, 1],
                query_paths=["/queries/a.jpg", "/queries/b.jpg"],
                output_path=paths["queries"],
                normalize=True,
            )
            loaded_vectors, offsets, query_paths = load_queries(paths["queries"])
            np.testing.assert_allclose(loaded_vectors, query_vectors)
            self.assertEqual(offsets.tolist(), [0, 1, 2])
            self.assertEqual(query_paths.tolist(), ["/queries/a.jpg", "/queries/b.jpg"])

            result = benchmark_index(
                restored_index,
                loaded_vectors,
                offsets,
                top_k=2,
                warmup=0,
                repeats=2,
            )
            self.assertEqual(result["query_images"], 2)
            self.assertEqual(result["samples"], 4)
            self.assertEqual(result["top_k"], 2)
            self.assertGreater(result["image_queries_per_second"], 0)

    def test_offsets_include_images_without_segments(self):
        offsets = build_image_offsets([0, 0, 2], image_count=4)
        self.assertEqual(offsets.tolist(), [0, 2, 2, 3, 3])

    def test_prepare_vectors_rejects_zero_vector(self):
        with self.assertRaisesRegex(ValueError, "zero-length"):
            prepare_vectors(np.zeros((1, 2), dtype=np.float32), normalize=True)


if __name__ == "__main__":
    unittest.main()
