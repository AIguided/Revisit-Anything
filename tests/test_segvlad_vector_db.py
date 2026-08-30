import csv
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
    load_ground_truth_csv,
    load_queries,
    prepare_vectors,
    resolve_source_paths,
    save_database,
    save_queries,
    save_retrieval_csv,
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

    def test_source_target_csv_outputs_results_and_metrics(self):
        source_paths = ["/source/alice.jpg", "/source/bob.jpg"]
        target_paths = ["/target/alice.png", "/target/bob.png"]

        with tempfile.TemporaryDirectory() as output_dir:
            truth_path = os.path.join(output_dir, "ground_truth.csv")
            with open(truth_path, "w", encoding="utf-8", newline="") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=["source", "target"])
                writer.writeheader()
                writer.writerow({"source": "alice.jpg", "target": "alice.png"})
                writer.writerow({"source": "bob.jpg", "target": "bob.png"})

            ground_truth = load_ground_truth_csv(
                truth_path, source_paths, target_paths
            )
            self.assertEqual([values.tolist() for values in ground_truth], [[0], [1]])

            results_path = os.path.join(output_dir, "results.csv")
            metrics_path = os.path.join(output_dir, "metrics.csv")
            written = save_retrieval_csv(
                predictions=[[0, 1], [1, 0]],
                source_paths=source_paths,
                target_paths=target_paths,
                results_path=results_path,
                metrics_path=metrics_path,
                ground_truth=ground_truth,
                top_k=2,
            )
            self.assertEqual(written["results"], results_path)
            self.assertEqual(written["metrics"], metrics_path)

            with open(results_path, encoding="utf-8", newline="") as input_file:
                rows = list(csv.DictReader(input_file))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["source"], "/source/alice.jpg")
            self.assertEqual(rows[0]["target"], "/target/alice.png")
            self.assertEqual(rows[0]["is_correct"], "True")

            with open(metrics_path, encoding="utf-8", newline="") as input_file:
                metrics = list(csv.DictReader(input_file))
            self.assertEqual(metrics[0]["accuracy"], "1.0")
            self.assertEqual(metrics[0]["precision"], "1.0")
            self.assertEqual(metrics[0]["f1"], "1.0")
            self.assertEqual(metrics[1]["precision"], "0.5")


if __name__ == "__main__":
    unittest.main()
