"""Persistent FAISS storage, retrieval reporting, and benchmark helpers."""

import csv
import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import numpy as np


def prepare_vectors(vectors: Any, normalize: bool) -> np.ndarray:
    """Convert Torch/NumPy vectors into the representation FAISS expects."""
    if hasattr(vectors, "detach"):
        vectors = vectors.detach().cpu().numpy()

    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("vectors must have shape [count, dimension] and be non-empty")
    if not np.isfinite(array).all():
        raise ValueError("vectors contain NaN or infinite values")

    array = np.ascontiguousarray(array)
    if normalize:
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("cannot normalize zero-length vectors")
        array = np.ascontiguousarray(array / norms, dtype=np.float32)
    return array


def build_flat_l2_index(vectors: np.ndarray) -> faiss.Index:
    """Build the exact L2 index used by the original evaluation pipeline."""
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index


def build_image_offsets(
    segment_to_image: Sequence[int], image_count: int
) -> np.ndarray:
    """Create CSR-style offsets for contiguous segment rows grouped by image."""
    image_ids = np.asarray(segment_to_image, dtype=np.int64)
    if image_count <= 0:
        raise ValueError("image_count must be positive")
    if image_ids.ndim != 1:
        raise ValueError("segment_to_image must be one-dimensional")
    if image_ids.size == 0:
        raise ValueError("segment_to_image must not be empty")
    if image_ids.min() < 0 or image_ids.max() >= image_count:
        raise ValueError("segment_to_image contains an out-of-range image id")
    if np.any(image_ids[1:] < image_ids[:-1]):
        raise ValueError("segment rows must be grouped in ascending image order")

    offsets = np.searchsorted(
        image_ids, np.arange(image_count + 1, dtype=np.int64), side="left"
    )
    return offsets.astype(np.int64, copy=False)


def artifact_paths(output_dir: str, name: str) -> Dict[str, str]:
    """Return stable artifact names shared by the builder and benchmark CLI."""
    return {
        "index": os.path.join(output_dir, name + ".faiss"),
        "metadata": os.path.join(output_dir, name + ".metadata.npz"),
        "queries": os.path.join(output_dir, name + ".queries.npz"),
        "benchmark": os.path.join(output_dir, name + ".benchmark.json"),
        "results": os.path.join(output_dir, name + ".results.csv"),
        "metrics": os.path.join(output_dir, name + ".metrics.csv"),
    }


def save_database(
    index: faiss.Index,
    index_vectors: np.ndarray,
    segment_to_image: Sequence[int],
    reference_paths: Sequence[str],
    output_dir: str,
    name: str,
    normalize: bool,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Persist a FAISS index plus the mapping from index rows to sources."""
    reference_paths_array = np.asarray(reference_paths, dtype=np.str_)
    segment_to_image_array = np.asarray(segment_to_image, dtype=np.int64)
    if index.ntotal != index_vectors.shape[0]:
        raise ValueError("FAISS row count does not match the supplied vectors")
    if segment_to_image_array.shape != (index.ntotal,):
        raise ValueError("segment_to_image must contain one id per FAISS row")
    if reference_paths_array.ndim != 1 or reference_paths_array.size == 0:
        raise ValueError("reference_paths must be a non-empty one-dimensional list")
    build_image_offsets(segment_to_image_array, len(reference_paths_array))

    os.makedirs(output_dir, exist_ok=True)
    paths = artifact_paths(output_dir, name)
    faiss.write_index(index, paths["index"])

    metadata = {
        "name": name,
        "metric": "squared_l2",
        "normalized": bool(normalize),
        "dimension": int(index.d),
        "segment_count": int(index.ntotal),
        "reference_count": int(reference_paths_array.size),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    np.savez_compressed(
        paths["metadata"],
        segment_to_image=segment_to_image_array,
        reference_paths=reference_paths_array,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return paths


def load_database(
    output_dir: str, name: str
) -> Tuple[faiss.Index, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load an index and the metadata needed to resolve rows to sources."""
    paths = artifact_paths(output_dir, name)
    index = faiss.read_index(paths["index"])
    with np.load(paths["metadata"], allow_pickle=False) as data:
        segment_to_image = np.asarray(data["segment_to_image"], dtype=np.int64)
        reference_paths = np.asarray(data["reference_paths"], dtype=np.str_)
        metadata = json.loads(str(data["metadata_json"]))
    if segment_to_image.shape != (index.ntotal,):
        raise ValueError("database metadata does not match the FAISS row count")
    return index, segment_to_image, reference_paths, metadata


def resolve_source_paths(
    segment_indices: Any,
    segment_to_image: np.ndarray,
    reference_paths: np.ndarray,
) -> np.ndarray:
    """Map FAISS segment result indices to their original image paths."""
    indices = np.asarray(segment_indices, dtype=np.int64)
    if indices.size and (indices.min() < 0 or indices.max() >= len(segment_to_image)):
        raise ValueError("segment result contains an out-of-range FAISS row")
    return reference_paths[segment_to_image[indices]]


def _path_index(value: str, paths: Sequence[str], column: str) -> int:
    """Resolve a CSV path/index against a set of image paths."""
    text = str(value).strip()
    if not text:
        raise ValueError(f"ground-truth {column} value must not be empty")
    if text.isdigit():
        index = int(text)
        if 0 <= index < len(paths):
            return index

    normalized = os.path.normcase(os.path.abspath(os.path.expanduser(text)))
    exact = {
        os.path.normcase(os.path.abspath(os.path.expanduser(str(path)))): index
        for index, path in enumerate(paths)
    }
    if normalized in exact:
        return exact[normalized]

    basename = os.path.normcase(os.path.basename(text))
    matches = [
        index
        for index, path in enumerate(paths)
        if os.path.normcase(os.path.basename(str(path))) == basename
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"ground-truth {column} basename is ambiguous; use a full path: {text}"
        )
    raise ValueError(f"ground-truth {column} image was not found: {text}")


def load_ground_truth_csv(
    input_path: str,
    source_paths: Sequence[str],
    target_paths: Sequence[str],
) -> List[np.ndarray]:
    """Load correct source/target image pairs for every target query.

    Values in the required ``source`` and ``target`` columns may be image
    indexes, full paths, or unambiguous basenames. Multiple source rows may be
    supplied for one target.
    """
    positives: List[set] = [set() for _ in target_paths]
    with open(input_path, "r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = {str(name).strip().lower(): name for name in reader.fieldnames or []}
        if "source" not in fieldnames or "target" not in fieldnames:
            raise ValueError("ground-truth CSV must contain source and target columns")
        for row_number, row in enumerate(reader, start=2):
            try:
                source_index = _path_index(
                    row[fieldnames["source"]], source_paths, "source"
                )
                target_index = _path_index(
                    row[fieldnames["target"]], target_paths, "target"
                )
            except ValueError as error:
                raise ValueError(f"ground-truth CSV row {row_number}: {error}") from error
            positives[target_index].add(source_index)

    missing = [index for index, values in enumerate(positives) if not values]
    if missing:
        names = ", ".join(os.path.basename(str(target_paths[index])) for index in missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(
            "ground-truth CSV has no source match for target(s): " + names + suffix
        )
    return [np.asarray(sorted(values), dtype=np.int64) for values in positives]


def save_retrieval_csv(
    predictions: Sequence[Sequence[int]],
    source_paths: Sequence[str],
    target_paths: Sequence[str],
    results_path: str,
    metrics_path: Optional[str] = None,
    ground_truth: Optional[Sequence[Sequence[int]]] = None,
    top_k: int = 5,
) -> Dict[str, Optional[str]]:
    """Write ranked source/target matches and optional retrieval metrics.

    Accuracy is the fraction of target queries with at least one correct source
    in the top-k results. Precision, recall, and F1 are micro-averaged over all
    source/target pairs at each cutoff.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if len(predictions) != len(target_paths):
        raise ValueError("predictions must contain one row per target image")
    if ground_truth is not None and len(ground_truth) != len(target_paths):
        raise ValueError("ground_truth must contain one row per target image")

    ranked_predictions: List[List[int]] = []
    for target_index, prediction in enumerate(predictions):
        unique = []
        seen = set()
        for raw_index in prediction:
            source_index = int(raw_index)
            if source_index < 0 or source_index >= len(source_paths):
                raise ValueError(
                    f"prediction for target {target_index} contains invalid source index "
                    f"{source_index}"
                )
            if source_index not in seen:
                seen.add(source_index)
                unique.append(source_index)
            if len(unique) == top_k:
                break
        ranked_predictions.append(unique)

    truth_sets = None
    if ground_truth is not None:
        truth_sets = [
            {
                int(value)
                for value in values
                if 0 <= int(value) < len(source_paths)
            }
            for values in ground_truth
        ]
        missing_truth = [index for index, values in enumerate(truth_sets) if not values]
        if missing_truth:
            raise ValueError(
                "ground_truth contains no valid source index for target "
                + str(missing_truth[0])
            )

    os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)
    with open(results_path, "w", encoding="utf-8", newline="") as output_file:
        fieldnames = [
            "target_index",
            "target",
            "rank",
            "source_index",
            "source",
            "is_correct",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for target_index, prediction in enumerate(ranked_predictions):
            for rank, source_index in enumerate(prediction, start=1):
                is_correct: Any = ""
                if truth_sets is not None:
                    is_correct = source_index in truth_sets[target_index]
                writer.writerow(
                    {
                        "target_index": target_index,
                        "target": str(target_paths[target_index]),
                        "rank": rank,
                        "source_index": source_index,
                        "source": str(source_paths[source_index]),
                        "is_correct": is_correct,
                    }
                )

    written_metrics = None
    if truth_sets is not None:
        if not metrics_path:
            raise ValueError("metrics_path is required when ground_truth is supplied")
        os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8", newline="") as output_file:
            fieldnames = [
                "top_k",
                "query_count",
                "correct_queries",
                "true_positives",
                "false_positives",
                "false_negatives",
                "precision",
                "recall",
                "accuracy",
                "f1",
            ]
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for cutoff in range(1, top_k + 1):
                true_positives = 0
                predicted_positives = 0
                actual_positives = sum(len(values) for values in truth_sets)
                correct_queries = 0
                for prediction, expected in zip(ranked_predictions, truth_sets):
                    selected = set(prediction[:cutoff])
                    matches = len(selected & expected)
                    true_positives += matches
                    predicted_positives += len(selected)
                    correct_queries += int(matches > 0)
                false_positives = predicted_positives - true_positives
                false_negatives = actual_positives - true_positives
                precision = (
                    true_positives / predicted_positives if predicted_positives else 0.0
                )
                recall = true_positives / actual_positives if actual_positives else 0.0
                accuracy = correct_queries / len(target_paths) if target_paths else 0.0
                f1 = (
                    2.0 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0.0
                )
                writer.writerow(
                    {
                        "top_k": cutoff,
                        "query_count": len(target_paths),
                        "correct_queries": correct_queries,
                        "true_positives": true_positives,
                        "false_positives": false_positives,
                        "false_negatives": false_negatives,
                        "precision": precision,
                        "recall": recall,
                        "accuracy": accuracy,
                        "f1": f1,
                    }
                )
        written_metrics = metrics_path

    return {"results": results_path, "metrics": written_metrics}


def save_queries(
    query_vectors: np.ndarray,
    segment_to_image: Sequence[int],
    query_paths: Sequence[str],
    output_path: str,
    normalize: bool,
) -> str:
    """Persist query vectors and per-image offsets for repeatable benchmarks."""
    query_paths_array = np.asarray(query_paths, dtype=np.str_)
    segment_to_image_array = np.asarray(segment_to_image, dtype=np.int64)
    if segment_to_image_array.shape != (query_vectors.shape[0],):
        raise ValueError("segment_to_image must contain one id per query vector")
    offsets = build_image_offsets(segment_to_image_array, len(query_paths_array))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez_compressed(
        output_path,
        vectors=query_vectors,
        image_offsets=offsets,
        segment_to_image=segment_to_image_array,
        query_paths=query_paths_array,
        normalized=np.asarray(bool(normalize)),
    )
    return output_path


def load_queries(query_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load vectors, offsets, and source paths from a saved query artifact."""
    with np.load(query_path, allow_pickle=False) as data:
        vectors = np.ascontiguousarray(data["vectors"], dtype=np.float32)
        offsets = np.asarray(data["image_offsets"], dtype=np.int64)
        paths = np.asarray(data["query_paths"], dtype=np.str_)
    if offsets.shape != (len(paths) + 1,):
        raise ValueError("query image offsets do not match the saved query paths")
    if offsets[0] != 0 or offsets[-1] != len(vectors):
        raise ValueError("query image offsets do not cover all saved vectors")
    return vectors, offsets, paths


def _query_slices(offsets: np.ndarray) -> Iterable[slice]:
    for image_id in range(len(offsets) - 1):
        start = int(offsets[image_id])
        end = int(offsets[image_id + 1])
        if end > start:
            yield slice(start, end)


def benchmark_index(
    index: faiss.Index,
    query_vectors: np.ndarray,
    image_offsets: np.ndarray,
    top_k: int = 200,
    warmup: int = 1,
    repeats: int = 5,
) -> Dict[str, Any]:
    """Measure FAISS search latency for every original query image."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if warmup < 0:
        raise ValueError("warmup cannot be negative")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if query_vectors.ndim != 2 or query_vectors.shape[1] != index.d:
        raise ValueError("query vector dimension does not match the FAISS index")

    slices = list(_query_slices(np.asarray(image_offsets, dtype=np.int64)))
    if not slices:
        raise ValueError("there are no non-empty query images to benchmark")
    effective_top_k = min(top_k, int(index.ntotal))

    for _ in range(warmup):
        for query_slice in slices:
            index.search(query_vectors[query_slice], effective_top_k)

    latencies_ms = []
    searched_segments = 0
    benchmark_start = time.perf_counter()
    for _ in range(repeats):
        for query_slice in slices:
            start = time.perf_counter()
            index.search(query_vectors[query_slice], effective_top_k)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            searched_segments += query_slice.stop - query_slice.start
    elapsed_seconds = time.perf_counter() - benchmark_start

    latency_array = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "index_type": type(index).__name__,
        "index_vectors": int(index.ntotal),
        "dimension": int(index.d),
        "query_images": len(slices),
        "query_segments": int(image_offsets[-1]),
        "top_k": effective_top_k,
        "warmup_repeats": warmup,
        "measured_repeats": repeats,
        "samples": int(latency_array.size),
        "mean_ms_per_image": float(latency_array.mean()),
        "p50_ms_per_image": float(np.percentile(latency_array, 50)),
        "p95_ms_per_image": float(np.percentile(latency_array, 95)),
        "p99_ms_per_image": float(np.percentile(latency_array, 99)),
        "image_queries_per_second": float(latency_array.size / elapsed_seconds),
        "segment_queries_per_second": float(searched_segments / elapsed_seconds),
        "elapsed_seconds": float(elapsed_seconds),
    }


def save_benchmark(result: Dict[str, Any], output_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return output_path
