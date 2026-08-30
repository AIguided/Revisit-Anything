"""Benchmark a persistent SegVLAD FAISS database with saved query vectors."""

import argparse
import json
import os

import faiss

from segvlad_vector_db import (
    artifact_paths,
    benchmark_index,
    load_database,
    load_queries,
    save_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark per-image FAISS lookup latency for a SegVLAD database."
    )
    parser.add_argument("--db-dir", required=True, help="Directory containing database artifacts")
    parser.add_argument("--name", required=True, help="Artifact stem created by place_rec_main.py")
    parser.add_argument("--top-k", type=int, default=200, help="Nearest segments to retrieve")
    parser.add_argument("--warmup", type=int, default=1, help="Unmeasured full-query passes")
    parser.add_argument("--repeats", type=int, default=5, help="Measured full-query passes")
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Optional number of FAISS OpenMP threads",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON result path; defaults to the database artifact directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = artifact_paths(args.db_dir, args.name)
    for artifact in (paths["index"], paths["metadata"], paths["queries"]):
        if not os.path.isfile(artifact):
            raise FileNotFoundError("Required artifact not found: " + artifact)

    if args.threads is not None:
        if args.threads <= 0:
            raise ValueError("--threads must be positive")
        faiss.omp_set_num_threads(args.threads)

    index, _, _, database_metadata = load_database(args.db_dir, args.name)
    query_vectors, image_offsets, _ = load_queries(paths["queries"])
    result = benchmark_index(
        index,
        query_vectors,
        image_offsets,
        top_k=args.top_k,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    result["faiss_threads"] = faiss.omp_get_max_threads()
    result["database"] = os.path.abspath(paths["index"])
    result["queries"] = os.path.abspath(paths["queries"])
    result["database_metadata"] = database_metadata

    output_path = args.output or paths["benchmark"]
    save_benchmark(result, output_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("Benchmark saved to " + output_path)


if __name__ == "__main__":
    main()
