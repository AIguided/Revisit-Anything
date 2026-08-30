#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source_dir=""
query_dir=""
output_dir=""
dataset="auto"
vocab="auto"
top_k=200
warmup=1
repeats=5
evaluation_top_k=5
preprocess="ask"
sam_checkpoint=""
ground_truth_csv=""
result_csv=""
metrics_csv=""
conda_prefix="${SEGVLAD_CONDA_PREFIX:-${CONDA_PREFIX:-}}"
python_selection_explicit="false"
if [[ -n "${PYTHON_BIN:-}" ]]; then
    python_bin="$PYTHON_BIN"
    python_selection_explicit="true"
elif [[ -n "$conda_prefix" && -x "$conda_prefix/bin/python" ]]; then
    # A nested virtualenv (for example .claudeapikey) can shadow `python`
    # even after `conda activate segvlad`; prefer the active Conda prefix.
    python_bin="$conda_prefix/bin/python"
else
    python_bin="python"
fi

usage() {
    cat <<'EOF'
Usage:
  ./run_vector_db.sh [SOURCE_DIR] [options]

The selection is terminal-only. SOURCE_DIR can be a 17places/custom root
containing ref/ and query/, or a directory containing arbitrary reference
images. For a plain image directory, also choose --target-dir.

Options:
  --source-dir PATH     Reference image directory or dataset root
  --target-dir PATH     Target/query image directory
  --query-dir PATH      Alias for --target-dir
  --output-dir PATH     Custom preprocessing/database output directory
  --ground-truth-csv P  CSV with correct source,target image pairs
  --result-csv PATH     Ranked source/target output CSV
  --metrics-csv PATH    Precision/recall/accuracy/F1 output CSV
  --evaluation-top-k N  Image ranks evaluated in CSV output (default: 5)
  --dataset VALUE       auto, 17places, or custom (default: auto)
  --vocab VALUE         auto, domain, or map (default: auto)
  --preprocess          Run SAM, DINO, and PCA when artifacts are missing
  --no-preprocess       Require existing precomputed artifacts
  --sam-checkpoint PATH SAM ViT-H checkpoint used during preprocessing
  --top-k N             Nearest segments per query (default: 200)
  --warmup N            Unmeasured benchmark passes (default: 1)
  --repeats N           Measured benchmark passes (default: 5)
  --python PATH         Python executable from the segvlad environment
  --conda-prefix PATH   Conda environment prefix (also accepted as --prefix)
  -h, --help            Show this help

Examples:
  ./run_vector_db.sh /data/17places
  ./run_vector_db.sh --source-dir /data/my_refs --target-dir /data/my_queries --preprocess
  ./run_vector_db.sh --dataset custom --source-dir /data/set/ref --target-dir /data/set/query --ground-truth-csv /data/set/ground_truth.csv
EOF
}

fail() {
    echo "Error: $*" >&2
    exit 1
}

has_images() {
    find "$1" -maxdepth 1 -type f \
        \( -iname '*.bmp' -o -iname '*.jpeg' -o -iname '*.jpg' \
        -o -iname '*.png' -o -iname '*.tif' -o -iname '*.tiff' \
        -o -iname '*.webp' \) -print -quit | grep -q .
}

need_value() {
    [[ $# -ge 2 && -n "$2" ]] || fail "$1 requires a value"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-dir) need_value "$@"; source_dir="$2"; shift 2 ;;
        --query-dir|--target-dir) need_value "$@"; query_dir="$2"; shift 2 ;;
        --output-dir) need_value "$@"; output_dir="$2"; shift 2 ;;
        --ground-truth-csv) need_value "$@"; ground_truth_csv="$2"; shift 2 ;;
        --result-csv) need_value "$@"; result_csv="$2"; shift 2 ;;
        --metrics-csv) need_value "$@"; metrics_csv="$2"; shift 2 ;;
        --evaluation-top-k) need_value "$@"; evaluation_top_k="$2"; shift 2 ;;
        --dataset) need_value "$@"; dataset="$2"; shift 2 ;;
        --vocab) need_value "$@"; vocab="$2"; shift 2 ;;
        --sam-checkpoint) need_value "$@"; sam_checkpoint="$2"; shift 2 ;;
        --top-k) need_value "$@"; top_k="$2"; shift 2 ;;
        --warmup) need_value "$@"; warmup="$2"; shift 2 ;;
        --repeats) need_value "$@"; repeats="$2"; shift 2 ;;
        --python) need_value "$@"; python_bin="$2"; python_selection_explicit="true"; shift 2 ;;
        --conda-prefix|--prefix) need_value "$@"; conda_prefix="$2"; python_bin="$conda_prefix/bin/python"; python_selection_explicit="true"; shift 2 ;;
        --preprocess) preprocess="yes"; shift ;;
        --no-preprocess) preprocess="no"; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) fail "unknown option: $1" ;;
        *)
            [[ -z "$source_dir" ]] || fail "only one source directory may be selected"
            source_dir="$1"
            shift
            ;;
    esac
done

[[ "$dataset" == "auto" || "$dataset" == "17places" || "$dataset" == "custom" ]] \
    || fail "--dataset must be auto, 17places, or custom"
[[ "$vocab" == "auto" || "$vocab" == "domain" || "$vocab" == "map" ]] \
    || fail "--vocab must be auto, domain, or map"
[[ "$top_k" =~ ^[1-9][0-9]*$ ]] || fail "--top-k must be a positive integer"
[[ "$warmup" =~ ^[0-9]+$ ]] || fail "--warmup must be a non-negative integer"
[[ "$repeats" =~ ^[1-9][0-9]*$ ]] || fail "--repeats must be a positive integer"
[[ "$evaluation_top_k" =~ ^[1-9][0-9]*$ ]] || fail "--evaluation-top-k must be a positive integer"

if [[ -z "$source_dir" ]]; then
    read -r -e -p "Reference images or dataset directory: " source_dir
fi
[[ -d "$source_dir" ]] || fail "source directory does not exist: $source_dir"
source_dir="$(cd -- "$source_dir" && pwd -P)"

dataset_root=""
if [[ -d "$source_dir/ref" && -d "$source_dir/query" ]]; then
    dataset_root="$source_dir"
    reference_dir="$source_dir/ref"
    [[ -n "$query_dir" ]] || query_dir="$source_dir/query"
elif [[ "$(basename -- "$source_dir")" == "ref" && -d "$source_dir/../query" ]]; then
    dataset_root="$(cd -- "$source_dir/.." && pwd -P)"
    reference_dir="$source_dir"
    [[ -n "$query_dir" ]] || query_dir="$dataset_root/query"
else
    reference_dir="$source_dir"
fi

if [[ "$dataset" == "auto" ]]; then
    if [[ -n "$dataset_root" && "$(basename -- "$dataset_root")" == "17places" ]]; then
        dataset="17places"
    else
        dataset="custom"
    fi
fi

if [[ "$dataset" == "17places" ]]; then
    [[ -n "$dataset_root" ]] || fail "17places selection must contain ref/ and query/"
    workdir_data="$(cd -- "$dataset_root/.." && pwd -P)"
    output_dir="$dataset_root/out"
else
    if [[ -z "$query_dir" ]]; then
        read -r -e -p "Query images directory: " query_dir
    fi
    [[ -d "$query_dir" ]] || fail "query directory does not exist: $query_dir"
    query_dir="$(cd -- "$query_dir" && pwd -P)"
    if [[ -z "$dataset_root" ]]; then
        dataset_root="$(cd -- "$reference_dir/.." && pwd -P)"
    fi
    [[ -n "$output_dir" ]] || output_dir="$dataset_root/revisit_out"
    workdir_data="$dataset_root"
fi

mkdir -p -- "$output_dir"
output_dir="$(cd -- "$output_dir" && pwd -P)"

if [[ "$vocab" == "auto" ]]; then
    if [[ -f "$output_dir/${dataset}_r_fitted_pca_model_order3_map.pkl" ]]; then
        vocab="map"
    else
        vocab="domain"
    fi
fi

if ! has_images "$reference_dir"; then
    fail "no reference images found in $reference_dir"
fi
if ! has_images "$query_dir"; then
    fail "no query images found in $query_dir"
fi
if [[ -n "$ground_truth_csv" ]]; then
    [[ -f "$ground_truth_csv" ]] || fail "ground-truth CSV does not exist: $ground_truth_csv"
    ground_truth_csv="$(cd -- "$(dirname -- "$ground_truth_csv")" && pwd -P)/$(basename -- "$ground_truth_csv")"
fi

command -v "$python_bin" >/dev/null 2>&1 || fail "Python executable not found: $python_bin"

required_python_modules=(
    cv2
    einops
    faiss
    fast_pytorch_kmeans
    h5py
    matplotlib
    natsort
    networkx
    pandas
    PIL
    psutil
    scipy
    sklearn
    timm
    torch
    torchvision
    tqdm
    transformers
    tyro
    utm
)

missing_python_modules() {
    "$1" - "${required_python_modules[@]}" <<'PY'
import importlib
import sys

for module in sys.argv[1:]:
    try:
        importlib.import_module(module)
    except Exception:
        print(module)
PY
}

missing_modules="$(missing_python_modules "$python_bin")"
if [[ -n "$missing_modules" && "$python_selection_explicit" == "false" ]] && command -v conda >/dev/null 2>&1; then
    segvlad_prefix="$(conda env list 2>/dev/null | awk '$1 == "segvlad" { print $NF; exit }')"
    if [[ -n "$segvlad_prefix" && -x "$segvlad_prefix/bin/python" ]]; then
        candidate_python="$segvlad_prefix/bin/python"
        candidate_missing="$(missing_python_modules "$candidate_python")"
        if [[ -z "$candidate_missing" ]]; then
            python_bin="$candidate_python"
            missing_modules=""
            echo "Using detected Conda environment: $segvlad_prefix"
        fi
    fi
fi

if [[ -n "$missing_modules" ]]; then
    echo "Python executable: $python_bin" >&2
    echo "Missing modules:" >&2
    while IFS= read -r module; do
        [[ -n "$module" ]] && echo "  - $module" >&2
    done <<< "$missing_modules"
    echo >&2
    echo "Create/activate the repository environment, then run this command again:" >&2
    echo "  conda env create -n segvlad -f $script_dir/segvlad.yaml" >&2
    echo "  conda activate segvlad" >&2
    echo "  \$CONDA_PREFIX/bin/python -m pip install opencv-python einops fast-pytorch-kmeans h5py matplotlib natsort networkx pandas psutil scipy scikit-learn timm tqdm transformers tyro utm" >&2
    echo "  \$CONDA_PREFIX/bin/python -m pip install -e $script_dir/sam" >&2
    echo >&2
    fail "the selected Python environment is missing SegVLAD dependencies"
fi

prefix="$dataset"
required_files=(
    "$output_dir/${prefix}_r_masks_320.h5"
    "$output_dir/${prefix}_q_masks_320.h5"
    "$output_dir/${prefix}_r_dino_640.h5"
    "$output_dir/${prefix}_q_dino_640.h5"
)
if [[ "$vocab" == "domain" ]]; then
    required_files+=("$output_dir/${prefix}_r_fitted_pca_model_order3.pkl")
else
    required_files+=("$output_dir/${prefix}_r_fitted_pca_model_order3_map.pkl")
fi

missing=()
for path in "${required_files[@]}"; do
    [[ -f "$path" ]] || missing+=("$path")
done

if [[ ${#missing[@]} -gt 0 && "$preprocess" == "ask" ]]; then
    echo "Missing ${#missing[@]} precomputed artifact(s)."
    read -r -p "Run SAM, DINO, and PCA preprocessing now? [y/N] " answer
    [[ "$answer" =~ ^[Yy]$ ]] && preprocess="yes" || preprocess="no"
fi
if [[ ${#missing[@]} -gt 0 && "$preprocess" == "no" ]]; then
    printf 'Missing: %s\n' "${missing[@]}" >&2
    fail "precomputed artifacts are required when --no-preprocess is used"
fi

common_env=(
    "REVISIT_WORKDIR=$workdir_data"
    "REVISIT_CUSTOM_DATASET_DIR=$dataset_root"
    "REVISIT_CUSTOM_REFERENCE_DIR=$reference_dir"
    "REVISIT_CUSTOM_QUERY_DIR=$query_dir"
    "REVISIT_CUSTOM_OUTPUT_DIR=$output_dir"
)

if [[ ${#missing[@]} -gt 0 && "$preprocess" == "yes" ]]; then
    if [[ -z "$sam_checkpoint" ]]; then
        default_checkpoint="$workdir_data/models/segment-anything/sam_vit_h_4b8939.pth"
        if [[ -f "$default_checkpoint" ]]; then
            sam_checkpoint="$default_checkpoint"
        else
            read -r -e -p "SAM ViT-H checkpoint: " sam_checkpoint
        fi
    fi
    [[ -f "$sam_checkpoint" ]] || fail "SAM checkpoint not found: $sam_checkpoint"
    common_env+=("REVISIT_SAM_CHECKPOINT=$sam_checkpoint")

    cd -- "$script_dir"
    env "${common_env[@]}" "$python_bin" place_rec_SAM_DINO.py --dataset "$dataset" --method SAM
    env "${common_env[@]}" "$python_bin" place_rec_SAM_DINO.py --dataset "$dataset" --method DINO
    env "${common_env[@]}" "$python_bin" place_rec_pca.py \
        --dataset "$dataset" \
        --experiment exp0_global_SegLoc_VLAD_PCA_o3 \
        --vocab-vlad "$vocab"
fi

echo "Dataset mode: $dataset"
echo "Reference images: $reference_dir"
echo "Query images: $query_dir"
echo "Output: $output_dir"
echo "Vocabulary: $vocab"

cd -- "$script_dir"
evaluation_args=(--evaluation-top-k "$evaluation_top_k")
[[ -n "$ground_truth_csv" ]] && evaluation_args+=(--ground-truth-csv "$ground_truth_csv")
[[ -n "$result_csv" ]] && evaluation_args+=(--result-csv "$result_csv")
[[ -n "$metrics_csv" ]] && evaluation_args+=(--metrics-csv "$metrics_csv")
env "${common_env[@]}" "$python_bin" place_rec_main.py \
    --dataset "$dataset" \
    --experiment exp0_global_SegLoc_VLAD_PCA_o3 \
    --vocab-vlad "$vocab" \
    --save-results \
    --save-vector-db \
    --benchmark-vector-db \
    --benchmark-top-k "$top_k" \
    --benchmark-warmup "$warmup" \
    --benchmark-repeats "$repeats" \
    "${evaluation_args[@]}"

artifact_name="${dataset}_exp0_global_SegLoc_VLAD_PCA_o3_${vocab}"
echo
echo "Database: $output_dir/vector_db/$artifact_name.faiss"
echo "Metadata: $output_dir/vector_db/$artifact_name.metadata.npz"
echo "Benchmark: $output_dir/vector_db/$artifact_name.benchmark.json"
echo "Results: ${result_csv:-$output_dir/vector_db/$artifact_name.results.csv}"
if [[ -n "$ground_truth_csv" || "$dataset" != "custom" ]]; then
    echo "Metrics: ${metrics_csv:-$output_dir/vector_db/$artifact_name.metrics.csv}"
else
    echo "Metrics: not written (provide --ground-truth-csv for a custom dataset)"
fi
