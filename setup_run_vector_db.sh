#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
env_name="${SEGVLAD_ENV_NAME:-segvlad}"
conda_bin="${CONDA_EXE:-conda}"
conda_prefix="${SEGVLAD_CONDA_PREFIX:-}"

runtime_packages=(
    opencv-python
    einops
    fast-pytorch-kmeans
    h5py
    matplotlib
    natsort
    networkx
    pandas
    psutil
    scipy
    scikit-learn
    timm
    tqdm
    transformers
    tyro
    utm
)

usage() {
    cat <<'EOF'
Usage:
  ./setup_run_vector_db.sh [SOURCE_DIR] [options]

Creates or repairs the segvlad environment, installs the repository's
PyTorch build and missing dependencies, installs the local SAM package,
then runs run_vector_db.sh with the same arguments.

All options are passed to run_vector_db.sh. Use --help there for the full
dataset/source/preprocessing option list.

Setup-only options:
  --conda-prefix PATH  Install/use this Conda prefix (also --prefix)

Environment variables:
  SEGVLAD_ENV_NAME  Conda environment name (default: segvlad)
  SEGVLAD_CONDA_PREFIX  Conda environment prefix path (overrides the name)
  CONDA_EXE         Conda executable (default: conda)
EOF
}

fail() {
    echo "Error: $*" >&2
    exit 1
}

need_value() {
    [[ $# -ge 2 && -n "$2" ]] || fail "$1 requires a value"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

# Consume setup-only prefix options and forward every other option unchanged
# to run_vector_db.sh.
forward_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda-prefix|--prefix)
            need_value "$@"
            conda_prefix="$2"
            shift 2
            ;;
        *)
            forward_args+=("$1")
            shift
            ;;
    esac
done

for setup_arg in "${forward_args[@]}"; do
    if [[ "$setup_arg" == "-h" || "$setup_arg" == "--help" ]]; then
        usage
        exit 0
    fi
done

if [[ -n "$conda_prefix" && "$conda_prefix" != /* ]]; then
    conda_prefix="$PWD/$conda_prefix"
fi

command -v "$conda_bin" >/dev/null 2>&1 || fail "Conda was not found"

if [[ -n "$conda_prefix" ]]; then
    env_prefix="$conda_prefix"
    if [[ ! -x "$env_prefix/bin/python" ]]; then
        echo "Creating Conda environment at prefix: $env_prefix"
        "$conda_bin" env create --prefix "$env_prefix" -f "$script_dir/segvlad.yaml"
    fi
else
    env_prefix="$("$conda_bin" env list 2>/dev/null | awk -v wanted="$env_name" '$1 == wanted { print $NF; exit }')"
    if [[ -z "$env_prefix" || ! -x "$env_prefix/bin/python" ]]; then
        echo "Creating Conda environment: $env_name"
        "$conda_bin" env create -n "$env_name" -f "$script_dir/segvlad.yaml"
        env_prefix="$("$conda_bin" env list 2>/dev/null | awk -v wanted="$env_name" '$1 == wanted { print $NF; exit }')"
    fi
fi

[[ -x "$env_prefix/bin/python" ]] || fail "Could not locate Conda Python executable: $env_prefix/bin/python"
python_bin="$env_prefix/bin/python"

# Install the pinned PyTorch build before packages that declare torch as a
# dependency. This prevents a fresh environment from silently receiving an
# incompatible/latest torch wheel.
if ! "$python_bin" -c '
import torch
import torchvision
assert torch.__version__.startswith("1.11.0+cu113")
assert torchvision.__version__.startswith("0.12.0+cu113")
' >/dev/null 2>&1; then
    echo "Installing the repository CUDA 11.3 PyTorch build"
    "$python_bin" -m pip install \
        torch==1.11.0+cu113 \
        torchvision==0.12.0+cu113 \
        torchaudio==0.11.0 \
        --extra-index-url https://download.pytorch.org/whl/cu113
fi

missing_modules="$("$python_bin" -c '
import importlib
modules = ("cv2", "einops", "faiss", "fast_pytorch_kmeans", "h5py", "matplotlib", "natsort", "networkx", "pandas", "PIL", "psutil", "scipy", "sklearn", "timm", "torch", "torchvision", "tqdm", "transformers", "tyro", "utm")
for module in modules:
    try:
        importlib.import_module(module)
    except Exception:
        print(module)
')"

if [[ -n "$missing_modules" ]]; then
    echo "Installing missing runtime packages into $env_prefix"
    "$python_bin" -m pip install "${runtime_packages[@]}"
fi

echo "Installing the local SAM package"
"$python_bin" -m pip install -e "$script_dir/sam"

echo "Verifying imports"
"$python_bin" -c 'import cv2, einops, faiss, fast_pytorch_kmeans, h5py, matplotlib, natsort, networkx, pandas, PIL, psutil, scipy, sklearn, timm, torch, torchvision, tqdm, transformers, tyro, utm; print("SegVLAD environment OK")'

echo "Starting vector database runner"
exec "$script_dir/run_vector_db.sh" --python "$python_bin" "${forward_args[@]}"
