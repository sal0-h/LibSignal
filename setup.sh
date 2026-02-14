#!/usr/bin/env bash
# ============================================================================
# LibSignal Setup Script
# Run this after cloning the repo on a fresh server:
#   chmod +x setup.sh && ./setup.sh
# ============================================================================
set -euo pipefail

CONDA_ENV="traffic"
PYTHON_VERSION="3.10"
SUMO_VERSION="1.26.0"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── 1. Install Miniconda if not present ─────────────────────────────────────
install_conda() {
    # Check if conda is on PATH or installed at common locations
    if command -v conda &>/dev/null; then
        info "Conda already installed: $(conda --version)"
        return 0
    fi

    # Check common install locations (conda may exist but not be on PATH yet)
    local conda_paths=("${HOME}/miniconda3" "${HOME}/anaconda3" "${HOME}/miniforge3" "/opt/conda")
    for cpath in "${conda_paths[@]}"; do
        if [ -f "${cpath}/bin/conda" ]; then
            info "Found existing conda at ${cpath} (not on PATH). Initializing..."
            eval "$("${cpath}/bin/conda" shell.bash hook)"
            conda init bash 2>/dev/null || true
            conda init zsh  2>/dev/null || true
            info "Conda initialized: $(conda --version)"
            return 0
        fi
    done

    info "Installing Miniconda..."
    local installer="Miniconda3-latest-Linux-x86_64.sh"
    curl -fsSL "https://repo.anaconda.com/miniconda/${installer}" -o "/tmp/${installer}"
    bash "/tmp/${installer}" -b -p "${HOME}/miniconda3"
    rm -f "/tmp/${installer}"

    # Init conda for current shell
    eval "$(${HOME}/miniconda3/bin/conda shell.bash hook)"
    conda init bash 2>/dev/null || true
    conda init zsh  2>/dev/null || true
    info "Miniconda installed at ${HOME}/miniconda3"
}

# ── 2. Create conda environment ────────────────────────────────────────────
create_env() {
    # Initialize conda for this shell session (required for non-interactive scripts)
    local conda_base
    conda_base="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
    source "${conda_base}/etc/profile.d/conda.sh"

    # Accept Anaconda TOS if required (conda >= 25.x)
    if conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main &>/dev/null; then
        true
    fi
    if conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r &>/dev/null; then
        true
    fi

    if conda env list | grep -qw "^${CONDA_ENV} "; then
        warn "Conda env '${CONDA_ENV}' already exists. Skipping creation."
    else
        info "Creating conda env '${CONDA_ENV}' with Python ${PYTHON_VERSION}..."
        conda create -y -n "${CONDA_ENV}" -c conda-forge python="${PYTHON_VERSION}"
    fi

    info "Activating '${CONDA_ENV}'..."
    conda activate "${CONDA_ENV}"
    info "Active Python: $(which python) ($(python --version))"
}

# ── 3. Install PyTorch (with CUDA if available) ────────────────────────────
install_pytorch() {
    info "Detecting GPU..."
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        # Get CUDA version from nvidia-smi
        local cuda_version
        cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        info "NVIDIA GPU detected (driver: ${cuda_version})"

        # Install PyTorch with CUDA 12.1 (broadly compatible)
        info "Installing PyTorch with CUDA support..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    else
        warn "No NVIDIA GPU detected. Installing CPU-only PyTorch."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
}

# ── 4. Install torch-geometric + torch-scatter ─────────────────────────────
install_torch_geometric() {
    info "Installing torch-geometric and torch-scatter..."
    # Let pyg find the right binary for our torch + cuda combo
    pip install torch-scatter torch-sparse -f "https://data.pyg.org/whl/torch-$(python -c 'import torch; print(torch.__version__.split("+")[0])')+$(python -c 'import torch; print("cu121" if torch.cuda.is_available() else "cpu")').html"
    pip install torch-geometric
}

# ── 5. Install SUMO packages (aligned versions) ────────────────────────────
install_sumo() {
    info "Installing SUMO packages (version ${SUMO_VERSION})..."
    # Install all SUMO packages at the same version to avoid conflicts
    pip install libsumo=="${SUMO_VERSION}" sumolib=="${SUMO_VERSION}" traci=="${SUMO_VERSION}" eclipse-sumo=="${SUMO_VERSION}"

    # Set SUMO_HOME to the pip-installed eclipse-sumo
    local sumo_home
    sumo_home=$(python -c "import os, sumolib; print(os.path.dirname(os.path.dirname(sumolib.__file__)))")
    info "SUMO_HOME will be set to: ${sumo_home}"

    # Persist SUMO_HOME in conda env activation
    mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
    mkdir -p "${CONDA_PREFIX}/etc/conda/deactivate.d"

    cat > "${CONDA_PREFIX}/etc/conda/activate.d/sumo_env.sh" << EOF
export SUMO_HOME="${sumo_home}"
EOF

    cat > "${CONDA_PREFIX}/etc/conda/deactivate.d/sumo_env.sh" << EOF
unset SUMO_HOME
EOF

    # Set it for the current session too
    export SUMO_HOME="${sumo_home}"
}

# ── 6. Install remaining Python dependencies ───────────────────────────────
install_deps() {
    info "Installing Python dependencies..."
    pip install \
        "gym>=0.21.0" \
        "gymnasium>=0.26.0" \
        "numpy<2.0" \
        "lmdb>=1.3.0" \
        "PyYAML>=6.0" \
        "pfrl>=0.3.0" \
        "stable-baselines3>=1.5.0" \
        "matplotlib>=3.5.0" \
        "pandas>=1.3.0" \
        "seaborn>=0.11.0" \
        "tqdm>=4.62.0" \
        "mpmath>=1.2.1" \
        "sympy>=1.10.1"
}

# ── 7. Verify installation ─────────────────────────────────────────────────
verify() {
    info "Verifying installation..."
    local failed=0

    python -c "import torch; print(f'  PyTorch {torch.__version__} — CUDA: {torch.cuda.is_available()}')" || { warn "PyTorch import failed"; failed=1; }
    python -c "import torch_geometric; print(f'  torch-geometric {torch_geometric.__version__}')" || { warn "torch-geometric import failed"; failed=1; }
    python -c "import pfrl; print(f'  pfrl {pfrl.__version__}')" || { warn "pfrl import failed"; failed=1; }
    python -c "import libsumo; print(f'  libsumo OK')" || { warn "libsumo import failed"; failed=1; }
    python -c "import sumolib; print(f'  sumolib OK')" || { warn "sumolib import failed"; failed=1; }
    python -c "import traci; print(f'  traci OK')" || { warn "traci import failed"; failed=1; }
    python -c "import gym; print(f'  gym {gym.__version__}')" || { warn "gym import failed"; failed=1; }
    python -c "import numpy; print(f'  numpy {numpy.__version__}')" || { warn "numpy import failed"; failed=1; }

    if [ "${failed}" -eq 0 ]; then
        info "All imports verified successfully!"
    else
        warn "Some imports failed — check warnings above."
    fi
}

# ── 8. Print summary ───────────────────────────────────────────────────────
print_summary() {
    echo ""
    echo "============================================="
    echo -e "${GREEN}  LibSignal setup complete!${NC}"
    echo "============================================="
    echo ""
    echo "  Conda env:  ${CONDA_ENV}"
    echo "  Activate:   conda activate ${CONDA_ENV}"
    echo ""
    echo "  Example run:"
    echo "    python run.py --task tsc --agent dqn --world sumo --network sumo1x1 --dataset onfly"
    echo ""
    echo "  With GPU:"
    echo "    python run.py --task tsc --agent presslight --world sumo --network sumo4x4 --ngpu 0 --dataset onfly"
    echo ""
    echo "============================================="
}

# ── Main ────────────────────────────────────────────────────────────────────
main() {
    info "Starting LibSignal setup..."
    echo ""

    install_conda
    create_env
    install_pytorch
    install_torch_geometric
    install_sumo
    install_deps
    verify
    print_summary
}

main "$@"
