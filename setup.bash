#!/usr/bin/env bash
# Copyright (c) 2025 PettingLLMs Contributors
# SPDX-License-Identifier: MIT
#
# PettingLLMs Environment Setup Script
# 
# This script automates the setup of a Python virtual environment and installs
# all required dependencies for the PettingLLMs project, including PyTorch and
# flash-attention with proper build isolation handling.
#
# Usage:
#   bash setup.bash
#
# Requirements:
#   - Python 3.12
#   - Git (for submodule management)
#   - CUDA 12.8+ (for GPU support)
#   - At least 10GB free disk space

set -euo pipefail

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Configuration
readonly PYTHON_BIN="/usr/bin/python3.12"
readonly VENV_NAME="pettingllms_venv"
readonly PYTORCH_VERSION="2.7.1"
readonly TORCHVISION_VERSION="0.22.1"
readonly TORCHAUDIO_VERSION="2.7.1"
readonly FLASH_ATTN_VERSION="2.8.3"
readonly CUDA_VERSION="cu128"
readonly REQUIREMENTS_FILE="requirements_venv.txt"

# Utility functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

print_header() {
    echo ""
    echo "========================================"
    echo "$*"
    echo "========================================"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Python exists
    if [[ ! -x "${PYTHON_BIN}" ]]; then
        log_error "Python 3.12 not found at ${PYTHON_BIN}"
        log_info "Please install Python 3.12: sudo apt-get install python3.12-dev"
        exit 1
    fi
    
    # Check if git is available
    if ! command -v git &> /dev/null; then
        log_error "Git is not installed or not in PATH"
        log_info "Please install Git: sudo apt install git"
        exit 1
    fi
    
    local python_version
    python_version=$(${PYTHON_BIN} --version | awk '{print $2}')
    log_info "Found Python ${python_version} at ${PYTHON_BIN}"
    
    # Check if requirements.txt exists
    if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
        log_error "Requirements file not found: ${REQUIREMENTS_FILE}"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}
# Create virtual environment
setup_venv() {
    print_header "Step 1/6: Setting up virtual environment"
    
    log_info "Creating virtual environment: ${VENV_NAME}"
    rm -rf "${VENV_NAME}"
    ${PYTHON_BIN} -m venv "${VENV_NAME}"
    source "${VENV_NAME}/bin/activate"
    log_success "Virtual environment created and activated"
}
# Initialize and update git submodules and install verl
init_submodules() {
    print_header "Step 2/6: Initializing git submodules and installing verl"
    
    log_info "Initializing and updating git submodules..."
    git submodule update --init --recursive
    log_success "Git submodules updated successfully"
    
    log_info "Installing verl in editable mode..."
    cd verl
    pip install -e .
    cd ..
    log_success "Successfully installed verl"
}



# Upgrade pip tools
upgrade_pip() {
    print_header "Step 3/6: Upgrading pip tools"
    
    log_info "Upgrading pip, setuptools, and wheel..."
    python -m pip install --upgrade pip setuptools wheel --quiet
    log_success "pip tools upgraded"
}

# Install PyTorch
install_pytorch() {
    print_header "Step 4/6: Installing PyTorch"
    
    log_info "Installing PyTorch ${PYTORCH_VERSION} with CUDA ${CUDA_VERSION}"
    pip install \
        "torch==${PYTORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
        --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"
    log_success "PyTorch installation completed"
}

# Install flash-attn
install_flash_attn() {
    print_header "Step 5/6: Installing flash-attn"
    
    log_info "Installing flash-attn ${FLASH_ATTN_VERSION}"
    pip install ninja --quiet || true
    pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation
    log_success "flash-attn installation completed"
}

# Install remaining requirements
install_requirements() {
    print_header "Step 6/6: Installing project dependencies"
    
    log_info "Installing dependencies from ${REQUIREMENTS_FILE}"
    pip install -r "${REQUIREMENTS_FILE}"
    log_success "All dependencies installed successfully"
}

# Print completion message
print_completion() {
    print_header "Installation Complete"
    
    echo ""
    log_success "Environment setup completed successfully!"
    echo ""
    echo "To activate the virtual environment, run:"
    echo "  source ${VENV_NAME}/bin/activate"
    echo ""
    echo "To verify the installation, run:"
    echo "  python -c 'import torch; print(torch.__version__)'"
    echo ""
}
    echo "  source pettingllms_venv/bin/activate"

# Error handler
error_handler() {
    local exit_code=$?
    log_error "Setup failed with exit code ${exit_code}"
    echo ""
    echo "Common fixes:"
    echo "  sudo apt install python3.12 python3.12-venv git"
    echo "  git submodule update --init --recursive"
    echo ""
    exit "${exit_code}"
}

# Main function
main() {
    trap error_handler ERR
    
    print_header "PettingLLMs Environment Setup"
    check_prerequisites
    setup_venv
    init_submodules
    upgrade_pip
    install_pytorch
    install_flash_attn
    install_requirements
    pip install -e .
    print_completion
}

# Run main function
main "$@"

