#!/bin/bash

# Exit on error
set -e

# Function to check if CUDA is available
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA GPU detected"
        return 0
    else
        echo "No CUDA GPU detected"
        return 1
    fi
}

# Function to check if conda is available
check_conda() {
    if command -v conda &> /dev/null; then
        echo "Conda is available"
        return 0
    else
        echo "Conda is not installed. Please install Conda first."
        return 1
    fi
}

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print step with color
print_step() {
    echo -e "${BLUE}[Step] ${1}${NC}"
}

# Main installation process
main() {
    # Check prerequisites
    check_conda || exit 1
    
    # Create and activate conda environment
    # if not exists, create it
    #if ! conda env list | grep -q "pettingllms"; then
    #    print_step "Creating conda environment 'pettingllms' with Python 3.12..."
    #    conda create -n pettingllms python=3.12 -y
    #else
    #    print_step "Conda environment 'pettingllms' already exists"
    #fi
    conda create -n pettingllms python=3.12 -y

   
    
    # Need to source conda for script environment
    eval "$(conda shell.bash hook)"
    conda activate pettingllms

    # Install comprehensive base runtime first to resolve all transient dependency conflicts
    print_step "Installing comprehensive base runtime dependencies..."
    python -m pip install "attrs>=22.2.0" "numpy<2.0.0" scipy "packaging>=20.0" \
        "protobuf>=3.20.0,<5.0" "pydantic>=2.5.2,<3" pillow "typing-extensions>=4.4.0" \
        jinja2 requests "tqdm>=4.66.3" filelock "fsspec[http]<=2025.3.0,>=2023.1.0" \
        "huggingface-hub>=0.24.0" "pyyaml>=5.1" ipython "llama-index-core>=0.12.0,<0.13.0" \
        "markdown-it-py>=2.2.0" "openai>=1.68.2,<1.76.0" "tokenizers>=0.15,<1" \
        "together>=0.21.0" "typer>=0.12.3" "distro>=1.7.0,<2" \
        "llama-index-llms-openai>=0.3.0,<0.4.0"

    # Upgrade pip tooling after base dependencies are in place
    print_step "Upgrading pip/setuptools/wheel..."
    python -m pip install -U pip setuptools wheel
    
    # Install PyTorch with CUDA if available
    if check_cuda; then
        print_step "CUDA detected, checking CUDA version..."
        
        if command -v nvcc &> /dev/null; then
            nvcc_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
            nvcc_major=$(echo $nvcc_version | cut -d. -f1)
            nvcc_minor=$(echo $nvcc_version | cut -d. -f2)
            
            print_step "Found NVCC version: $nvcc_version"
            
            # Require CUDA >= 12.4 to match the chosen PyTorch/cu124 wheels
            if [[ "$nvcc_major" -gt 12 || ("$nvcc_major" -eq 12 && "$nvcc_minor" -ge 4) ]]; then
                print_step "CUDA $nvcc_version is already installed and meets requirements (>=12.4)"
                export CUDA_HOME=${CUDA_HOME:-$(dirname $(dirname $(which nvcc)))}
            else
                print_step "CUDA version < 12.4, installing CUDA toolkit 12.4..."
                conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit -y
                export CUDA_HOME=$CONDA_PREFIX
            fi
        else
            print_step "NVCC not found, installing CUDA toolkit 12.4..."
            conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit -y
            export CUDA_HOME=$CONDA_PREFIX
        fi
        
        print_step "Installing PyTorch with CUDA support (cu124)..."
        pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

        # flash-attn prefers being installed right after torch with no build isolation
        print_step "Installing flash-attention..."
        pip install flash-attn==2.7.4.post1 --no-build-isolation
    else
        print_step "Installing PyTorch without CUDA support..."
        pip install torch==2.6.0
    fi
    
    # Install all project requirements after PyTorch is ready to avoid resolver noise


    # Initialize and install verl (with dependencies) after requirements are in place

    
    print_step "Installing project requirements..."
    pip install -r requirements.txt

    print_step "Setting up verl submodule and installing..."
    git submodule init
    git submodule update
    pushd verl >/dev/null
    pip install -e .
    popd >/dev/null

    # Install pettingllms package in editable mode
    print_step "Installing pettingllms package..."
    pip install -e .

    # Validate environment consistency
    print_step "Validating environment (pip check)..."
    pip check | cat

    echo -e "${GREEN}Installation completed successfully!${NC}"
    echo "To activate the environment, run: conda activate pettingllms"
    
    # export CMAKE_POLICY_VERSION_MINIMUM=3.5 && pip install alfworld[full]
    # alfworld-download
}

# Run main installation
main