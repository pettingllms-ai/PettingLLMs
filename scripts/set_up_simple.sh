#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print step with color
print_step() {
    echo -e "${BLUE}[Step] ${1}${NC}"
}

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

print_step "Setting up PettingLLMs in current Python environment..."

# Upgrade pip first
print_step "Upgrading pip/setuptools/wheel..."
python -m pip install -U pip setuptools wheel

# Install PyTorch first based on CUDA availability
if check_cuda; then
    print_step "Installing PyTorch with CUDA support (cu124)..."
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    
    print_step "Installing flash-attention..."
    pip install flash-attn==2.7.4.post1 --no-build-isolation
else
    print_step "Installing PyTorch without CUDA support..."
    pip install torch==2.6.0
fi

# Install all requirements at once to let pip resolve dependencies
print_step "Installing all project requirements..."
pip install -r requirements.txt

# Initialize and install verl submodule
print_step "Setting up verl submodule..."
git submodule init
git submodule update
pushd verl >/dev/null
pip install -e .
popd >/dev/null

# Install pettingllms package
print_step "Installing pettingllms package..."
pip install -e .

# Final validation
print_step "Validating environment..."
pip check | cat || echo "Some dependency conflicts may remain but should not affect functionality"

echo -e "${GREEN}Installation completed!${NC}"
echo "Note: If you see dependency conflicts, they are usually warnings and won't prevent the code from running."
