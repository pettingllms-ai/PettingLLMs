#!/bin/bash
echo "=== PATH ==="
echo "$PATH"
echo ""
echo "=== which ninja ==="
which ninja 2>&1
echo ""
echo "=== which nvcc ==="
which nvcc 2>&1
echo ""
echo "=== ninja --version ==="
ninja --version 2>&1
echo ""
echo "=== python location ==="
which python3 2>&1
python3 --version 2>&1
echo ""
echo "=== flashinfer import test ==="
python3 -c "import flashinfer; print('flashinfer:', flashinfer.__version__)" 2>&1
echo ""
echo "=== flashinfer JIT cache ==="
ls -la /root/.cache/flashinfer/ 2>&1 || echo "/root/.cache/flashinfer/ not found"
ls -la ~/.cache/flashinfer/ 2>&1 || echo "~/.cache/flashinfer/ not found"
echo ""
echo "=== CUDA_HOME ==="
echo "$CUDA_HOME"
echo ""
echo "=== env vars ==="
env | grep -E "CONDA|VENV|VIRTUAL|PATH|CUDA|NINJA|FLASHINFER" | sort
echo ""
echo "=== whoami & home ==="
whoami 2>&1
echo "HOME=$HOME"
echo ""
echo "=== /usr/bin/ninja exists? ==="
ls -la /usr/bin/ninja 2>&1 || echo "not found"
echo ""
echo "=== pip show ninja ==="
pip show ninja 2>&1 | head -5
