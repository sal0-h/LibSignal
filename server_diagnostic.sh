#!/bin/bash

echo "=== LibSignal Server Diagnostic Script ==="
echo "Date: $(date)"
echo "User: $(whoami)"
echo "Host: $(hostname)"
echo ""

# Check conda environment
echo "=== Conda Environment ==="
if command -v conda &> /dev/null; then
    echo "Conda available"
    conda info --envs
    echo "Current environment: $CONDA_DEFAULT_ENV"
else
    echo "Conda not found!"
fi
echo ""

# Check Python and key packages
echo "=== Python Environment ==="
python --version
python -c "
import sys
print('Python executable:', sys.executable)
try:
    import torch
    print('PyTorch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU count:', torch.cuda.device_count())
except ImportError as e:
    print('PyTorch import failed:', e)

try:
    import libsumo
    print('libsumo: OK')
except ImportError as e:
    print('libsumo import failed:', e)

try:
    import traci
    print('traci: OK')
except ImportError as e:
    print('traci import failed:', e)

try:
    import torch_geometric
    print('torch_geometric: OK')
except ImportError as e:
    print('torch_geometric import failed:', e)
"
echo ""

# Check system libraries
echo "=== System Libraries ==="
libs=("libGL.so.1" "libglib-2.0.so.0" "libgthread-2.0.so.0" "libgtk-3.so.0")
for lib in "${libs[@]}"; do
    if ldconfig -p | grep -q "$lib"; then
        echo "$lib: Found"
    else
        echo "$lib: MISSING"
    fi
done
echo ""

# Check SUMO
echo "=== SUMO Installation ==="
if command -v sumo &> /dev/null; then
    echo "SUMO binary found: $(which sumo)"
    sumo --version
else
    echo "SUMO binary not found"
fi

if command -v sumo-gui &> /dev/null; then
    echo "SUMO GUI found: $(which sumo-gui)"
else
    echo "SUMO GUI not found"
fi

# Check SUMO_HOME
echo "SUMO_HOME: $SUMO_HOME"
if [ -z "$SUMO_HOME" ]; then
    echo "SUMO_HOME not set!"
else
    if [ -d "$SUMO_HOME" ]; then
        echo "SUMO_HOME directory exists"
        ls -la "$SUMO_HOME" | head -10
    else
        echo "SUMO_HOME directory does not exist!"
    fi
fi
echo ""

# Try to run a simple test
echo "=== Quick LibSignal Test ==="
# cd /path/to/LibSignal  # Replace with actual path
python -c "
try:
    from common.registry import Registry
    import agent
    print('Registry agents:', list(Registry.mapping['model_mapping'].keys()))
    print('colight available:', 'colight' in Registry.mapping['model_mapping'])
    print('Basic import test: PASSED')
except Exception as e:
    print('Basic import test: FAILED -', e)
" 2>&1

echo ""
echo "=== Diagnostic Complete ==="