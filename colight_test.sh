#!/bin/bash

echo "=== CoLight Import Test ==="
cd "$(dirname "$0")"

conda activate traffic

echo "Testing CoLight import directly..."
python -c "
try:
    from agent.colight import CoLightAgent
    print('Direct CoLight import: SUCCESS')
    print('CoLightAgent class:', CoLightAgent)
except Exception as e:
    print('Direct CoLight import: FAILED')
    print('Error:', e)
    import traceback
    traceback.print_exc()
"

echo ""
echo "Testing agent module import..."
python -c "
try:
    import agent
    print('Agent module import: SUCCESS')
    from common.registry import Registry
    print('Registry agents:', list(Registry.mapping['model_mapping'].keys()))
    print('colight in registry:', 'colight' in Registry.mapping['model_mapping'])
except Exception as e:
    print('Agent module import: FAILED')
    print('Error:', e)
    import traceback
    traceback.print_exc()
"