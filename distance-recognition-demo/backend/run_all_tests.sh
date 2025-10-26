#!/bin/bash
echo "========================================"
echo "RUNNING ALL TESTS"
echo "========================================"

python test_smoke.py && \
python test_data_formats.py && \
python test_api.py && \
echo "" && \
echo "========================================"
echo "✅ ALL TESTS PASSED"
echo "========================================"