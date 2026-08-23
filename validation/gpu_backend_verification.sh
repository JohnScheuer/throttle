#!/usr/bin/env bash
#
# GPU Backend Verification Script
#
# Requirements:
# - Linux machine with NVIDIA GPU
# - CUDA drivers installed
# - Python 3.11+
# - At least 16GB GPU memory for small models
#
# Usage:
#   ./gpu_backend_verification.sh [backend]
#   backend: vllm | sglang | lmdeploy | all (default: all)
#

set -euo pipefail

BACKEND="${1:-all}"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"  # Small model for testing
THROTTLE_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== GPU Backend Verification ==="
echo "Backend: $BACKEND"
echo "Model: $MODEL"
echo "Throttle repo: $THROTTLE_REPO_DIR"
echo ""

# Create test environment
TEST_DIR="/tmp/throttle-gpu-test"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

# Install throttle and test dependencies
python3 -m venv venv
. venv/bin/activate
pip install -q --upgrade pip
pip install -q httpx pytest
pip install -q -e "$THROTTLE_REPO_DIR"

test_vllm() {
    echo "=== Testing vLLM ==="

    # Install vLLM
    pip install -q vllm

    # Start vLLM server in background
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port 8100 \
        --dtype half \
        --max-model-len 512 \
        > vllm.log 2>&1 &

    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"

    # Wait for server to start
    echo "Waiting for vLLM server..."
    for i in {1..60}; do
        if curl -s http://localhost:8100/health > /dev/null 2>&1; then
            echo "vLLM server ready"
            break
        fi
        sleep 1
    done

    # Run integration tests
    echo "Running integration tests against vLLM..."
    cd "$THROTTLE_REPO_DIR"
    BACKEND_URL="http://localhost:8100" \
    BACKEND_MODEL="$MODEL" \
        python3 -m pytest tests/test_proxy_integration.py -v --tb=short

    # Cleanup
    kill $VLLM_PID 2>/dev/null || true

    echo "vLLM tests complete"
    echo ""
}

test_sglang() {
    echo "=== Testing SGLang ==="

    # Install SGLang
    pip install -q "sglang[all]"

    # Start SGLang server in background
    python3 -m sglang.launch_server \
        --model-path "$MODEL" \
        --port 8101 \
        --dtype float16 \
        --max-total-tokens 512 \
        > sglang.log 2>&1 &

    SGLANG_PID=$!
    echo "SGLang PID: $SGLANG_PID"

    # Wait for server to start
    echo "Waiting for SGLang server..."
    for i in {1..60}; do
        if curl -s http://localhost:8101/health > /dev/null 2>&1; then
            echo "SGLang server ready"
            break
        fi
        sleep 1
    done

    # Run integration tests
    echo "Running integration tests against SGLang..."
    cd "$THROTTLE_REPO_DIR"
    BACKEND_URL="http://localhost:8101" \
    BACKEND_MODEL="$MODEL" \
        python3 -m pytest tests/test_proxy_integration.py -v --tb=short

    # Cleanup
    kill $SGLANG_PID 2>/dev/null || true

    echo "SGLang tests complete"
    echo ""
}

test_lmdeploy() {
    echo "=== Testing LMDeploy ==="

    # Install LMDeploy
    pip install -q lmdeploy

    # Start LMDeploy server in background
    lmdeploy serve api_server \
        "$MODEL" \
        --server-port 8102 \
        --dtype float16 \
        --max-total-token-num 512 \
        > lmdeploy.log 2>&1 &

    LMDEPLOY_PID=$!
    echo "LMDeploy PID: $LMDEPLOY_PID"

    # Wait for server to start
    echo "Waiting for LMDeploy server..."
    for i in {1..60}; do
        if curl -s http://localhost:8102/health > /dev/null 2>&1; then
            echo "LMDeploy server ready"
            break
        fi
        sleep 1
    done

    # Run integration tests
    echo "Running integration tests against LMDeploy..."
    cd "$THROTTLE_REPO_DIR"
    BACKEND_URL="http://localhost:8102" \
    BACKEND_MODEL="$MODEL" \
        python3 -m pytest tests/test_proxy_integration.py -v --tb=short

    # Cleanup
    kill $LMDEPLOY_PID 2>/dev/null || true

    echo "LMDeploy tests complete"
    echo ""
}

# Run tests
case "$BACKEND" in
    vllm)
        test_vllm
        ;;
    sglang)
        test_sglang
        ;;
    lmdeploy)
        test_lmdeploy
        ;;
    all)
        test_vllm
        test_sglang
        test_lmdeploy
        ;;
    *)
        echo "Unknown backend: $BACKEND"
        echo "Usage: $0 [vllm|sglang|lmdeploy|all]"
        exit 1
        ;;
esac

echo "=== All tests complete ==="
