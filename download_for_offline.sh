#!/usr/bin/env bash
# Run this script ON A MACHINE WITH INTERNET to prepare everything for offline use.
# Then copy the entire qwen_tts/ directory to the offline machine.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== 1. Downloading model to models/ ==="
mkdir -p models
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
        --local-dir models/Qwen3-TTS-12Hz-0.6B-Base
else
    echo "huggingface-cli not found. Install with: pip install huggingface_hub"
    echo "Or manually clone: git lfs clone https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base models/Qwen3-TTS-12Hz-0.6B-Base"
    exit 1
fi

echo "=== 2. Downloading pip wheels to wheels/ ==="
mkdir -p wheels
pip download -r requirements.txt -d wheels/ --python-version 3.11 --platform manylinux2014_x86_64 --only-binary=:all:
# Also grab numpy and typing_extensions
pip download numpy typing_extensions -d wheels/ --python-version 3.11 --platform manylinux2014_x86_64 --only-binary=:all:

echo ""
echo "=== Done! ==="
echo "Copy the entire directory to your offline machine."
echo "On the offline machine:"
echo "  - Run directly:  MODEL_PATH=models/Qwen3-TTS-12Hz-0.6B-Base uvicorn app:app --host 0.0.0.0 --port 8000"
echo "  - Or with Docker: edit Dockerfile to use the wheels/ offline install lines, then:"
echo "    docker build -t qwen-tts . && docker run --gpus all -p 8000:8000 qwen-tts"
