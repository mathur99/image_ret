#!/usr/bin/env bash
set -e

echo "=== Image Retrieval System — Setup ==="

# 1. install dependencies
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "Installing Python dependencies..."
uv sync

# 2. unzip image data
if [ -f "data.zip" ]; then
    if [ -d "data/database" ] && [ "$(ls -A data/database 2>/dev/null)" ]; then
        echo "data/database/ already exists — skipping unzip."
    else
        echo "Extracting images from data.zip..."
        unzip -o data.zip
        echo "Extracted to data/database/ and data/query/"
    fi
else
    echo "Warning: data.zip not found — no images to extract."
    echo "  Place your images manually in data/database/ and data/query/"
fi

# 3. create output directory
mkdir -p data/index

echo ""
echo "Done! Edit config.yaml and run:"
echo "  uv run image-ret"
