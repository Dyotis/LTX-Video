#!/bin/bash

# Setup script for LTX-Video model weights
# This script downloads necessary model files and sets up the environment

set -e

echo "🚀 Setting up LTX-Video environment..."

# Create model directories
mkdir -p models
mkdir -p configs
mkdir -p /tmp

# Set proper permissions
chmod 755 models configs

# Check if we're in a CI environment (skip downloads to save space/time)
if [ "${CI:-false}" = "true" ] || [ "${GITHUB_ACTIONS:-false}" = "true" ]; then
    echo "🔧 CI environment detected - skipping model downloads during build"
    echo "📝 Models will be downloaded on first inference call"
    
    # Create placeholder files to satisfy any existence checks
    touch models/.gitkeep
    
    # Set up environment variables
    export MODEL_DIR="./models"
    export HF_HOME="./models"
    export TORCH_HOME="./models"
    
    echo "✅ CI setup complete - ready for runtime model download"
    exit 0
fi

# Check available disk space (only for non-CI environments)
AVAILABLE_SPACE=$(df /tmp | tail -1 | awk '{print $4}')
echo "Available disk space: ${AVAILABLE_SPACE}KB"

if [ "$AVAILABLE_SPACE" -lt 20971520 ]; then  # 20GB in KB
    echo "⚠️ Warning: Less than 20GB available. Model download may fail."
fi

# Download model weights with error handling and retries
download_with_retry() {
    local url=$1
    local output=$2
    local max_attempts=3
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo "Attempt $attempt/$max_attempts: Downloading $(basename $output)..."
        if wget --progress=bar:force:noscroll -O "$output" "$url"; then
            echo "✅ Downloaded: $(basename $output)"
            return 0
        else
            echo "❌ Download failed (attempt $attempt)"
            rm -f "$output"  # Remove partial download
            attempt=$((attempt + 1))
            if [ $attempt -le $max_attempts ]; then
                echo "Retrying in 5 seconds..."
                sleep 5
            fi
        fi
    done
    
    echo "❌ Failed to download $url after $max_attempts attempts"
    return 1
}

# Download 2B distilled model (recommended for most use cases)
if [ ! -f "models/ltxv-2b-0.9.6-distilled.safetensors" ]; then
    echo "📥 Downloading 2B distilled model (recommended)..."
    download_with_retry \
        "https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-2b-0.9.6-distilled-04-25.safetensors" \
        "models/ltxv-2b-0.9.6-distilled.safetensors"
else
    echo "✅ 2B distilled model already exists"
fi

# Download 2B regular model (fallback)
if [ ! -f "models/ltxv-2b-0.9.6-dev.safetensors" ]; then
    echo "📥 Downloading 2B dev model (fallback)..."
    download_with_retry \
        "https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-2b-0.9.6-dev-04-25.safetensors" \
        "models/ltxv-2b-0.9.6-dev.safetensors"
else
    echo "✅ 2B dev model already exists"
fi

# Verify downloaded models
echo "🔍 Verifying downloaded models..."
for model in models/*.safetensors; do
    if [ -f "$model" ]; then
        size=$(du -h "$model" | cut -f1)
        echo "  ✅ $(basename $model): $size"
    fi
done

# Check if we have at least one model
if [ ! -f "models/ltxv-2b-0.9.6-distilled.safetensors" ] && [ ! -f "models/ltxv-2b-0.9.6-dev.safetensors" ]; then
    echo "❌ Error: No model files downloaded successfully"
    exit 1
fi

# Verify configuration files exist
echo "🔍 Checking configuration files..."
if [ -d "configs" ] && [ "$(ls -A configs/*.yaml 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "✅ Configuration files found:"
    ls -la configs/*.yaml 2>/dev/null || echo "  No YAML configs found"
else
    echo "⚠️ Warning: No configuration files found in configs/ directory"
    echo "   Make sure to include the YAML config files from the original repository"
fi

# Set up environment variables
export MODEL_DIR="./models"
export HF_HOME="./models"
export TORCH_HOME="./models"

# Create a simple test script
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
import os
import torch
import sys

def test_setup():
    print("🧪 Testing setup...")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("⚠️ CUDA not available")
    
    # Check models
    model_dir = "./models"
    models = [f for f in os.listdir(model_dir) if f.endswith('.safetensors')]
    if models:
        print(f"✅ Found {len(models)} model file(s)")
        for model in models:
            size = os.path.getsize(os.path.join(model_dir, model)) / 1024**3
            print(f"   {model}: {size:.1f}GB")
    else:
        print("⚠️ No model files found - will download on first inference")
    
    # Check configs
    config_dir = "./configs"
    configs = [f for f in os.listdir(config_dir) if f.endswith('.yaml')] if os.path.exists(config_dir) else []
    if configs:
        print(f"✅ Found {len(configs)} config file(s)")
    else:
        print("⚠️ No config files found")
    
    return True

if __name__ == "__main__":
    success = test_setup()
    sys.exit(0 if success else 1)
EOF

python3 test_setup.py

# Set permissions
chmod +x predict.py
chmod +x test_setup.py

echo "🎬 LTX-Video setup complete!"
echo ""
echo "📊 Setup Summary:"
echo "  Models: $(ls -1 models/*.safetensors 2>/dev/null | wc -l) files"
echo "  Configs: $(ls -1 configs/*.yaml 2>/dev/null | wc -l) files"
echo "  GPU: $(python3 -c "import torch; print('Available' if torch.cuda.is_available() else 'Not available')")"
echo ""
echo "🚀 Ready for inference!"
