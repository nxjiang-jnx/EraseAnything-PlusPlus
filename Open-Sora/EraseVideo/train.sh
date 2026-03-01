#!/bin/bash

# Training script for Open-Sora video concept erasure

# Set CUDA device
export CUDA_VISIBLE_DEVICES="5"

# Choose config file
CONFIG="configs/erase_trump.yaml"

echo "Starting Open-Sora LoRA training for concept erasure..."
echo "Config: $CONFIG"
echo "========================================"

python train_opensora_lora.py --config "$CONFIG"

echo "========================================"
echo "Training completed!"
echo "Check outputs in the directory specified in config"

