#!/bin/bash

export CUDA_VISIBLE_DEVICES="5"

# Choose config file
CONFIG="configs/erase_nudity.yaml"

python train_opensora_lora.py --config "$CONFIG"
