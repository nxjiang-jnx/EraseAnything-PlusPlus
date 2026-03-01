#!/bin/bash

VIDEO="samples/after_ablation/video_256px/prompt_0000.mp4"
OUT_DIR="samples/after_ablation/video_256px/frames"

[[ ! -f "$VIDEO" ]] && { echo "missing: $VIDEO"; exit 1; }
mkdir -p "$OUT_DIR"
ffmpeg -i "$VIDEO" -vf "select=between(n\,0\,128)" -vsync 0 "$OUT_DIR/frame_%04d.png"
