#!/usr/bin/env bash
set -euo pipefail
PART="${1:?Usage: train_gpu.sh <part_number>}"
python -m baremetal_llm.cli train --part "$PART" --gpu
