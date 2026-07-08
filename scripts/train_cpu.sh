#!/usr/bin/env bash
set -euo pipefail
python -m baremetal_llm.cli demo --all --cpu "$@"
