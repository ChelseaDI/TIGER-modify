#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p train_log
datetime=$(date "+%Y%m%d-%H%M%S")
# Forward CLI options, including the optional EXP1 preference switch.
python -u main.py "$@" > "./train_log/${datetime}.log" 2>&1 &
echo "Training PID: $!; log: ./train_log/${datetime}.log"
