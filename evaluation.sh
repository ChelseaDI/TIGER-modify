#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p evaluation_log
datetime=$(date "+%Y%m%d-%H%M%S")
python -u evaluation.py "$@" > "./evaluation_log/${datetime}.log" 2>&1 &
echo "Evaluation PID: $!; log: ./evaluation_log/${datetime}.log"
