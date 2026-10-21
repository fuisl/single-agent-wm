#!/bin/bash
# Run scripts/lewm_memory.sh from a frozen copy, so editing the script cannot change a running job:
#   bash scripts/run_frozen.sh <log name> <stage ...>
set -euo pipefail
cd "$(dirname "$0")/.."
log=$1; shift
copy=runs/.frozen/lewm_memory_$(date +%Y%m%d_%H%M%S)_$log.sh
mkdir -p runs/.frozen && cp scripts/lewm_memory.sh $copy
LEWM_ROOT=$PWD bash $copy "$@" > logs/$log.log 2>&1
