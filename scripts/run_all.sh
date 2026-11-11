#!/usr/bin/env bash
# Build, test, solve, run MPC, cross-check with IPOPT, plot. About 2 minutes.
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
(cd build && ctest --output-on-failure)
mkdir -p results
./build/ddp_experiments all --out results | tee results/experiments.txt
python3 scripts/crosscheck_casadi.py --results results | tee results/crosscheck.txt
python3 scripts/plot.py --results results
