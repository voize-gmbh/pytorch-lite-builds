#!/bin/bash
#
# Builds tests/mmap_test.mm against the simulator-arm64 LibTorch-Lite produced
# by scripts/build-ios-sim-fast.sh and runs it on a booted simulator.
#
# Usage: tests/run-mmap-test.sh <model.ptl> [more-models.ptl ...]
#
# For each model it runs three checks:
#   verify - every uncompressed, aligned record is byte-identical between the
#            mmap-aliased path and the normal copying path
#   copy   - baseline physical footprint after a normal load
#   mmap   - physical footprint after an mmapped load
#
# Each mode runs in a fresh process: freed heap memory is not necessarily
# returned to the OS, so loading both ways in one process would understate the
# difference.

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 <model.ptl> [more-models.ptl ...]" >&2
  exit 2
fi

REPO_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
INSTALL="${REPO_ROOT}/build/ios/pytorch/build_ios/install"
BIN="${REPO_ROOT}/build/ios/mmap_test"

if [ ! -d "${INSTALL}/lib" ]; then
  echo "error: no simulator build found at ${INSTALL}." >&2
  echo "Run scripts/build-ios-sim-fast.sh first." >&2
  exit 1
fi

SDK="$(xcrun --sdk iphonesimulator --show-sdk-path)"

echo "Building test binary..."
# force_load on libtorch_cpu.a so the operator registrations, which are only
# referenced via static initializers, are not dropped by the linker.
clang++ -std=c++17 -O1 -g \
  -target arm64-apple-ios15.0-simulator -isysroot "${SDK}" \
  -I"${INSTALL}/include" \
  "${REPO_ROOT}/tests/mmap_test.mm" -o "${BIN}" \
  -Wl,-force_load,"${INSTALL}/lib/libtorch_cpu.a" \
  "${INSTALL}/lib/libc10.a" \
  "${INSTALL}/lib/libXNNPACK.a" \
  "${INSTALL}/lib/libpthreadpool.a" \
  "${INSTALL}/lib/libcpuinfo.a" \
  "${INSTALL}/lib/libclog.a" \
  "${INSTALL}/lib/libpytorch_qnnpack.a" \
  "${INSTALL}/lib/libeigen_blas.a" \
  "${INSTALL}/lib/libfmt.a" \
  -framework Accelerate -framework Foundation -framework CoreML \
  2>&1 | grep -E "error:" && { echo "build failed" >&2; exit 1; } || true

UDID="$(xcrun simctl list devices booted --json | python3 -c '
import json, sys
devices = json.load(sys.stdin)["devices"]
for runtime, entries in devices.items():
    if "iOS" not in runtime:
        continue
    for entry in entries:
        print(entry["udid"])
        raise SystemExit
')"

if [ -z "${UDID}" ]; then
  echo "error: no booted iOS simulator. Boot one with:" >&2
  echo "  xcrun simctl boot <udid> && xcrun simctl bootstatus <udid> -b" >&2
  exit 1
fi
echo "Using simulator ${UDID}"

# Set FORWARD_SAMPLES to also run a forward pass over a 1-D float input of that
# length, which checks that inference produces the same output either way and
# that the mmap advantage survives it (activation buffers grow the footprint in
# both modes; the weight saving should remain).
FORWARD_SAMPLES="${FORWARD_SAMPLES:-}"

status=0
for model in "$@"; do
  echo
  echo "===== $(basename "${model}")"
  for mode in verify copy mmap; do
    if [ "${mode}" != "verify" ] && [ -n "${FORWARD_SAMPLES}" ]; then
      xcrun simctl spawn "${UDID}" "${BIN}" "${model}" "${mode}" \
        "${FORWARD_SAMPLES}" || status=1
    else
      xcrun simctl spawn "${UDID}" "${BIN}" "${model}" "${mode}" || status=1
    fi
    echo "---"
  done
done

exit "${status}"
