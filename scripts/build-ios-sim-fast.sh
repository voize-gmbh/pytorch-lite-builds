#!/bin/bash
#
# Fast iteration loop: builds LibTorch-Lite for the iOS simulator on Apple
# Silicon only (arm64), skipping the device and x86_64 slices and the
# xcframework packaging that build-ios-xcframework.sh does.
#
# Use this while working on a patch under patches/. The full script is what
# produces a releasable xcframework; this one just gets you something you can
# link a test binary against, and it rebuilds incrementally on re-runs.
#
# Requires cmake < 3.28 (PyTorch 2.1.0 does not build with newer cmake). The
# repo's .venv-build provides a pinned 3.27.7 -- see README.

set -euo pipefail

PYTORCH_VERSION=2.1.0

REPO_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
PATCHES_DIR="${REPO_ROOT}/patches"
VENV="${REPO_ROOT}/.venv-build"

if [ -d "${VENV}" ]; then
  export PATH="${VENV}/bin:${PATH}"
  export PYTHON="${VENV}/bin/python"
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "error: cmake not found. Create the pinned build venv first:" >&2
  echo "  python3 -m venv ${VENV}" >&2
  echo "  ${VENV}/bin/pip install 'cmake==3.27.7' ninja pyyaml typing_extensions setuptools" >&2
  exit 1
fi

CMAKE_VERSION="$(cmake --version | head -1 | awk '{print $3}')"
echo "Using cmake ${CMAKE_VERSION} ($(command -v cmake))"

mkdir -p "${REPO_ROOT}/build/ios"
cd "${REPO_ROOT}/build/ios"

if [ ! -d "pytorch" ]; then
  git clone --branch "v${PYTORCH_VERSION}" --recursive --depth 1 \
    --shallow-submodules https://github.com/pytorch/pytorch
fi

cd pytorch

for patch in "${PATCHES_DIR}"/*.patch; do
  [ -e "$patch" ] || continue
  if git apply --reverse --check "$patch" 2>/dev/null; then
    echo "Patch already applied, skipping: $(basename "$patch")"
  else
    echo "Applying patch: $(basename "$patch")"
    git apply --verbose "$patch"
  fi
done

# Incremental when build_ios already exists, so an edit to a patched file
# recompiles just that translation unit instead of the whole tree.
if [ -d build_ios ] && [ -f build_ios/CMakeCache.txt ]; then
  echo "Reusing existing build_ios (incremental build)"
  cd build_ios
  make install -j"$(sysctl -n hw.ncpu)"
else
  IOS_PLATFORM=SIMULATOR IOS_ARCH=arm64 ./scripts/build_ios.sh
fi

echo
echo "Static libs: $(cd "${REPO_ROOT}/build/ios/pytorch" && pwd)/build_ios/install/lib"
echo "Headers:     $(cd "${REPO_ROOT}/build/ios/pytorch" && pwd)/build_ios/install/include"
