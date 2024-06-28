#!/bin/bash

set -x

PYTORCH_VERSION=2.1.0

mkdir -p build/ios
cd build/ios

if [ ! -d "pytorch" ]; then
  git clone --branch v${PYTORCH_VERSION} --recursive https://github.com/pytorch/pytorch
else
  echo "PyTorch repository already cloned. If you changed the version you need to remove the pytorch folder and run this script again."
fi

cd pytorch
# remove previous build if it exists
rm -rf build_ios

if [ ! -d "build_ios_iphoneos_arm64" ]; then
  IOS_PLATFORM=OS IOS_ARCH=arm64 ./scripts/build_ios.sh
  mv build_ios build_ios_iphoneos_arm64
else
  echo "build_ios_iphoneos_arm64 already exists. Skipping build..."
fi

if [ ! -d "build_ios_iphonesimulator_x86_64" ]; then
  IOS_PLATFORM=SIMULATOR IOS_ARCH=x86_64 ./scripts/build_ios.sh
  mv build_ios build_ios_iphonesimulator_x86_64
else
  echo "build_ios_iphonesimulator_x86_64 already exists. Skipping build..."
fi

if [ ! -d "build_ios_iphonesimulator_arm64" ]; then
  IOS_PLATFORM=SIMULATOR IOS_ARCH=arm64 ./scripts/build_ios.sh
  mv build_ios build_ios_iphonesimulator_arm64
else
  echo "build_ios_iphonesimulator_arm64 already exists. Skipping build..."
fi

ARCH_BUILDS=(
  "build_ios_iphoneos_arm64"
  "build_ios_iphonesimulator_x86_64"
  "build_ios_iphonesimulator_arm64"
)

# merge libs using libtools into a single libtorch_lite.a
for build in ${ARCH_BUILDS[*]}
do
  libtool -static -o $build/libtorch_lite.a $(find $build/install/lib -name '*.a')
done

# merge the two libtorch_lite.a simulator libs using lipo
mkdir build_ios_iphonesimulator
lipo -create build_ios_iphonesimulator_x86_64/libtorch_lite.a build_ios_iphonesimulator_arm64/libtorch_lite.a -o build_ios_iphonesimulator/libtorch_lite.a 

# create xcframework
xcodebuild -create-xcframework \
  -library build_ios_iphonesimulator/libtorch_lite.a -headers build_ios_iphonesimulator_arm64/install/include \
  -library build_ios_iphoneos_arm64/libtorch_lite.a -headers build_ios_iphoneos_arm64/install/include \
  -output LibTorchLite.xcframework
