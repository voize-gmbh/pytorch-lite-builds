# PyTorch-Lite builds

## Toolchain

PyTorch 2.1.0 does not configure with cmake >= 3.28, so cmake has to be pinned.
The easiest way to get a pinned toolchain without disturbing a system install is
a venv, which both build scripts pick up automatically if present:

```bash
python3 -m venv .venv-build
./.venv-build/bin/pip install 'cmake==3.27.7' ninja pyyaml typing_extensions setuptools
```

`pyyaml` and `typing_extensions` are needed by PyTorch's codegen, not just by
cmake.

## Patches

`patches/*.patch` are applied to the upstream `v2.1.0` tag by both build
scripts, in filename order, before configuring. Application is idempotent, so
re-running a script on an existing checkout is safe.

| Patch | What it does |
| --- | --- |
| `0001-mmap-tensor-storages-for-mobile-load.patch` | Adds `MobileModuleLoadOptions::USE_MMAP`, which memory-maps the model file and lets tensor storages alias the mapping instead of copying weights onto the heap. See below. |

To change a patch: edit the files under `build/ios/pytorch`, then regenerate
with `git -C build/ios/pytorch diff > patches/<name>.patch`.

### `USE_MMAP`

Opt in per load:

```cpp
uint64_t options = torch::jit::kDefaultMobileLoadOptions
                 | torch::jit::MobileModuleLoadOptions::USE_MMAP;
torch::jit::ExtraFilesMap extraFiles;
auto module = torch::jit::_load_for_mobile(path, c10::nullopt, extraFiles, options);
```

Only the filename overload can map; the `ReadAdapterInterface` overload reads
the whole file into memory before parsing, which would defeat the purpose.

Weight pages then stay clean and file-backed, so they do not count against the
process's physical footprint — the figure iOS compares against the per-process
memory limit. Records that were not stored uncompressed and 64-byte aligned
fall back to copying, so enabling the option is always safe; the benefit
depends on how the model was serialized. Models written by PyTorch qualify (its
writer stores records uncompressed and aligned precisely so that archives can
be mapped); a `.ptl` repacked by an ordinary zip tool may not.

The mapping is `PROT_READ`. An in-place write to a mapped weight therefore
faults rather than silently copying the page and giving the memory back.

## Building

### Fast iteration (simulator, arm64 only)

```bash
./scripts/build-ios-sim-fast.sh
./tests/run-mmap-test.sh path/to/model.ptl
```

Builds one slice and skips xcframework packaging; re-runs are incremental, so
an edit to a patched file recompiles just that translation unit. Not
releasable — use it while working on a patch.

`tests/run-mmap-test.sh` needs a booted simulator and, per model, checks that
every mappable record is byte-identical to what the copying reader returns,
then reports the physical footprint of a copying load against an mmapped one.

### Release XCFramework

```bash
./scripts/build-ios-xcframework.sh
```

Builds device arm64 plus both simulator slices and produces
`LibTorchLite.xcframework`. Each arch is skipped if its output directory
already exists, so delete those to force a rebuild.
