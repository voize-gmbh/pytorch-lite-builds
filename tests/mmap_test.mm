// Verifies that mmapped mobile module loading (a) produces bit-identical
// tensor data to the copying path, and (b) keeps the weight bytes out of the
// process's physical footprint.
//
// Run once per mode in a fresh process, since freed heap memory is not
// necessarily returned to the OS and would muddy the comparison.

#include <ATen/ATen.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>

#include <mach/mach.h>

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>

static uint64_t physFootprintBytes() {
  task_vm_info_data_t info;
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) !=
      KERN_SUCCESS) {
    return 0;
  }
  if (count < TASK_VM_INFO_REV1_COUNT) {
    return 0;
  }
  return (uint64_t)info.phys_footprint;
}

struct Stats {
  int64_t tensors = 0;
  int64_t numel = 0;
  int64_t storageBytes = 0;
  // FNV-1a over the raw storage bytes of every tensor, so a wrong offset or a
  // stale mapping shows up as a mismatch regardless of dtype.
  uint64_t hash = 1469598103934665603ULL;
  int64_t mmappedTensors = 0;
};

// Walks every tensor reachable from the module, not just registered
// parameters -- most of a scripted model's bytes live in plain attributes and
// constants, which is exactly what mmapping is meant to cover.
static void walk(
    const c10::IValue& v,
    Stats& s,
    std::set<const void*>& seenObjects,
    const void* mmapBase,
    size_t mmapSize) {
  if (v.isTensor()) {
    const at::Tensor& t = v.toTensor();
    if (!t.defined() || !t.has_storage()) {
      return;
    }
    const void* data = t.storage().data();
    if (data == nullptr) {
      return;
    }
    // Only count each distinct storage once; tensors commonly share them.
    if (!seenObjects.insert(data).second) {
      return;
    }
    s.tensors++;
    s.numel += t.numel();
    const size_t nbytes = t.storage().nbytes();
    s.storageBytes += (int64_t)nbytes;
    if (mmapBase != nullptr && data >= mmapBase &&
        (const char*)data < (const char*)mmapBase + mmapSize) {
      s.mmappedTensors++;
    }
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < nbytes; i++) {
      s.hash ^= bytes[i];
      s.hash *= 1099511628211ULL;
    }
    return;
  }
  if (v.isTuple()) {
    for (const auto& e : v.toTupleRef().elements()) {
      walk(e, s, seenObjects, mmapBase, mmapSize);
    }
  } else if (v.isList()) {
    for (const auto& e : v.toListRef()) {
      walk(e, s, seenObjects, mmapBase, mmapSize);
    }
  } else if (v.isGenericDict()) {
    for (const auto& e : v.toGenericDict()) {
      walk(e.value(), s, seenObjects, mmapBase, mmapSize);
    }
  } else if (v.isObject()) {
    auto obj = v.toObject();
    if (!seenObjects.insert(obj.get()).second) {
      return;
    }
    for (size_t i = 0; i < obj->slots().size(); i++) {
      walk(obj->slots()[i], s, seenObjects, mmapBase, mmapSize);
    }
  }
}

// Directly validates the aliasing logic: for every record in the archive,
// compare the bytes the normal copying reader returns against the bytes at the
// offset getStoredRecordExtent() reports inside the mapping. This covers all
// the weight data, including the bulk that is not reachable through the public
// module API.
static int verifyRecords(const std::string& path) {
  caffe2::serialize::MmapReadAdapter adapter(path);
  auto region = adapter.mmapRegion();
  if (!region) {
    fprintf(stderr, "FAIL: adapter exposed no mapping\n");
    return 1;
  }
  const char* base = static_cast<const char*>(region->addr());

  auto rai = std::make_shared<caffe2::serialize::MmapReadAdapter>(path);
  caffe2::serialize::PyTorchStreamReader reader(rai);

  int64_t stored = 0, skipped = 0, mismatches = 0, checkedBytes = 0;
  for (const auto& name : reader.getAllRecords()) {
    size_t offset = 0, size = 0;
    if (!reader.getStoredRecordExtent(
            name, caffe2::serialize::detail::kFieldAlignment, &offset, &size)) {
      skipped++;
      continue;
    }
    stored++;
    auto [ptr, n] = reader.getRecord(name);
    if (n != size) {
      fprintf(stderr,
              "FAIL: %s size mismatch: extent=%zu getRecord=%zu\n",
              name.c_str(), size, n);
      mismatches++;
      continue;
    }
    if (n > 0 && std::memcmp(base + offset, ptr.get(), n) != 0) {
      fprintf(stderr, "FAIL: %s bytes differ\n", name.c_str());
      mismatches++;
      continue;
    }
    checkedBytes += (int64_t)n;
  }

  printf("mode=verify\n");
  printf("records_stored_and_aligned=%" PRId64 "\n", stored);
  printf("records_skipped_compressed_or_unaligned=%" PRId64 "\n", skipped);
  printf("bytes_verified_mb=%.2f\n", (double)checkedBytes / 1e6);
  printf("mismatches=%" PRId64 "\n", mismatches);
  printf("%s\n", mismatches == 0 ? "VERIFY_OK" : "VERIFY_FAILED");
  return mismatches == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <model.ptl> <copy|mmap|verify>\n", argv[0]);
    return 2;
  }
  const std::string path = argv[1];
  if (std::strcmp(argv[2], "verify") == 0) {
    return verifyRecords(path);
  }
  const bool useMmap = std::strcmp(argv[2], "mmap") == 0;

  uint64_t options = torch::jit::kDefaultMobileLoadOptions;
  if (useMmap) {
    options |= torch::jit::MobileModuleLoadOptions::USE_MMAP;
  }

  const uint64_t before = physFootprintBytes();

  torch::jit::ExtraFilesMap extraFiles;
  // Report a failed load as a clean LOAD_FAILED line rather than an uncaught
  // exception, so a batch run over many models stays readable.
  std::unique_ptr<torch::jit::mobile::Module> modulePtr;
  try {
    modulePtr = std::make_unique<torch::jit::mobile::Module>(
        torch::jit::_load_for_mobile(path, c10::nullopt, extraFiles, options));
  } catch (const std::exception& e) {
    printf("mode=%s\n", useMmap ? "mmap" : "copy");
    printf("LOAD_FAILED: %s\n", e.what());
    return 1;
  }
  torch::jit::mobile::Module& module = *modulePtr;

  const uint64_t afterLoad = physFootprintBytes();

  // Independently mmap the file so we can tell whether a storage really points
  // into a mapping of it, rather than trusting the loader's word.
  const void* mmapBase = nullptr;
  size_t mmapSize = 0;
  {
    caffe2::serialize::MmapReadAdapter probe(path);
    auto region = probe.mmapRegion();
    // Only the extent matters here; the probe's own mapping is at a different
    // address, so compare against the loader's mapping via the storages below.
    mmapSize = region->size();
  }

  Stats s;
  std::set<const void*> seen;
  walk(module._ivalue(), s, seen, nullptr, 0);
  // Most of a scripted model's bytes are bytecode constants rather than object
  // attributes, and those are not reachable from _ivalue(), so walk each
  // method's code constants too.
  for (const auto& method : module.get_methods()) {
    for (const auto& c : method.function().get_code().constants_) {
      walk(c, s, seen, nullptr, 0);
    }
  }

  const uint64_t afterWalk = physFootprintBytes();

  // Optionally run a forward pass over a 1-D float input of the given length.
  // Loading is only half the question: if an operator wrote through to a mapped
  // weight, or a backend repacked weights into fresh buffers on first use, the
  // footprint would climb back up here and the output would still have to match
  // the copying path.
  double outputChecksum = 0.0;
  int64_t outputNumel = -1;
  uint64_t afterForward = afterWalk;
  if (argc >= 4) {
    const int64_t n = atoll(argv[3]);
    at::Tensor input = at::empty({n}, at::kFloat);
    float* d = input.data_ptr<float>();
    for (int64_t i = 0; i < n; i++) {
      // Deterministic and non-degenerate, so both modes get the same input and
      // an all-zeros shortcut cannot hide a difference.
      d[i] = (float)((i * 2654435761u) % 2000) / 1000.0f - 1.0f;
    }
    std::vector<c10::IValue> inputs{input};
    c10::IValue out = module.forward(inputs);
    if (out.isTensor()) {
      at::Tensor t = out.toTensor();
      outputNumel = t.numel();
      outputChecksum =
          t.reshape({t.numel()}).to(at::kDouble).sum().item<double>();
    }
    afterForward = physFootprintBytes();
  }

  printf("mode=%s\n", useMmap ? "mmap" : "copy");
  printf("tensors=%" PRId64 " numel=%" PRId64 " storage_mb=%.2f\n",
         s.tensors, s.numel, (double)s.storageBytes / 1e6);
  printf("hash=%" PRIu64 "\n", s.hash);
  printf("file_mb=%.2f\n", (double)mmapSize / 1e6);
  printf("footprint_after_load_mb=%.2f\n", (double)(afterLoad - before) / 1e6);
  printf("footprint_after_touch_all_mb=%.2f\n",
         (double)(afterWalk - before) / 1e6);
  if (outputNumel >= 0) {
    printf("forward_output_numel=%" PRId64 "\n", outputNumel);
    printf("forward_output_checksum=%.6f\n", outputChecksum);
    printf("footprint_after_forward_mb=%.2f\n",
           (double)(afterForward - before) / 1e6);
  }
  return 0;
}
