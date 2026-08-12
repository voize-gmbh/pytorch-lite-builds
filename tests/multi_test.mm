// Loads many models in ONE process, the way a real app does, and reports how
// the physical footprint grows per model -- plus, for every reachable tensor
// storage, whether it lives on the malloc heap or outside it.
//
// The per-model harness (mmap_test.mm) loads a single model per process, which
// can hide effects that only appear once several models are resident at once.
//
// Usage: multi_test <copy|mmap> <model1.ptl> [model2.ptl ...]

#include <ATen/ATen.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>

#include <mach/mach.h>
#include <malloc/malloc.h>

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>

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
  int64_t heapTensors = 0;
  int64_t heapBytes = 0;
  int64_t nonHeapTensors = 0;
  int64_t nonHeapBytes = 0;
};

// malloc_zone_from_ptr returns the owning zone for a heap pointer and NULL for
// anything the allocator does not own -- which is exactly the distinction we
// need: an aliased storage points into the file mapping, so it has no zone,
// while a copied storage was allocated by c10 through malloc and has one.
static void classify(
    const c10::IValue& v,
    Stats& s,
    std::set<const void*>& seen) {
  if (v.isTensor()) {
    const at::Tensor& t = v.toTensor();
    if (!t.defined() || !t.has_storage()) {
      return;
    }
    void* data = const_cast<void*>(t.storage().data());
    if (data == nullptr || !seen.insert(data).second) {
      return;
    }
    const int64_t nbytes = (int64_t)t.storage().nbytes();
    if (malloc_zone_from_ptr(data) == nullptr) {
      s.nonHeapTensors++;
      s.nonHeapBytes += nbytes;
    } else {
      s.heapTensors++;
      s.heapBytes += nbytes;
    }
    return;
  }
  if (v.isTuple()) {
    for (const auto& e : v.toTupleRef().elements()) classify(e, s, seen);
  } else if (v.isList()) {
    for (const auto& e : v.toListRef()) classify(e, s, seen);
  } else if (v.isGenericDict()) {
    for (const auto& e : v.toGenericDict()) classify(e.value(), s, seen);
  } else if (v.isObject()) {
    auto obj = v.toObject();
    if (!seen.insert(obj.get()).second) return;
    for (size_t i = 0; i < obj->slots().size(); i++) {
      classify(obj->slots()[i], s, seen);
    }
  }
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <copy|mmap> <model1.ptl> [model2.ptl ...]\n",
            argv[0]);
    return 2;
  }
  const bool useMmap = std::strcmp(argv[1], "mmap") == 0;
  uint64_t options = torch::jit::kDefaultMobileLoadOptions;
  if (useMmap) {
    options |= torch::jit::MobileModuleLoadOptions::USE_MMAP;
  }

  const uint64_t start = physFootprintBytes();
  printf("mode=%s\nstart_footprint_mb=%.1f\n", useMmap ? "mmap" : "copy",
         (double)start / 1e6);

  // Keep every module alive, so the footprint reflects all of them resident at
  // once rather than one at a time.
  std::vector<std::unique_ptr<torch::jit::mobile::Module>> modules;
  Stats total;
  std::set<const void*> seen;

  for (int i = 2; i < argc; i++) {
    const uint64_t before = physFootprintBytes();
    torch::jit::ExtraFilesMap extraFiles;
    std::unique_ptr<torch::jit::mobile::Module> m;
    try {
      m = std::make_unique<torch::jit::mobile::Module>(
          torch::jit::_load_for_mobile(argv[i], c10::nullopt, extraFiles,
                                       options));
    } catch (const std::exception& e) {
      printf("  LOAD_FAILED %s: %s\n", argv[i], e.what());
      continue;
    }
    const uint64_t after = physFootprintBytes();

    Stats s;
    classify(m->_ivalue(), s, seen);
    for (const auto& method : m->get_methods()) {
      for (const auto& c : method.function().get_code().constants_) {
        classify(c, s, seen);
      }
    }
    total.heapTensors += s.heapTensors;
    total.heapBytes += s.heapBytes;
    total.nonHeapTensors += s.nonHeapTensors;
    total.nonHeapBytes += s.nonHeapBytes;

    const char* base = strrchr(argv[i], '/');
    printf("  %-42s +%7.1f MB  reachable: heap %6.2f MB / mapped %6.2f MB\n",
           base ? base + 1 : argv[i], (double)(after - before) / 1e6,
           (double)s.heapBytes / 1e6, (double)s.nonHeapBytes / 1e6);
    modules.push_back(std::move(m));
  }

  const uint64_t end = physFootprintBytes();
  printf("models_loaded=%zu\n", modules.size());
  printf("total_footprint_growth_mb=%.1f\n", (double)(end - start) / 1e6);
  printf("reachable_storages_on_heap_mb=%.2f (%" PRId64 " tensors)\n",
         (double)total.heapBytes / 1e6, total.heapTensors);
  printf("reachable_storages_mapped_mb=%.2f (%" PRId64 " tensors)\n",
         (double)total.nonHeapBytes / 1e6, total.nonHeapTensors);
  return 0;
}
