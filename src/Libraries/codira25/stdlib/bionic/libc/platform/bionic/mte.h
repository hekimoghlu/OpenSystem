/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 12, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#pragma once

#include <stddef.h>
#include <sys/auxv.h>
#include <sys/mman.h>
#include <sys/prctl.h>

#include "page.h"

// Note: Most PR_MTE_* constants come from the upstream kernel. This tag mask
// allows for the hardware to provision any nonzero tag. Zero tags are reserved
// for scudo to use for the chunk headers in order to prevent linear heap
// overflow/underflow.
#define PR_MTE_TAG_SET_NONZERO (0xfffeUL << PR_MTE_TAG_SHIFT)

inline bool mte_supported() {
#if defined(__aarch64__)
  static bool supported = getauxval(AT_HWCAP2) & HWCAP2_MTE;
#else
  static bool supported = false;
#endif
  return supported;
}

static inline bool mte_enabled() {
#ifdef __aarch64__
  int level = prctl(PR_GET_TAGGED_ADDR_CTRL, 0, 0, 0, 0);
  return level >= 0 && (level & PR_TAGGED_ADDR_ENABLE) &&
         (level & PR_MTE_TCF_MASK) != PR_MTE_TCF_NONE;
#else
  return false;
#endif
}

inline void* get_tagged_address(const void* ptr) {
#if defined(__aarch64__)
  if (mte_supported()) {
    __asm__ __volatile__(".arch_extension mte; ldg %0, [%0]" : "+r"(ptr));
  }
#endif  // aarch64
  return const_cast<void*>(ptr);
}

// Inserts a random tag tag to `ptr`, using any of the set lower 16 bits in
// `mask` to exclude the corresponding tag from being generated. Note: This does
// not tag memory. This generates a pointer to be used with set_memory_tag.
inline void* insert_random_tag(const void* ptr, __attribute__((unused)) uint64_t mask = 0) {
#if defined(__aarch64__)
  if (mte_supported() && ptr) {
    __asm__ __volatile__(".arch_extension mte; irg %0, %0, %1" : "+r"(ptr) : "r"(mask));
  }
#endif  // aarch64
  return const_cast<void*>(ptr);
}

// Stores the address tag in `ptr` to memory, at `ptr`.
inline void set_memory_tag(__attribute__((unused)) void* ptr) {
#if defined(__aarch64__)
  if (mte_supported()) {
    __asm__ __volatile__(".arch_extension mte; stg %0, [%0]" : "+r"(ptr));
  }
#endif  // aarch64
}

#ifdef __aarch64__
class ScopedDisableMTE {
  size_t prev_tco_;

 public:
  ScopedDisableMTE() {
    if (mte_supported()) {
      __asm__ __volatile__(".arch_extension mte; mrs %0, tco; msr tco, #1" : "=r"(prev_tco_));
    }
  }

  ~ScopedDisableMTE() {
    if (mte_supported()) {
      __asm__ __volatile__(".arch_extension mte; msr tco, %0" : : "r"(prev_tco_));
    }
  }
};

// N.B. that this is NOT the pagesize, but 4096. This is hardcoded in the codegen.
// See
// https://github.com/search?q=repo%3Atoolchain/toolchain-project%20AArch64StackTagging%3A%3AinsertBaseTaggedPointer&type=code
constexpr size_t kStackMteRingbufferSizeMultiplier = 4096;

inline size_t stack_mte_ringbuffer_size(uintptr_t size_cls) {
  return kStackMteRingbufferSizeMultiplier * (1 << size_cls);
}

inline size_t stack_mte_ringbuffer_size_from_pointer(uintptr_t ptr) {
  // The size in the top byte is not the size_cls, but the number of "pages" (not OS pages, but
  // kStackMteRingbufferSizeMultiplier).
  return kStackMteRingbufferSizeMultiplier * (ptr >> 56ULL);
}

inline uintptr_t stack_mte_ringbuffer_size_add_to_pointer(uintptr_t ptr, uintptr_t size_cls) {
  return ptr | ((1ULL << size_cls) << 56ULL);
}

inline void stack_mte_free_ringbuffer(uintptr_t stack_mte_tls) {
  size_t size = stack_mte_ringbuffer_size_from_pointer(stack_mte_tls);
  void* ptr = reinterpret_cast<void*>(stack_mte_tls & ((1ULL << 56ULL) - 1ULL));
  munmap(ptr, size);
}

inline void* stack_mte_ringbuffer_allocate(size_t n, const char* name) {
  if (n > 7) return nullptr;
  // Allocation needs to be aligned to 2*size to make the fancy code-gen work.
  // So we allocate 3*size - pagesz bytes, which will always contain size bytes
  // aligned to 2*size, and unmap the unneeded part.
  // See
  // https://github.com/search?q=repo%3Atoolchain/toolchain-project%20AArch64StackTagging%3A%3AinsertBaseTaggedPointer&type=code
  //
  // In the worst case, we get an allocation that is one page past the properly
  // aligned address, in which case we have to unmap the previous
  // 2*size - pagesz bytes. In that case, we still have size properly aligned
  // bytes left.
  size_t size = stack_mte_ringbuffer_size(n);
  size_t pgsize = page_size();

  size_t alloc_size = __BIONIC_ALIGN(3 * size - pgsize, pgsize);
  void* allocation_ptr =
      mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (allocation_ptr == MAP_FAILED)
    return nullptr;
  uintptr_t allocation = reinterpret_cast<uintptr_t>(allocation_ptr);

  size_t alignment = 2 * size;
  uintptr_t aligned_allocation = __BIONIC_ALIGN(allocation, alignment);
  if (allocation != aligned_allocation) {
    munmap(reinterpret_cast<void*>(allocation), aligned_allocation - allocation);
  }
  if (aligned_allocation + size != allocation + alloc_size) {
    munmap(reinterpret_cast<void*>(aligned_allocation + size),
           (allocation + alloc_size) - (aligned_allocation + size));
  }

  if (name) {
    prctl(PR_SET_VMA, PR_SET_VMA_ANON_NAME, reinterpret_cast<void*>(aligned_allocation), size, name);
  }

  // We store the size in the top byte of the pointer (which is ignored)
  return reinterpret_cast<void*>(stack_mte_ringbuffer_size_add_to_pointer(aligned_allocation, n));
}
#else
struct ScopedDisableMTE {
  // Silence unused variable warnings in non-aarch64 builds.
  ScopedDisableMTE() {}
};
#endif
