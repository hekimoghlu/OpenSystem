/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 7, 2024.
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
#include "private/bionic_allocator.h"

#include <stdlib.h>
#include <sys/cdefs.h>
#include <unistd.h>

#include <atomic>

#include <async_safe/log.h>

static BionicAllocator g_bionic_allocator;
static std::atomic<pid_t> fallback_tid(0);

// Used by libdebuggerd_handler to switch allocators during a crash dump, in
// case the linker heap is corrupted. Do not use this function.
extern "C" bool __linker_enable_fallback_allocator() {
  pid_t expected = 0;
  return fallback_tid.compare_exchange_strong(expected, gettid());
}

extern "C" void __linker_disable_fallback_allocator() {
  pid_t previous = fallback_tid.exchange(0);
  if (previous == 0) {
    async_safe_fatal("attempted to disable unused fallback allocator");
  } else if (previous != gettid()) {
    async_safe_fatal("attempted to disable fallback allocator in use by another thread (%d)",
                     previous);
  }
}

static BionicAllocator& get_fallback_allocator() {
  static BionicAllocator fallback_allocator;
  return fallback_allocator;
}

static BionicAllocator& get_allocator() {
  if (__predict_false(fallback_tid) && __predict_false(gettid() == fallback_tid)) {
    return get_fallback_allocator();
  }
  return g_bionic_allocator;
}

void* malloc(size_t byte_count) {
  return get_allocator().alloc(byte_count);
}

void* memalign(size_t alignment, size_t byte_count) {
  return get_allocator().memalign(alignment, byte_count);
}

void* aligned_alloc(size_t alignment, size_t byte_count) {
  return get_allocator().memalign(alignment, byte_count);
}

void* calloc(size_t item_count, size_t item_size) {
  return get_allocator().alloc(item_count*item_size);
}

void* realloc(void* p, size_t byte_count) {
  return get_allocator().realloc(p, byte_count);
}

void* reallocarray(void* p, size_t item_count, size_t item_size) {
  size_t byte_count;
  if (__builtin_mul_overflow(item_count, item_size, &byte_count)) {
    errno = ENOMEM;
    return nullptr;
  }
  return get_allocator().realloc(p, byte_count);
}

void free(void* ptr) {
  get_allocator().free(ptr);
}
