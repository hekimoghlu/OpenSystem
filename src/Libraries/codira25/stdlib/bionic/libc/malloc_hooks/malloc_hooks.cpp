/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 9, 2022.
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
#include <errno.h>
#include <malloc.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/param.h>
#include <unistd.h>

#include <private/bionic_malloc_dispatch.h>

// ------------------------------------------------------------------------
// Global Data
// ------------------------------------------------------------------------
const MallocDispatch* g_dispatch;
// ------------------------------------------------------------------------

// ------------------------------------------------------------------------
// Use C style prototypes for all exported functions. This makes it easy
// to do dlsym lookups during libc initialization when hooks are enabled.
// ------------------------------------------------------------------------
__BEGIN_DECLS

bool hooks_initialize(const MallocDispatch* malloc_dispatch, bool* zygote_child,
    const char* options);
void hooks_finalize();
void hooks_get_malloc_leak_info(
    uint8_t** info, size_t* overall_size, size_t* info_size, size_t* total_memory,
    size_t* backtrace_size);
ssize_t hooks_malloc_backtrace(void* pointer, uintptr_t* frames, size_t frame_count);
void hooks_free_malloc_leak_info(uint8_t* info);
size_t hooks_malloc_usable_size(void* pointer);
void* hooks_malloc(size_t size);
int hooks_malloc_info(int options, FILE* fp);
void hooks_free(void* pointer);
void* hooks_memalign(size_t alignment, size_t bytes);
void* hooks_aligned_alloc(size_t alignment, size_t bytes);
void* hooks_realloc(void* pointer, size_t bytes);
void* hooks_calloc(size_t nmemb, size_t bytes);
struct mallinfo hooks_mallinfo();
int hooks_mallopt(int param, int value);
int hooks_posix_memalign(void** memptr, size_t alignment, size_t size);
int hooks_malloc_iterate(uintptr_t base, size_t size,
    void (*callback)(uintptr_t base, size_t size, void* arg), void* arg);
void hooks_malloc_disable();
void hooks_malloc_enable();
bool hooks_write_malloc_leak_info(FILE*);

#if defined(HAVE_DEPRECATED_MALLOC_FUNCS)
void* hooks_pvalloc(size_t bytes);
void* hooks_valloc(size_t size);
#endif

static void* default_malloc_hook(size_t bytes, const void*) {
  return g_dispatch->malloc(bytes);
}

static void* default_realloc_hook(void* pointer, size_t bytes, const void*) {
  return g_dispatch->realloc(pointer, bytes);
}

static void default_free_hook(void* pointer, const void*) {
  g_dispatch->free(pointer);
}

static void* default_memalign_hook(size_t alignment, size_t bytes, const void*) {
  return g_dispatch->memalign(alignment, bytes);
}

__END_DECLS
// ------------------------------------------------------------------------

bool hooks_initialize(const MallocDispatch* malloc_dispatch, bool*, const char*) {
  g_dispatch = malloc_dispatch;
  __malloc_hook = default_malloc_hook;
  __realloc_hook = default_realloc_hook;
  __free_hook = default_free_hook;
  __memalign_hook = default_memalign_hook;
  return true;
}

void hooks_finalize() {
}

void hooks_get_malloc_leak_info(uint8_t** info, size_t* overall_size,
    size_t* info_size, size_t* total_memory, size_t* backtrace_size) {
  *info = nullptr;
  *overall_size = 0;
  *info_size = 0;
  *total_memory = 0;
  *backtrace_size = 0;
}

void hooks_free_malloc_leak_info(uint8_t*) {
}

size_t hooks_malloc_usable_size(void* pointer) {
  return g_dispatch->malloc_usable_size(pointer);
}

void* hooks_malloc(size_t size) {
  if (__malloc_hook != nullptr && __malloc_hook != default_malloc_hook) {
    return __malloc_hook(size, __builtin_return_address(0));
  }
  return g_dispatch->malloc(size);
}

void hooks_free(void* pointer) {
  if (__free_hook != nullptr && __free_hook != default_free_hook) {
    return __free_hook(pointer, __builtin_return_address(0));
  }
  return g_dispatch->free(pointer);
}

void* hooks_memalign(size_t alignment, size_t bytes) {
  if (__memalign_hook != nullptr && __memalign_hook != default_memalign_hook) {
    return __memalign_hook(alignment, bytes, __builtin_return_address(0));
  }
  return g_dispatch->memalign(alignment, bytes);
}

void* hooks_realloc(void* pointer, size_t bytes) {
  if (__realloc_hook != nullptr && __realloc_hook != default_realloc_hook) {
    return __realloc_hook(pointer, bytes, __builtin_return_address(0));
  }
  return g_dispatch->realloc(pointer, bytes);
}

void* hooks_calloc(size_t nmemb, size_t bytes) {
  if (__malloc_hook != nullptr && __malloc_hook != default_malloc_hook) {
    size_t size;
    if (__builtin_mul_overflow(nmemb, bytes, &size)) {
      return nullptr;
    }
    void* ptr = __malloc_hook(size, __builtin_return_address(0));
    if (ptr != nullptr) {
      memset(ptr, 0, size);
    }
    return ptr;
  }
  return g_dispatch->calloc(nmemb, bytes);
}

struct mallinfo hooks_mallinfo() {
  return g_dispatch->mallinfo();
}

int hooks_mallopt(int param, int value) {
  return g_dispatch->mallopt(param, value);
}

int hooks_malloc_info(int options, FILE* fp) {
  return g_dispatch->malloc_info(options, fp);
}

void* hooks_aligned_alloc(size_t alignment, size_t size) {
  if (__memalign_hook != nullptr && __memalign_hook != default_memalign_hook) {
    if (!powerof2(alignment) || (size % alignment) != 0) {
      errno = EINVAL;
      return nullptr;
    }
    void* ptr = __memalign_hook(alignment, size, __builtin_return_address(0));
    if (ptr == nullptr) {
      errno = ENOMEM;
    }
    return ptr;
  }
  return g_dispatch->aligned_alloc(alignment, size);
}

int hooks_posix_memalign(void** memptr, size_t alignment, size_t size) {
  if (__memalign_hook != nullptr && __memalign_hook != default_memalign_hook) {
    if (alignment < sizeof(void*) || !powerof2(alignment)) {
      return EINVAL;
    }
    *memptr = __memalign_hook(alignment, size, __builtin_return_address(0));
    if (*memptr == nullptr) {
      return ENOMEM;
    }
    return 0;
  }
  return g_dispatch->posix_memalign(memptr, alignment, size);
}

int hooks_malloc_iterate(uintptr_t, size_t, void (*)(uintptr_t, size_t, void*), void*) {
  return 0;
}

void hooks_malloc_disable() {
}

void hooks_malloc_enable() {
}

ssize_t hooks_malloc_backtrace(void*, uintptr_t*, size_t) {
  return 0;
}

bool hooks_write_malloc_leak_info(FILE*) {
  return true;
}

#if defined(HAVE_DEPRECATED_MALLOC_FUNCS)
void* hooks_pvalloc(size_t bytes) {
  size_t pagesize = getpagesize();
  size_t size = __BIONIC_ALIGN(bytes, pagesize);
  if (size < bytes) {
    // Overflow
    errno = ENOMEM;
    return nullptr;
  }
  return hooks_memalign(pagesize, size);
}

void* hooks_valloc(size_t size) {
  return hooks_memalign(getpagesize(), size);
}
#endif
