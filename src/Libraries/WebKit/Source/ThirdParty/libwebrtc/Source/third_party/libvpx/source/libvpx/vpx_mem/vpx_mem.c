/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 25, 2024.
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
#include "vpx_mem.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "include/vpx_mem_intrnl.h"
#include "vpx/vpx_integer.h"

#if !defined(VPX_MAX_ALLOCABLE_MEMORY)
#if SIZE_MAX > (1ULL << 40)
#define VPX_MAX_ALLOCABLE_MEMORY (1ULL << 40)
#else
// For 32-bit targets keep this below INT_MAX to avoid valgrind warnings.
#define VPX_MAX_ALLOCABLE_MEMORY ((1ULL << 31) - (1 << 16))
#endif
#endif

// Returns 0 in case of overflow of nmemb * size.
static int check_size_argument_overflow(uint64_t nmemb, uint64_t size) {
  const uint64_t total_size = nmemb * size;
  if (nmemb == 0) return 1;
  if (size > VPX_MAX_ALLOCABLE_MEMORY / nmemb) return 0;
  if (total_size != (size_t)total_size) return 0;

  return 1;
}

static size_t *get_malloc_address_location(void *const mem) {
  return ((size_t *)mem) - 1;
}

static uint64_t get_aligned_malloc_size(size_t size, size_t align) {
  return (uint64_t)size + align - 1 + ADDRESS_STORAGE_SIZE;
}

static void set_actual_malloc_address(void *const mem,
                                      const void *const malloc_addr) {
  size_t *const malloc_addr_location = get_malloc_address_location(mem);
  *malloc_addr_location = (size_t)malloc_addr;
}

static void *get_actual_malloc_address(void *const mem) {
  size_t *const malloc_addr_location = get_malloc_address_location(mem);
  return (void *)(*malloc_addr_location);
}

void *vpx_memalign(size_t align, size_t size) {
  void *x = NULL, *addr;
  const uint64_t aligned_size = get_aligned_malloc_size(size, align);
  if (!check_size_argument_overflow(1, aligned_size)) return NULL;

  addr = malloc((size_t)aligned_size);
  if (addr) {
    x = align_addr((unsigned char *)addr + ADDRESS_STORAGE_SIZE, align);
    set_actual_malloc_address(x, addr);
  }
  return x;
}

void *vpx_malloc(size_t size) { return vpx_memalign(DEFAULT_ALIGNMENT, size); }

void *vpx_calloc(size_t num, size_t size) {
  void *x;
  if (!check_size_argument_overflow(num, size)) return NULL;

  x = vpx_malloc(num * size);
  if (x) memset(x, 0, num * size);
  return x;
}

void vpx_free(void *memblk) {
  if (memblk) {
    void *addr = get_actual_malloc_address(memblk);
    free(addr);
  }
}
