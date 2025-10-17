/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 2, 2022.
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
#include "aom_mem.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "include/aom_mem_intrnl.h"
#include "aom/aom_integer.h"

static size_t GetAllocationPaddingSize(size_t align) {
  assert(align > 0);
  assert(align < SIZE_MAX - ADDRESS_STORAGE_SIZE);
  return align - 1 + ADDRESS_STORAGE_SIZE;
}

// Returns 0 in case of overflow of nmemb * size.
static int check_size_argument_overflow(size_t nmemb, size_t size,
                                        size_t align) {
  if (nmemb == 0) return 1;
  const size_t alloc_padding = GetAllocationPaddingSize(align);
#if defined(AOM_MAX_ALLOCABLE_MEMORY)
  assert(AOM_MAX_ALLOCABLE_MEMORY >= alloc_padding);
  assert(AOM_MAX_ALLOCABLE_MEMORY <= SIZE_MAX);
  if (size > (AOM_MAX_ALLOCABLE_MEMORY - alloc_padding) / nmemb) return 0;
#else
  if (size > (SIZE_MAX - alloc_padding) / nmemb) return 0;
#endif
  return 1;
}

static size_t *GetMallocAddressLocation(void *const mem) {
  return ((size_t *)mem) - 1;
}

static void SetActualMallocAddress(void *const mem,
                                   const void *const malloc_addr) {
  size_t *const malloc_addr_location = GetMallocAddressLocation(mem);
  *malloc_addr_location = (size_t)malloc_addr;
}

static void *GetActualMallocAddress(void *const mem) {
  const size_t *const malloc_addr_location = GetMallocAddressLocation(mem);
  return (void *)(*malloc_addr_location);
}

void *aom_memalign(size_t align, size_t size) {
  void *x = NULL;
  if (!check_size_argument_overflow(1, size, align)) return NULL;
  const size_t aligned_size = size + GetAllocationPaddingSize(align);
  void *const addr = malloc(aligned_size);
  if (addr) {
    x = aom_align_addr((unsigned char *)addr + ADDRESS_STORAGE_SIZE, align);
    SetActualMallocAddress(x, addr);
  }
  return x;
}

void *aom_malloc(size_t size) { return aom_memalign(DEFAULT_ALIGNMENT, size); }

void *aom_calloc(size_t num, size_t size) {
  if (!check_size_argument_overflow(num, size, DEFAULT_ALIGNMENT)) return NULL;
  const size_t total_size = num * size;
  void *const x = aom_malloc(total_size);
  if (x) memset(x, 0, total_size);
  return x;
}

void aom_free(void *memblk) {
  if (memblk) {
    void *addr = GetActualMallocAddress(memblk);
    free(addr);
  }
}
