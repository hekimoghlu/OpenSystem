/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 3, 2024.
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

#include <stdlib.h>
#include <stdio.h>

#define HEAP_SIZE (128 * 1024)

__attribute__((aligned(16)))
char heap[HEAP_SIZE] = {};

size_t next_heap_index = 0;

void *calloc(size_t count, size_t size) {
  size_t total_size = count * size;
  printf("malloc(%ld)", total_size);

  if (total_size % 16 != 0)
    total_size = total_size + (16 - total_size % 16);

  if (next_heap_index + total_size > HEAP_SIZE) {
    puts("HEAP EXHAUSTED\n");
    __builtin_trap();
  }
  void *p = &heap[next_heap_index];
  next_heap_index += total_size;
  printf("-> %p\n", p);
  return p;
}

void *malloc(size_t count) {
  return calloc(count, 1);
}

int posix_memalign(void **memptr, size_t alignment, size_t size) {
  *memptr = calloc(size + alignment, 1);
  if (((uintptr_t)*memptr) % alignment == 0) return 0;
  *(uintptr_t *)memptr += alignment - ((uintptr_t)*memptr % alignment);
  return 0;
}

void free(void *ptr) {
  // don't actually free
  printf("free(%p)\n", ptr);
}
