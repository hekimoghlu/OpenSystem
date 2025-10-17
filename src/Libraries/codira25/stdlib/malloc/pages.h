/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 22, 2023.
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

#ifndef PAGES_H
#define PAGES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "util.h"

#define PAGE_SHIFT 12
#ifndef PAGE_SIZE
#define PAGE_SIZE ((size_t)1 << PAGE_SHIFT)
#endif

void *allocate_pages(size_t usable_size, size_t guard_size, bool unprotect, const char *name);
void *allocate_pages_aligned(size_t usable_size, size_t alignment, size_t guard_size, const char *name);
void deallocate_pages(void *usable, size_t usable_size, size_t guard_size);

static inline size_t page_align(size_t size) {
    return align(size, PAGE_SIZE);
}

static inline size_t hash_page(const void *p) {
    uintptr_t u = (uintptr_t)p >> PAGE_SHIFT;
    size_t sum = u;
    sum = (sum << 7) - sum + (u >> 16);
    sum = (sum << 7) - sum + (u >> 32);
    sum = (sum << 7) - sum + (u >> 48);
    return sum;
}

#endif
