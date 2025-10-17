/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 13, 2025.
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
#ifndef _PRIVATE_BIONIC_MALLOC_DISPATCH_H
#define _PRIVATE_BIONIC_MALLOC_DISPATCH_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <private/bionic_config.h>

// Entry in malloc dispatch table.
typedef void* (*MallocCalloc)(size_t, size_t);
typedef void (*MallocFree)(void*);
typedef struct mallinfo (*MallocMallinfo)();
typedef void* (*MallocMalloc)(size_t);
typedef int (*MallocMallocInfo)(int, FILE*);
typedef size_t (*MallocMallocUsableSize)(const void*);
typedef void* (*MallocMemalign)(size_t, size_t);
typedef int (*MallocPosixMemalign)(void**, size_t, size_t);
typedef void* (*MallocRealloc)(void*, size_t);
typedef int (*MallocIterate)(uintptr_t, size_t, void (*)(uintptr_t, size_t, void*), void*);
typedef void (*MallocMallocDisable)();
typedef void (*MallocMallocEnable)();
typedef int (*MallocMallopt)(int, int);
typedef void* (*MallocAlignedAlloc)(size_t, size_t);

#if defined(HAVE_DEPRECATED_MALLOC_FUNCS)
typedef void* (*MallocPvalloc)(size_t);
typedef void* (*MallocValloc)(size_t);
#endif

struct MallocDispatch {
  MallocCalloc calloc;
  MallocFree free;
  MallocMallinfo mallinfo;
  MallocMalloc malloc;
  MallocMallocUsableSize malloc_usable_size;
  MallocMemalign memalign;
  MallocPosixMemalign posix_memalign;
#if defined(HAVE_DEPRECATED_MALLOC_FUNCS)
  MallocPvalloc pvalloc;
#endif
  MallocRealloc realloc;
#if defined(HAVE_DEPRECATED_MALLOC_FUNCS)
  MallocValloc valloc;
#endif
  MallocIterate malloc_iterate;
  MallocMallocDisable malloc_disable;
  MallocMallocEnable malloc_enable;
  MallocMallopt mallopt;
  MallocAlignedAlloc aligned_alloc;
  MallocMallocInfo malloc_info;
} __attribute__((aligned(32)));

#endif
