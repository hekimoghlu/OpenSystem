/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 21, 2023.
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

#include <stdint.h>
#include <stdio.h>
#include <malloc.h>

#include <private/bionic_config.h>

__BEGIN_DECLS

void* scudo_aligned_alloc(size_t, size_t);
void* scudo_calloc(size_t, size_t);
void scudo_free(void*);
struct mallinfo scudo_mallinfo();
void* scudo_malloc(size_t);
int scudo_malloc_info(int, FILE*);
size_t scudo_malloc_usable_size(const void*);
int scudo_mallopt(int, int);
void* scudo_memalign(size_t, size_t);
void* scudo_realloc(void*, size_t);
int scudo_posix_memalign(void**, size_t, size_t);
#if defined(HAVE_DEPRECATED_MALLOC_FUNCS)
void* scudo_pvalloc(size_t);
void* scudo_valloc(size_t);
#endif

int scudo_malloc_iterate(uintptr_t, size_t, void (*)(uintptr_t, size_t, void*), void*);
void scudo_malloc_disable();
void scudo_malloc_enable();

void* scudo_svelte_aligned_alloc(size_t, size_t);
void* scudo_svelte_calloc(size_t, size_t);
void scudo_svelte_free(void*);
struct mallinfo scudo_svelte_mallinfo();
void* scudo_svelte_malloc(size_t);
int scudo_svelte_malloc_info(int, FILE*);
size_t scudo_svelte_malloc_usable_size(const void*);
int scudo_svelte_mallopt(int, int);
void* scudo_svelte_memalign(size_t, size_t);
void* scudo_svelte_realloc(void*, size_t);
int scudo_svelte_posix_memalign(void**, size_t, size_t);
#if defined(HAVE_DEPRECATED_MALLOC_FUNCS)
void* scudo_svelte_pvalloc(size_t);
void* scudo_svelte_valloc(size_t);
#endif

int scudo_svelte_malloc_iterate(uintptr_t, size_t, void (*)(uintptr_t, size_t, void*), void*);
void scudo_svelte_malloc_disable();
void scudo_svelte_malloc_enable();

__END_DECLS
