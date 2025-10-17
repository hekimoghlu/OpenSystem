/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 22, 2023.
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
#ifndef PAS_UTILITY_HEAP_H
#define PAS_UTILITY_HEAP_H

#include "pas_allocation_kind.h"
#include "pas_intrinsic_heap_support.h"
#include "pas_allocator_counts.h"
#include "pas_lock.h"
#include "pas_segregated_heap.h"
#include "pas_utility_heap_support.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_segregated_heap;
struct pas_segregated_page;
struct pas_segregated_size_directory;
typedef struct pas_segregated_heap pas_segregated_heap;
typedef struct pas_segregated_page pas_segregated_page;
typedef struct pas_segregated_size_directory pas_segregated_size_directory;

/* The utility heap is for allocating high volumes of small objects as part of bookkeeping for
   the heaps that libpas actually provides. It can only handle objects up to some size (roughly
   PAS_UTILITY_LOOKUP_SIZE_UPPER_BOUND, but not exactly - the math gets discretized). You have
   to hold the heap lock to allocate and deallocate in it. */

PAS_API extern pas_utility_heap_support pas_utility_heap_support_instance;
PAS_API extern pas_heap_runtime_config pas_utility_heap_runtime_config;
PAS_API extern pas_segregated_heap pas_utility_segregated_heap;
PAS_API extern pas_allocator_counts pas_utility_allocator_counts;

PAS_API void* pas_utility_heap_try_allocate(size_t size, const char* name);
PAS_API void* pas_utility_heap_allocate(size_t size, const char* name);

static inline void* pas_utility_heap_allocate_with_asserted_kind(
    size_t size, const char* name, pas_allocation_kind allocation_kind)
{
    PAS_ASSERT(allocation_kind == pas_object_allocation);
    return pas_utility_heap_allocate(size, name);
}

PAS_API void* pas_utility_heap_try_allocate_with_alignment(
    size_t size,
    size_t alignment,
    const char* name);
PAS_API void* pas_utility_heap_allocate_with_alignment(
    size_t size,
    size_t alignment,
    const char* name);

PAS_API void pas_utility_heap_deallocate(void* ptr);

static inline void pas_utility_heap_deallocate_with_ignored_size_and_asserted_kind(
    void* ptr, size_t size, pas_allocation_kind kind)
{
    PAS_ASSERT(kind == pas_object_allocation);
    PAS_UNUSED_PARAM(size);
    pas_utility_heap_deallocate(ptr);
}

PAS_API size_t pas_utility_heap_get_num_free_bytes(void); /* Gotta hold the heap lock. */

typedef bool (*pas_utility_heap_for_each_live_object_callback)(
    uintptr_t begin,
    size_t size,
    void* arg);

PAS_API bool pas_utility_heap_for_each_live_object(
    pas_utility_heap_for_each_live_object_callback callback,
    void* arg);

PAS_API bool pas_utility_heap_for_all_allocators(pas_allocator_scavenge_action action,
                                                 pas_lock_hold_mode heap_lock_hold_mode);

PAS_END_EXTERN_C;

#endif /* PAS_UTILITY_HEAP_H */

