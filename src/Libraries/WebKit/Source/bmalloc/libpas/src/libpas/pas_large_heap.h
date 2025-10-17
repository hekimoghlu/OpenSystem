/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 7, 2025.
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
#ifndef PAS_LARGE_HEAP_H
#define PAS_LARGE_HEAP_H

#include "pas_fast_large_free_heap.h"
#include "pas_allocation_mode.h"
#include "pas_heap_summary.h"
#include "pas_heap_table_state.h"
#include "pas_physical_memory_transaction.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_heap_config;
struct pas_large_heap;
typedef struct pas_heap_config pas_heap_config;
typedef struct pas_large_heap pas_large_heap;

struct pas_large_heap {
    pas_fast_large_free_heap free_heap;
    uint16_t index;
    pas_heap_table_state table_state : 8;
    bool is_megapage_heap;
};

/* Note that all of these functions have to be called with the heap lock held. */

/* NOTE: it's only valid to construct a large heap that is a member of a pas_heap. */
PAS_API void pas_large_heap_construct(pas_large_heap* heap, bool is_megapage_heap);

PAS_API pas_allocation_result
pas_large_heap_try_allocate_and_forget(pas_large_heap* heap,
                                       size_t size, size_t alignment,
                                       pas_allocation_mode allocation_mode,
                                       const pas_heap_config* config,
                                       pas_physical_memory_transaction* transaction);

PAS_API pas_allocation_result
pas_large_heap_try_allocate(pas_large_heap* heap,
                            size_t size, size_t alignment,
                            pas_allocation_mode allocation_mode,
                            const pas_heap_config* config,
                            pas_physical_memory_transaction* transaction);

/* Returns true if an object was found and deallocated. */
PAS_API bool pas_large_heap_try_deallocate(uintptr_t base,
                                           const pas_heap_config* config);

/* Returns true if an object was found and shrunk. */
PAS_API bool pas_large_heap_try_shrink(uintptr_t base,
                                       size_t new_size,
                                       const pas_heap_config* config);

/* This is a super crazy function that lets you shove memory into the allocator. There is
   one user (the large region) and it only does it to one heap (the primitive heap). It's
   not something you probably ever want to do. */
PAS_API void pas_large_heap_shove_into_free(pas_large_heap* heap, uintptr_t begin, uintptr_t end,
                                            pas_zero_mode zero_mode,
                                            const pas_heap_config* config);

typedef bool (*pas_large_heap_for_each_live_object_callback)(pas_large_heap* heap,
                                                             uintptr_t begin,
                                                             uintptr_t end,
                                                             void *arg);

PAS_API bool pas_large_heap_for_each_live_object(
    pas_large_heap* heap,
    pas_large_heap_for_each_live_object_callback callback,
    void *arg);

PAS_API pas_large_heap* pas_large_heap_for_object(uintptr_t begin);

PAS_API size_t pas_large_heap_get_num_free_bytes(pas_large_heap* heap);

PAS_API pas_heap_summary pas_large_heap_compute_summary(pas_large_heap* heap);

PAS_END_EXTERN_C;

#endif /* PAS_LARGE_HEAP_H */

