/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 13, 2025.
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
#ifndef PAS_HEAP_H
#define PAS_HEAP_H

#include "pas_heap_ref.h"
#include "pas_heap_summary.h"
#include "pas_large_heap.h"
#include "pas_segregated_heap.h"

PAS_BEGIN_EXTERN_C;

struct pas_heap;
struct pas_heap_ref;
struct pas_segregated_size_directory;
struct pas_segregated_page;
typedef struct pas_heap pas_heap;
typedef struct pas_heap_ref pas_heap_ref;
typedef struct pas_segregated_size_directory pas_segregated_size_directory;
typedef struct pas_segregated_page pas_segregated_page;

struct pas_heap {
    pas_segregated_heap segregated_heap;
    pas_large_heap megapage_large_heap;
    pas_large_heap large_heap;
    const pas_heap_type* type;
    pas_heap_ref* heap_ref;
    pas_compact_heap_ptr next_heap;
    pas_heap_config_kind config_kind : 6;
    pas_heap_ref_kind heap_ref_kind : 2;
};

PAS_API pas_heap* pas_heap_create(pas_heap_ref* heap_ref,
                                  pas_heap_ref_kind heap_ref_kind,
                                  const pas_heap_config* config,
                                  pas_heap_runtime_config* runtime_config);

/* Returns 1 for NULL heap. */
PAS_API size_t pas_heap_get_type_size(pas_heap* heap);
PAS_API size_t pas_heap_get_type_alignment(pas_heap* heap);

/* All large heaps belong to the heap in such a way that given a large heap, we can find the
   heap. */
static inline pas_heap* pas_heap_for_large_heap(pas_large_heap* large_heap)
{
    size_t offset = large_heap->is_megapage_heap
        ? PAS_OFFSETOF(pas_heap, megapage_large_heap)
        : PAS_OFFSETOF(pas_heap, large_heap);
    return (pas_heap*)((uintptr_t)large_heap - offset);
}

/* FIXME: It would be so much simpler if every segregated_heap belong to a heap, or if they were just
   merged into a single data structure. */
static inline pas_heap* pas_heap_for_segregated_heap(pas_segregated_heap* segregated_heap)
{
    if (!segregated_heap->runtime_config->is_part_of_heap)
        return NULL;
    return (pas_heap*)((uintptr_t)segregated_heap - PAS_OFFSETOF(pas_heap, segregated_heap));
}

PAS_API size_t pas_heap_get_num_free_bytes(pas_heap* heap);

typedef bool (*pas_heap_for_each_live_object_callback)(pas_heap* heap,
                                                       uintptr_t begin,
                                                       size_t size,
                                                       void* arg);

PAS_API bool pas_heap_for_each_live_object(pas_heap* heap,
                                           pas_heap_for_each_live_object_callback callback,
                                           void *arg,
                                           pas_lock_hold_mode heap_lock_hold_mode);

PAS_API pas_heap_summary pas_heap_compute_summary(pas_heap* heap,
                                                  pas_lock_hold_mode heap_lock_hold_mode);

PAS_API void pas_heap_reset_heap_ref(pas_heap* heap);

PAS_API bool pas_check_pgm_entry_exists(void *ptr);

PAS_END_EXTERN_C;

#endif /* PAS_HEAP_H */
