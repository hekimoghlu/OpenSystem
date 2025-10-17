/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 13, 2022.
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
#ifndef PAS_HEAP_TABLE_H
#define PAS_HEAP_TABLE_H

#include "pas_large_heap.h"
#include "pas_log.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

/* We want the table to hold the first 2^16 heaps. */
#define PAS_HEAP_TABLE_SIZE 65536

PAS_API extern pas_large_heap** pas_heap_table;
PAS_API extern unsigned pas_heap_table_bump_index;

/* Call with heap lock held. */
PAS_API void pas_heap_table_try_allocate_index(pas_large_heap* heap);

static inline bool pas_heap_table_has_index(pas_large_heap* heap)
{
    static const bool verbose = PAS_SHOULD_LOG(PAS_LOG_HEAP_INFRASTRUCTURE);

    if (heap->table_state == pas_heap_table_state_uninitialized)
        pas_heap_table_try_allocate_index(heap);
    switch (heap->table_state) {
    case pas_heap_table_state_failed:
        if (verbose)
            pas_log("failed to get index for heap %p.\n", heap);
        return false;
    case pas_heap_table_state_has_index:
        if (verbose)
            pas_log("going to get an index for heap %p.\n", heap);
        return true;
    default:
        PAS_ASSERT(!"Should not be reached");
        return false;
    }
}

static inline uint16_t pas_heap_table_get_index(pas_large_heap* heap)
{
    PAS_ASSERT(heap->table_state == pas_heap_table_state_has_index);
    return heap->index;
}

PAS_END_EXTERN_C;

#endif /* PAS_HEAP_TABLE_H */

