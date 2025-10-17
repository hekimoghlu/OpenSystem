/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 14, 2025.
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
#ifndef PAS_FAST_LARGE_FREE_HEAP_H
#define PAS_FAST_LARGE_FREE_HEAP_H

#include "pas_config.h"

#include "pas_cartesian_tree.h"
#include "pas_large_free.h"
#include "pas_large_free_visitor.h"

PAS_BEGIN_EXTERN_C;

struct pas_fast_large_free_heap;
struct pas_fast_large_free_heap_node;
typedef struct pas_fast_large_free_heap pas_fast_large_free_heap;
typedef struct pas_fast_large_free_heap_node pas_fast_large_free_heap_node;

typedef uintptr_t pas_fast_large_free_heap_end_hashtable_key;
typedef pas_fast_large_free_heap_node* pas_fast_large_free_heap_end_hashtable_entry;

/* This uses a Cartesian tree with X = address and Y = size. */

struct pas_fast_large_free_heap_node {
    pas_cartesian_tree_node tree_node;
    pas_large_free free;
};

struct pas_fast_large_free_heap {
    pas_cartesian_tree tree;
    size_t num_mapped_bytes;
};

#define PAS_FAST_LARGE_FREE_HEAP_INITIALIZER { \
        .tree = PAS_CARTESIAN_TREE_INITIALIZER, \
        .num_mapped_bytes = 0 \
    }

PAS_API void pas_fast_large_free_heap_construct(pas_fast_large_free_heap* heap);

PAS_API pas_allocation_result
pas_fast_large_free_heap_try_allocate(pas_fast_large_free_heap* heap,
                                      size_t size,
                                      pas_alignment alignment,
                                      pas_large_free_heap_config* config);

PAS_API void pas_fast_large_free_heap_deallocate(pas_fast_large_free_heap* heap,
                                                 uintptr_t begin,
                                                 uintptr_t end,
                                                 pas_zero_mode zero_mode,
                                                 pas_large_free_heap_config* config);

PAS_API void pas_fast_large_free_heap_for_each_free(pas_fast_large_free_heap* heap,
                                                    pas_large_free_visitor visitor,
                                                    void* arg);

PAS_API size_t pas_fast_large_free_heap_get_num_free_bytes(pas_fast_large_free_heap* heap);

static inline size_t pas_fast_large_free_heap_get_num_mapped_bytes(
    pas_fast_large_free_heap* heap)
{
    return heap->num_mapped_bytes;
}

PAS_API void pas_fast_large_free_heap_validate(pas_fast_large_free_heap* heap);
PAS_API void pas_fast_large_free_heap_dump_to_printf(pas_fast_large_free_heap* heap);

PAS_END_EXTERN_C;

#endif /* PAS_FAST_LARGE_FREE_HEAP_H */

