/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 16, 2022.
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
#ifndef PAS_SIMPLE_LARGE_FREE_HEAP_H
#define PAS_SIMPLE_LARGE_FREE_HEAP_H

#include "pas_alignment.h"
#include "pas_allocation_result.h"
#include "pas_large_free_visitor.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_large_free;
struct pas_simple_large_free_heap;
struct pas_large_free_heap_config;
typedef struct pas_large_free pas_large_free;
typedef struct pas_simple_large_free_heap pas_simple_large_free_heap;
typedef struct pas_large_free_heap_config pas_large_free_heap_config;

struct pas_simple_large_free_heap {
    pas_large_free* free_list;
    size_t free_list_size; /* in units of pas_large_free, not byte size. */
    size_t free_list_capacity; /* in units of pas_large_free, not byte size. */
    size_t num_mapped_bytes;
};

/* Note that all of these functions have to be called with the heap lock held. */

#define PAS_SIMPLE_LARGE_FREE_HEAP_INITIALIZER { \
        .free_list = NULL, \
        .free_list_size = 0, \
        .free_list_capacity = 0, \
        .num_mapped_bytes = 0 \
    }

PAS_API void pas_simple_large_free_heap_construct(pas_simple_large_free_heap* heap);

PAS_API pas_allocation_result
pas_simple_large_free_heap_try_allocate(pas_simple_large_free_heap* heap,
                                        size_t size,
                                        pas_alignment alignment,
                                        pas_large_free_heap_config* config);

PAS_API void pas_simple_large_free_heap_deallocate(pas_simple_large_free_heap* heap,
                                                   uintptr_t begin,
                                                   uintptr_t end,
                                                   pas_zero_mode zero_mode,
                                                   pas_large_free_heap_config* config);

PAS_API void pas_simple_large_free_heap_for_each_free(pas_simple_large_free_heap* heap,
                                                      pas_large_free_visitor visitor,
                                                      void* arg);

PAS_API size_t pas_simple_large_free_heap_get_num_free_bytes(pas_simple_large_free_heap* heap);

/* This is a hilarious function that only works if pasmalloc is not the system malloc. */
PAS_API void pas_simple_large_free_heap_dump_to_printf(pas_simple_large_free_heap* heap);

PAS_END_EXTERN_C;

#endif /* PAS_SIMPLE_LARGE_FREE_HEAP_H */

