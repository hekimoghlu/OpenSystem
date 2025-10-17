/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 11, 2022.
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
#ifndef PAS_SEGREGATED_HEAP_INLINES_H
#define PAS_SEGREGATED_HEAP_INLINES_H

#include "pas_compact_expendable_memory.h"
#include "pas_large_expendable_memory.h"
#include "pas_segregated_heap.h"
#include "pas_segregated_size_directory.h"

PAS_BEGIN_EXTERN_C;

PAS_API pas_segregated_size_directory* pas_segregated_heap_size_directory_for_index_slow(
    pas_segregated_heap* heap,
    size_t index,
    unsigned* cached_index,
    const pas_heap_config* config);

static PAS_ALWAYS_INLINE pas_segregated_size_directory*
pas_segregated_heap_size_directory_for_index(
    pas_segregated_heap* heap,
    size_t index,
    unsigned* cached_index,
    const pas_heap_config* config)
{
    pas_compact_atomic_segregated_size_directory_ptr* index_to_size_directory;
    pas_segregated_size_directory* result;
    
    if (index >= (size_t)heap->small_index_upper_bound)
        goto slow;
    
    index_to_size_directory = heap->index_to_small_size_directory;
    if (!index_to_size_directory) {
        /* Code that holds no locks may see this since we have no ordering between when the
           upper_bound is set and when this is set. */
        goto slow;
    }
    
    result = pas_compact_atomic_segregated_size_directory_ptr_load(index_to_size_directory + index);
    if (result)
        return result;

    /* It's possible for basic_size_directory to be set, the size directory lookup table to be created, and
       for the entry corresponding to the basic_size_directory to be unset in the lookup table! */

slow:
    return pas_segregated_heap_size_directory_for_index_slow(heap, index, cached_index, config);
}

static PAS_ALWAYS_INLINE pas_segregated_size_directory*
pas_segregated_heap_size_directory_for_size(
    pas_segregated_heap* heap,
    size_t size,
    pas_heap_config config,
    unsigned* cached_index)
{
    size_t index;
    
    index = pas_segregated_heap_index_for_size(size, config);
    
    return pas_segregated_heap_size_directory_for_index(heap, index, cached_index, config.config_ptr);
}

static PAS_ALWAYS_INLINE bool pas_segregated_heap_touch_lookup_tables(
    pas_segregated_heap* heap, pas_expendable_memory_touch_kind kind)
{
    pas_segregated_heap_rare_data* data;
    bool result;

    result = false;
    
    if (!heap->runtime_config->statically_allocated) {
        unsigned small_index_upper_bound;

        small_index_upper_bound = heap->small_index_upper_bound;
        if (small_index_upper_bound) {
            pas_allocator_index* index_to_small_allocator_index;
            pas_compact_atomic_segregated_size_directory_ptr* index_to_small_size_directory;

            index_to_small_allocator_index = heap->index_to_small_allocator_index;
            if (index_to_small_allocator_index) {
                result |= pas_large_expendable_memory_touch(
                    index_to_small_allocator_index,
                    sizeof(pas_allocator_index) * small_index_upper_bound,
                    kind);
            }

            index_to_small_size_directory = heap->index_to_small_size_directory;
            if (index_to_small_size_directory) {
                result |= pas_large_expendable_memory_touch(
                    index_to_small_size_directory,
                    sizeof(pas_compact_atomic_segregated_size_directory_ptr) * small_index_upper_bound,
                    kind);
            }
        }
    }

    data = pas_segregated_heap_rare_data_ptr_load(&heap->rare_data);
    if (data) {
        unsigned num_medium_directories;
        pas_segregated_heap_medium_directory_tuple* tuples;

        num_medium_directories = data->num_medium_directories;
        tuples = pas_segregated_heap_medium_directory_tuple_ptr_load(
            &data[pas_depend(num_medium_directories)].medium_directories);

        if (num_medium_directories) {
            PAS_ASSERT(tuples);
            result |= pas_compact_expendable_memory_touch(
                tuples,
                num_medium_directories * sizeof(pas_segregated_heap_medium_directory_tuple),
                kind);
        }
    }

    return result;
}

PAS_END_EXTERN_C;

#endif /* PAS_SEGREGATED_HEAP_INLINES_H */
