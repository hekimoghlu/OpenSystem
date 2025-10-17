/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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
#include "pas_config.h"

#if LIBPAS_ENABLED

#include "pas_enumerable_range_list.h"

#include "pas_heap_lock.h"
#include "pas_immortal_heap.h"

void pas_enumerable_range_list_append(pas_enumerable_range_list* list,
                                      pas_range range)
{
    pas_enumerable_range_list_chunk* chunk;

    pas_heap_lock_assert_held();

    if (pas_range_is_empty(range))
        return;

    chunk = pas_compact_atomic_enumerable_range_list_chunk_ptr_load(&list->head);

    if (!chunk || chunk->num_entries >= PAS_ENUMERABLE_RANGE_LIST_CHUNK_SIZE) {
        pas_enumerable_range_list_chunk* new_chunk;
        
        PAS_ASSERT_WITH_DETAIL(!chunk || chunk->num_entries == PAS_ENUMERABLE_RANGE_LIST_CHUNK_SIZE);

        new_chunk = pas_immortal_heap_allocate(sizeof(pas_enumerable_range_list_chunk),
                                               "pas_enumerable_range_list_chunk",
                                               pas_object_allocation);
        pas_compact_atomic_enumerable_range_list_chunk_ptr_store(&new_chunk->next, chunk);
        new_chunk->num_entries = 0;
        pas_compiler_fence();
        pas_compact_atomic_enumerable_range_list_chunk_ptr_store(&list->head, new_chunk);
        chunk = new_chunk;
    }

    PAS_ASSERT_WITH_DETAIL(chunk->num_entries < PAS_ENUMERABLE_RANGE_LIST_CHUNK_SIZE);

    chunk->entries[chunk->num_entries] = range;
    pas_compiler_fence();
    chunk->num_entries++;
}

bool pas_enumerable_range_list_iterate(
    pas_enumerable_range_list* list,
    pas_enumerable_range_list_iterate_callback callback,
    void* arg)
{
    pas_enumerable_range_list_chunk* chunk;

    for (chunk = pas_compact_atomic_enumerable_range_list_chunk_ptr_load(&list->head);
         chunk;
         chunk = pas_compact_atomic_enumerable_range_list_chunk_ptr_load(&chunk->next)) {
        size_t index;

        PAS_ASSERT_WITH_DETAIL(chunk->num_entries <= PAS_ENUMERABLE_RANGE_LIST_CHUNK_SIZE);

        for (index = chunk->num_entries; index--;) {
            pas_range range;

            range = chunk->entries[index];

            if (!callback(range, arg))
                return false;
        }
    }

    return true;
}

bool pas_enumerable_range_list_iterate_remote(
    pas_enumerable_range_list* remote_list,
    pas_enumerator* enumerator,
    pas_enumerable_range_list_iterate_remote_callback callback,
    void* arg)
{
    pas_enumerable_range_list* list;
    pas_enumerable_range_list_chunk* chunk;

    list = pas_enumerator_read(enumerator, remote_list, sizeof(pas_enumerable_range_list));
    if (!list)
        return false;

    for (chunk = pas_compact_atomic_enumerable_range_list_chunk_ptr_load_remote(enumerator,
                                                                                &list->head);
         chunk;
         chunk = pas_compact_atomic_enumerable_range_list_chunk_ptr_load_remote(enumerator,
                                                                                &chunk->next)) {
        size_t index;

        PAS_ASSERT_WITH_DETAIL(chunk->num_entries <= PAS_ENUMERABLE_RANGE_LIST_CHUNK_SIZE);

        for (index = chunk->num_entries; index--;) {
            pas_range range;

            range = chunk->entries[index];

            if (!callback(enumerator, range, arg))
                return false;
        }
    }

    return true;
}

#endif /* LIBPAS_ENABLED */
