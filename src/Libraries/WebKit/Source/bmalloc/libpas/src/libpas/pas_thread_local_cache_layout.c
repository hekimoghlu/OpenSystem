/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 3, 2025.
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

#include "pas_thread_local_cache_layout.h"

#include "pas_heap_lock.h"
#include "pas_large_utility_free_heap.h"
#include "pas_redundant_local_allocator_node.h"
#include "pas_segregated_size_directory_inlines.h"
#include "pas_utility_heap.h"
#include "pas_utils.h"

pas_thread_local_cache_layout_segment* pas_thread_local_cache_layout_first_segment = NULL;
static pas_thread_local_cache_layout_segment* pas_thread_local_cache_layout_last_segment = NULL;
static unsigned pas_thread_local_cache_layout_last_segment_size = 0;

pas_allocator_index pas_thread_local_cache_layout_next_allocator_index =
    PAS_LOCAL_ALLOCATOR_UNSELECTED_NUM_INDICES;
pas_thread_local_cache_layout_hashtable pas_thread_local_cache_layout_hashtable_instance =
    PAS_HASHTABLE_INITIALIZER;

PAS_DEFINE_LOCK(pas_thread_local_cache_layout_hashtable);

pas_allocator_index pas_thread_local_cache_layout_add_node(pas_thread_local_cache_layout_node node)
{
    static const bool verbose = false;

    pas_allocator_index result;
    bool did_overflow;
    pas_compact_thread_local_cache_layout_node compact_node;

    pas_heap_lock_assert_held();

    switch (pas_thread_local_cache_layout_node_get_kind(node)) {
    case pas_thread_local_cache_layout_segregated_size_directory_node_kind:
    case pas_thread_local_cache_layout_redundant_local_allocator_node_kind:
        PAS_ASSERT(!pas_thread_local_cache_layout_node_get_allocator_index_generic(node));
        break;
    case pas_thread_local_cache_layout_local_view_cache_node_kind:
        PAS_ASSERT(
            pas_thread_local_cache_layout_node_get_allocator_index_generic(node)
            == (pas_allocator_index)UINT_MAX);
        break;
    }

    result = pas_thread_local_cache_layout_next_allocator_index;

    PAS_ASSERT(result < (pas_allocator_index)UINT_MAX);
    pas_thread_local_cache_layout_node_set_allocator_index(node, result);

    if (verbose) {
        pas_log("pas_thread_local_cache_layout_add_node: node = %p, allocator_index = %u, "
                "num_allocator_indices = %u\n",
                node, result, pas_thread_local_cache_layout_node_num_allocator_indices(node));
    }

    did_overflow = __builtin_add_overflow(
        pas_thread_local_cache_layout_next_allocator_index,
        pas_thread_local_cache_layout_node_num_allocator_indices(node),
        &pas_thread_local_cache_layout_next_allocator_index);
    PAS_ASSERT(!did_overflow);

    if (!pas_thread_local_cache_layout_last_segment || pas_thread_local_cache_layout_last_segment_size == PAS_THREAD_LOCAL_CACHE_LAYOUT_SEGMENT_SIZE) {
        pas_thread_local_cache_layout_segment* segment;

        segment = (pas_thread_local_cache_layout_segment*)pas_utility_heap_allocate(sizeof(pas_thread_local_cache_layout_segment), "pas_thread_local_cache_layout_segment");
        pas_zero_memory(segment, sizeof(pas_thread_local_cache_layout_segment));

        pas_compact_atomic_thread_local_cache_layout_node_store(&segment->nodes[0], node);
        pas_thread_local_cache_layout_last_segment_size = 1;

        pas_fence();
        if (!pas_thread_local_cache_layout_last_segment) {
            PAS_ASSERT(!pas_thread_local_cache_layout_first_segment);
            PAS_ASSERT(result == PAS_LOCAL_ALLOCATOR_UNSELECTED_NUM_INDICES);
            pas_thread_local_cache_layout_first_segment = segment;
            pas_thread_local_cache_layout_last_segment = segment;
        } else {
            pas_thread_local_cache_layout_last_segment->next = segment;
            pas_thread_local_cache_layout_last_segment = segment;
        }
    } else {
        PAS_ASSERT(pas_thread_local_cache_layout_last_segment);
        PAS_ASSERT(result > PAS_LOCAL_ALLOCATOR_UNSELECTED_NUM_INDICES);
        pas_fence();
        pas_compact_atomic_thread_local_cache_layout_node_store(&pas_thread_local_cache_layout_last_segment->nodes[pas_thread_local_cache_layout_last_segment_size++], node);
    }

    pas_thread_local_cache_layout_hashtable_lock_lock();
    pas_compact_thread_local_cache_layout_node_store(&compact_node, node);
    pas_thread_local_cache_layout_hashtable_add_new(
        &pas_thread_local_cache_layout_hashtable_instance, compact_node,
        NULL, &pas_large_utility_free_heap_allocation_config);
    pas_thread_local_cache_layout_hashtable_lock_unlock();
    
    return result;
}

pas_allocator_index pas_thread_local_cache_layout_add(
    pas_segregated_size_directory* directory)
{
    static const bool verbose = false;

    if (verbose) {
        pas_log("Adding TLC layout node for directory = %p, object_size = %u, page_config_kind = %s\n",
                directory, directory->object_size,
                pas_segregated_page_config_kind_get_string(directory->base.page_config_kind));
    }
    
    return pas_thread_local_cache_layout_add_node(
        pas_wrap_segregated_size_directory(directory));
}

pas_allocator_index pas_thread_local_cache_layout_duplicate(
    pas_segregated_size_directory* directory)
{
    pas_redundant_local_allocator_node* redundant_node;

    redundant_node = pas_redundant_local_allocator_node_create(directory);

    return pas_thread_local_cache_layout_add_node(
        pas_wrap_redundant_local_allocator_node(redundant_node));
}

pas_allocator_index pas_thread_local_cache_layout_add_view_cache(
    pas_segregated_size_directory* directory)
{
    return pas_thread_local_cache_layout_add_node(pas_wrap_local_view_cache_node(directory));
}

pas_thread_local_cache_layout_node pas_thread_local_cache_layout_get_node_for_index(pas_allocator_index index)
{
    pas_compact_thread_local_cache_layout_node compact_node;

    pas_thread_local_cache_layout_hashtable_lock_lock();
    compact_node = pas_thread_local_cache_layout_hashtable_get(
        &pas_thread_local_cache_layout_hashtable_instance, index);
    pas_thread_local_cache_layout_hashtable_lock_unlock();

    return pas_compact_thread_local_cache_layout_node_load(&compact_node);
}

pas_thread_local_cache_layout_node pas_thread_local_cache_layout_get_last_node(void)
{
    pas_heap_lock_assert_held();
    if (!pas_thread_local_cache_layout_last_segment)
        return NULL;
    PAS_ASSERT(pas_thread_local_cache_layout_last_segment_size);
    return pas_thread_local_cache_layout_segment_get_node(pas_thread_local_cache_layout_last_segment, pas_thread_local_cache_layout_last_segment_size - 1);
}

#endif /* LIBPAS_ENABLED */
