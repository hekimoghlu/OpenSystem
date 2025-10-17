/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 3, 2023.
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

#include "pas_thread_local_cache_layout_node.h"

#include "pas_local_view_cache.h"
#include "pas_redundant_local_allocator_node.h"
#include "pas_segregated_size_directory_inlines.h"

pas_segregated_size_directory*
pas_thread_local_cache_layout_node_get_directory(pas_thread_local_cache_layout_node node)
{
    if (pas_is_wrapped_segregated_size_directory(node))
        return pas_unwrap_segregated_size_directory(node);

    if (pas_is_wrapped_redundant_local_allocator_node(node)){
        return pas_compact_segregated_size_directory_ptr_load_non_null(
            &pas_unwrap_redundant_local_allocator_node(node)->directory);
    }

    return pas_unwrap_local_view_cache_node(node);
}

pas_allocator_index
pas_thread_local_cache_layout_node_num_allocator_indices(pas_thread_local_cache_layout_node node)
{
    pas_segregated_size_directory* directory;

    directory = pas_thread_local_cache_layout_node_get_directory(node);
    
    if (pas_thread_local_cache_layout_node_represents_allocator(node))
        return pas_segregated_size_directory_num_allocator_indices(directory);

    return pas_segregated_size_directory_num_allocator_indices_for_allocator_size(
        PAS_LOCAL_VIEW_CACHE_SIZE(pas_segregated_size_directory_view_cache_capacity(directory)));
}

static pas_allocator_index*
allocator_index_ptr(pas_thread_local_cache_layout_node node)
{
    if (pas_is_wrapped_segregated_size_directory(node)) {
        return &pas_unwrap_segregated_size_directory(node)->allocator_index;
    }

    if (pas_is_wrapped_redundant_local_allocator_node(node))
        return &pas_unwrap_redundant_local_allocator_node(node)->allocator_index;

    return &pas_unwrap_local_view_cache_node(node)->view_cache_index;
}

pas_allocator_index
pas_thread_local_cache_layout_node_get_allocator_index_generic(pas_thread_local_cache_layout_node node)
{
    return *allocator_index_ptr(node);
}

pas_allocator_index
pas_thread_local_cache_layout_node_get_allocator_index_for_allocator(pas_thread_local_cache_layout_node node)
{
    PAS_ASSERT(pas_thread_local_cache_layout_node_represents_allocator(node));
    return pas_thread_local_cache_layout_node_get_allocator_index_generic(node);
}

pas_allocator_index
pas_thread_local_cache_layout_node_get_allocator_index_for_view_cache(pas_thread_local_cache_layout_node node)
{
    PAS_ASSERT(pas_thread_local_cache_layout_node_represents_view_cache(node));
    return pas_thread_local_cache_layout_node_get_allocator_index_generic(node);
}

void
pas_thread_local_cache_layout_node_set_allocator_index(pas_thread_local_cache_layout_node node,
                                                       pas_allocator_index index)
{
    *allocator_index_ptr(node) = index;
}

void pas_thread_local_cache_layout_node_commit_and_construct(pas_thread_local_cache_layout_node node,
                                                             pas_thread_local_cache* cache)
{
    pas_thread_local_cache_layout_node_ensure_committed(node, cache);
    
    if (pas_thread_local_cache_layout_node_represents_allocator(node)) {
        pas_local_allocator_construct(
            pas_thread_local_cache_get_local_allocator_direct_for_initialization(
                cache, pas_thread_local_cache_layout_node_get_allocator_index_for_allocator(node)),
            pas_thread_local_cache_layout_node_get_directory(node));
        return;
    }

    pas_local_view_cache_construct(
        pas_thread_local_cache_get_local_allocator_direct_for_initialization(
            cache, pas_thread_local_cache_layout_node_get_allocator_index_for_view_cache(node)),
        pas_segregated_size_directory_view_cache_capacity(
            pas_thread_local_cache_layout_node_get_directory(node)));
}

void pas_thread_local_cache_layout_node_move(pas_thread_local_cache_layout_node node,
                                             pas_thread_local_cache* to_cache,
                                             pas_thread_local_cache* from_cache)
{
    pas_allocator_index allocator_index;
    pas_local_allocator_scavenger_data* scavenger_data;

    PAS_ASSERT(pas_thread_local_cache_layout_node_is_committed(node, to_cache));
    
    allocator_index = pas_thread_local_cache_layout_node_get_allocator_index_generic(node);
    
    if (!pas_thread_local_cache_layout_node_is_committed(node, from_cache)) {
        pas_thread_local_cache_layout_node_commit_and_construct(node, to_cache);
        return;
    }
        
    scavenger_data = pas_thread_local_cache_get_local_allocator_direct(from_cache, allocator_index);

    if (scavenger_data->kind == pas_local_allocator_decommitted_kind) {
        pas_thread_local_cache_layout_node_commit_and_construct(node, to_cache);
        return;
    }
    
    if (pas_thread_local_cache_layout_node_represents_allocator(node)) {
        pas_local_allocator_move(
            pas_thread_local_cache_get_local_allocator_direct_for_initialization(to_cache, allocator_index),
            (pas_local_allocator*)scavenger_data);
        return;
    }

    pas_local_view_cache_move(
        pas_thread_local_cache_get_local_allocator_direct_for_initialization(to_cache, allocator_index),
        (pas_local_view_cache*)scavenger_data);
}

void pas_thread_local_cache_layout_node_stop(pas_thread_local_cache_layout_node node,
                                             pas_thread_local_cache* cache,
                                             pas_lock_lock_mode page_lock_mode,
                                             pas_lock_hold_mode heap_lock_hold_mode)
{
    static const bool verbose = false;
    
    pas_allocator_index allocator_index;
    void* allocator;

    allocator_index = pas_thread_local_cache_layout_node_get_allocator_index_generic(node);
    allocator = pas_thread_local_cache_get_local_allocator_direct(cache, allocator_index);
    
    if (verbose)
        pas_log("Stopping allocator %p because pas_thread_local_cache_stop_local_allocators\n", allocator);
    
    if (pas_thread_local_cache_layout_node_represents_allocator(node)) {
        pas_local_allocator_stop(allocator, page_lock_mode, heap_lock_hold_mode);
        return;
    }

    pas_local_view_cache_stop(allocator, page_lock_mode);
}

bool pas_thread_local_cache_layout_node_is_committed(pas_thread_local_cache_layout_node node,
                                                     pas_thread_local_cache* cache)
{
    pas_allocator_index allocator_index;

    allocator_index = pas_thread_local_cache_layout_node_get_allocator_index_generic(node);

    return pas_thread_local_cache_is_committed(
        cache,
        allocator_index,
        allocator_index + pas_thread_local_cache_layout_node_num_allocator_indices(node));
}

void pas_thread_local_cache_layout_node_ensure_committed(pas_thread_local_cache_layout_node node,
                                                         pas_thread_local_cache* cache)
{
    pas_allocator_index allocator_index;

    allocator_index = pas_thread_local_cache_layout_node_get_allocator_index_generic(node);

    pas_thread_local_cache_ensure_committed(
        cache,
        allocator_index,
        allocator_index + pas_thread_local_cache_layout_node_num_allocator_indices(node));
}

void pas_thread_local_cache_layout_node_prepare_to_decommit(pas_thread_local_cache_layout_node node,
                                                            pas_thread_local_cache* cache,
                                                            pas_range decommit_range)
{
    pas_allocator_index allocator_index;
    pas_allocator_index end_allocator_index;
    pas_range allocator_range;
    pas_local_allocator_scavenger_data* scavenger_data;

    PAS_ASSERT(pas_thread_local_cache_layout_node_is_committed(node, cache));

    allocator_index = pas_thread_local_cache_layout_node_get_allocator_index_generic(node);
    end_allocator_index = allocator_index + pas_thread_local_cache_layout_node_num_allocator_indices(node);
    allocator_range = pas_range_create(
        pas_thread_local_cache_offset_of_allocator(allocator_index),
        pas_thread_local_cache_offset_of_allocator(end_allocator_index));

    if (!pas_range_overlaps(allocator_range, decommit_range))
        return;
    
    scavenger_data = (pas_local_allocator_scavenger_data*)
        pas_thread_local_cache_get_local_allocator_direct(cache, allocator_index);

    pas_local_allocator_scavenger_data_prepare_to_decommit(scavenger_data);
}

#endif /* LIBPAS_ENABLED */
