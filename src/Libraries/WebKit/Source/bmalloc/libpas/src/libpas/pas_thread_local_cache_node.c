/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 30, 2024.
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

#include "pas_thread_local_cache_node.h"

#include "pas_heap_lock.h"
#include "pas_immortal_heap.h"
#include "pas_log.h"

pas_thread_local_cache_node* pas_thread_local_cache_node_first_free;
pas_thread_local_cache_node* pas_thread_local_cache_node_first;

static unsigned num_nodes;

pas_thread_local_cache_node* pas_thread_local_cache_node_allocate(void)
{
    static const bool verbose = false;
    
    pas_thread_local_cache_node* result;
    
    pas_heap_lock_assert_held();

    result = pas_thread_local_cache_node_first_free;
    if (!result) {
        const size_t interference_padding = 64;
        
        size_t size;

        size = PAS_MAX(
            interference_padding,
            pas_round_up_to_next_power_of_2(sizeof(pas_thread_local_cache_node)));

        if (verbose)
            pas_log("Allocating TLC node with size/alignment = %zu.\n", size);
        
        result = pas_immortal_heap_allocate_with_alignment(
            size, size,
            "pas_thread_local_cache_node",
            pas_object_allocation);
        
        result->next_free = NULL;
        
        result->next = pas_thread_local_cache_node_first;

        pas_lock_construct(&result->page_lock);
        pas_lock_construct(&result->scavenger_lock);

        if (verbose)
            pas_log("TLC page lock: %p\n", &result->page_lock);
        
        result->cache = NULL; /* This is set by the caller. */

        pas_fence(); /* This means we can iterate the list of caches without holding locks. */
        
        pas_thread_local_cache_node_first = result;

        if (verbose)
            pas_log("Allocated new node %p (num live = %u)\n", result, ++num_nodes);
        return result;
    }
    
    pas_thread_local_cache_node_first_free = result->next_free;
    result->next_free = NULL;
    result->cache = NULL;
    if (verbose)
        pas_log("Allocated existing node %p (num live = %u)\n", result, ++num_nodes);
    
    return result;
}

void pas_thread_local_cache_node_deallocate(pas_thread_local_cache_node* node)
{
    static const bool verbose = false;
    
    PAS_ASSERT(!node->next_free);
    pas_heap_lock_assert_held();

    if (verbose)
        pas_log("Deallocating node %p (num live = %u)\n", node, --num_nodes);
    
    node->cache = NULL;
    pas_compiler_fence();
    
    node->next_free = pas_thread_local_cache_node_first_free;
    pas_thread_local_cache_node_first_free = node;
}

#endif /* LIBPAS_ENABLED */
