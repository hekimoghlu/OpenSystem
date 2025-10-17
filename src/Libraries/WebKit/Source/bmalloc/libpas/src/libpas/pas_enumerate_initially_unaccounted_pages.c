/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 14, 2022.
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

#include "pas_enumerate_initially_unaccounted_pages.h"

#include "pas_enumerable_range_list.h"
#include "pas_large_sharing_pool.h"
#include "pas_ptr_worklist.h"
#include "pas_root.h"

static bool range_list_iterate_add_unaccounted_callback(pas_enumerator* enumerator,
                                                        pas_range range,
                                                        void* arg)
{
    PAS_ASSERT_WITH_DETAIL(!arg);
    pas_enumerator_add_unaccounted_pages(enumerator, (void*)range.begin, pas_range_size(range));
    return true;
}

static bool range_list_iterate_exclude_accounted_callback(pas_enumerator* enumerator,
                                                          pas_range range,
                                                          void* arg)
{
    PAS_ASSERT_WITH_DETAIL(!arg);
    pas_enumerator_exclude_accounted_pages(enumerator, (void*)range.begin, pas_range_size(range));
    return true;
}

bool pas_enumerate_initially_unaccounted_pages(pas_enumerator* enumerator)
{
    pas_ptr_worklist worklist;
    pas_red_black_tree* large_sharing_tree;
    pas_red_black_tree_jettisoned_nodes* large_sharing_tree_jettisoned_nodes;
    pas_large_sharing_node* large_sharing_node;
    uintptr_t* compact_heap_reservation_base;
    uintptr_t* compact_heap_reservation_bump;
    uintptr_t* compact_heap_reservation_guard_size;

    compact_heap_reservation_base = pas_enumerator_read(
        enumerator, enumerator->root->compact_heap_reservation_base, sizeof(uintptr_t));
    if (!compact_heap_reservation_base)
        return false;

    compact_heap_reservation_bump = pas_enumerator_read(
        enumerator, enumerator->root->compact_heap_reservation_bump, sizeof(size_t));
    if (!compact_heap_reservation_bump)
        return false;

    compact_heap_reservation_guard_size = pas_enumerator_read(
        enumerator, enumerator->root->compact_heap_reservation_guard_size, sizeof(size_t));
    if (!compact_heap_reservation_guard_size)
        return false;

    pas_enumerator_add_unaccounted_pages(
        enumerator,
        (char*)*compact_heap_reservation_base + *compact_heap_reservation_guard_size,
        pas_round_up_to_power_of_2(
            *compact_heap_reservation_bump - *compact_heap_reservation_guard_size,
            enumerator->root->page_malloc_alignment));

    if (!pas_enumerable_range_list_iterate_remote(
            enumerator->root->enumerable_page_malloc_page_list,
            enumerator,
            range_list_iterate_add_unaccounted_callback,
            NULL))
        return false;

    if (!pas_enumerable_range_list_iterate_remote(
            enumerator->root->payload_reservation_page_list,
            enumerator,
            range_list_iterate_exclude_accounted_callback,
            NULL))
        return false;

    large_sharing_tree = pas_enumerator_read(
        enumerator, enumerator->root->large_sharing_tree, sizeof(pas_red_black_tree));
    if (!large_sharing_tree)
        return false;

    large_sharing_tree_jettisoned_nodes = pas_enumerator_read(
        enumerator,
        enumerator->root->large_sharing_tree_jettisoned_nodes,
        sizeof(pas_red_black_tree_jettisoned_nodes));
    if (!large_sharing_tree_jettisoned_nodes)
        return false;

    pas_ptr_worklist_construct(&worklist);
    pas_ptr_worklist_push(
        &worklist,
        pas_red_black_tree_node_ptr_load_remote(enumerator, &large_sharing_tree->root),
        &enumerator->allocation_config);
    pas_ptr_worklist_push(
        &worklist,
        pas_enumerator_read_compact(
            enumerator, large_sharing_tree_jettisoned_nodes->first_rotate_jettisoned),
        &enumerator->allocation_config);
    pas_ptr_worklist_push(
        &worklist,
        pas_enumerator_read_compact(
            enumerator, large_sharing_tree_jettisoned_nodes->second_rotate_jettisoned),
        &enumerator->allocation_config);
    pas_ptr_worklist_push(
        &worklist,
        pas_enumerator_read_compact(
            enumerator, large_sharing_tree_jettisoned_nodes->remove_jettisoned),
        &enumerator->allocation_config);

    while ((large_sharing_node = pas_ptr_worklist_pop(&worklist))) {
        /* Disregard cases where the sharing node says that it has live bytes and is decommitted.
           We assume that this might happen if the sharing pool is in flux and that this really means
           that the node represents committed memory. */
        if (!large_sharing_node->is_committed && !large_sharing_node->num_live_bytes) {
            pas_enumerator_exclude_accounted_pages(
                enumerator,
                (void*)large_sharing_node->range.begin,
                pas_range_size(large_sharing_node->range));
        }

        pas_ptr_worklist_push(
            &worklist,
            pas_red_black_tree_node_ptr_load_remote(enumerator, &large_sharing_node->tree_node.left),
            &enumerator->allocation_config);
        pas_ptr_worklist_push(
            &worklist,
            pas_red_black_tree_node_ptr_load_remote(enumerator, &large_sharing_node->tree_node.right),
            &enumerator->allocation_config);
    }

    return true;
}

#endif /* LIBPAS_ENABLED */
