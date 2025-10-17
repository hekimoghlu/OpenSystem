/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#include "pas_local_allocator_scavenger_data.h"

#include "pas_allocator_index.h"
#include "pas_baseline_allocator_table.h"
#include "pas_local_allocator.h"
#include "pas_local_view_cache.h"
#include "pas_heap_lock.h"
#include "pas_thread_local_cache.h"
#include "pas_thread_local_cache_layout.h"
#include "pas_thread_local_cache_layout_node.h"
#include "pas_thread_local_cache_node.h"

uint8_t pas_local_allocator_should_stop_count_for_suspend = 5;

bool pas_local_allocator_scavenger_data_is_baseline_allocator(pas_local_allocator_scavenger_data* data)
{
    uintptr_t data_ptr;
    uintptr_t baseline_begin;
    uintptr_t baseline_end;

    data_ptr = (uintptr_t)data;
    baseline_begin = (uintptr_t)pas_baseline_allocator_table;
    baseline_end = (uintptr_t)(pas_baseline_allocator_table + pas_baseline_allocator_table_bound);

    return data_ptr >= baseline_begin && data_ptr < baseline_end;
}

bool pas_local_allocator_scavenger_data_is_stopped(pas_local_allocator_scavenger_data* data)
{
    return pas_local_allocator_kind_is_stopped(data->kind);
}

void pas_local_allocator_scavenger_data_commit_if_necessary_slow(
    pas_local_allocator_scavenger_data* data,
    pas_local_allocator_scavenger_data_commit_if_necessary_slow_mode mode,
    pas_local_allocator_kind expected_kind)
{
    pas_thread_local_cache* cache;
    pas_allocator_index index;
    pas_thread_local_cache_layout_node layout_node;
    pas_lock_hold_mode scavenger_lock_hold_mode;
    bool is_in_use;

    PAS_ASSERT(expected_kind == pas_local_allocator_allocator_kind
               || expected_kind == pas_local_allocator_view_cache_kind);

    switch (mode) {
    case pas_local_allocator_scavenger_data_commit_if_necessary_slow_is_in_use_with_no_locks_held_mode:
        scavenger_lock_hold_mode = pas_lock_is_not_held;
        is_in_use = true;
        break;
    case pas_local_allocator_scavenger_data_commit_if_necessary_slow_is_not_in_use_with_scavenger_lock_held_mode:
        scavenger_lock_hold_mode = pas_lock_is_held;
        is_in_use = false;
        break;
    }

    /* NOTE: this can only be called when is_in_use, but it's possible that this allocator will get
       decommitted at any time, which may result in is_in_use being cleared. */

    if (pas_local_allocator_scavenger_data_is_baseline_allocator(data)) {
        PAS_ASSERT(data->kind == pas_local_allocator_stopped_allocator_kind);
        PAS_ASSERT(expected_kind == pas_local_allocator_allocator_kind);
        PAS_ASSERT(data->is_in_use == is_in_use);
        data->kind = pas_local_allocator_allocator_kind;
        return;
    }

    cache = pas_thread_local_cache_try_get();
    PAS_ASSERT(cache);
    
    if (data->kind == pas_local_allocator_stopped_allocator_kind
        || data->kind == pas_local_allocator_stopped_view_cache_kind) {
        bool done;
        
        done = false;
        
        /* Taking scavenger_lock ensures that this page will not be decommitted while we
         * change data->kind below to indicate that we're keeping the page.
         * Since data->kind is set to pas_local_allocator_decommitted_kind before decommit,
         * if it is not pas_local_allocator_decommitted_kind by this point, this allocator is not
         * decommited. */
        pas_lock_lock_conditionally(&cache->node->scavenger_lock, scavenger_lock_hold_mode);
        pas_lock_testing_assert_held(&cache->node->scavenger_lock);
        switch (data->kind) {
        case pas_local_allocator_decommitted_kind:
            break;
        case pas_local_allocator_stopped_allocator_kind:
            PAS_ASSERT(expected_kind == pas_local_allocator_allocator_kind);
            data->kind = pas_local_allocator_allocator_kind;
            done = true;
            break;
        case pas_local_allocator_stopped_view_cache_kind:
            PAS_ASSERT(expected_kind == pas_local_allocator_view_cache_kind);
            data->kind = pas_local_allocator_view_cache_kind;
            pas_local_view_cache_did_restart((pas_local_view_cache*)data);
            done = true;
            break;
        default:
            PAS_ASSERT(!"Should not be reached");
            break;
        }
        pas_lock_unlock_conditionally(&cache->node->scavenger_lock, scavenger_lock_hold_mode);
        if (done) {
            PAS_ASSERT(data->is_in_use == is_in_use);
            PAS_ASSERT(data->kind == expected_kind);
            return;
        }
    }

    PAS_ASSERT(data->kind == pas_local_allocator_decommitted_kind);

    pas_lock_lock_conditionally(&cache->node->scavenger_lock, scavenger_lock_hold_mode);
    pas_lock_testing_assert_held(&cache->node->scavenger_lock);

    PAS_ASSERT(data->kind == pas_local_allocator_decommitted_kind);

    index = pas_thread_local_cache_allocator_index_for_allocator(cache, data);
    layout_node = pas_thread_local_cache_layout_get_node_for_index(index);

    pas_thread_local_cache_layout_node_commit_and_construct(layout_node, cache);

    PAS_ASSERT(data->kind == expected_kind);
    PAS_ASSERT(data->kind == pas_local_allocator_allocator_kind
               || data->kind == pas_local_allocator_view_cache_kind);

    data->is_in_use = is_in_use;
    
    pas_lock_unlock_conditionally(&cache->node->scavenger_lock, scavenger_lock_hold_mode);
}

bool pas_local_allocator_scavenger_data_stop(
    pas_local_allocator_scavenger_data* data,
    pas_lock_lock_mode page_lock_mode,
    pas_lock_hold_mode heap_lock_hold_mode)
{
    switch (data->kind) {
    case pas_local_allocator_decommitted_kind:
    case pas_local_allocator_stopped_allocator_kind:
    case pas_local_allocator_stopped_view_cache_kind:
        return true;
    case pas_local_allocator_allocator_kind:
        return pas_local_allocator_stop((pas_local_allocator*)data, page_lock_mode, heap_lock_hold_mode);
    case pas_local_allocator_view_cache_kind:
        return pas_local_view_cache_stop((pas_local_view_cache*)data, page_lock_mode);
    }
    PAS_ASSERT(!"Should not be reached");
    return false;
}

void pas_local_allocator_scavenger_data_prepare_to_decommit(pas_local_allocator_scavenger_data* data)
{
    /* Fun fact: the allocator might have is_in_use = true when we're in there, but in that case, that
       allocator can't do anything without grabbing the scavenger_lock (because the allocator says it's
       stopped). */
    
    PAS_ASSERT(pas_local_allocator_scavenger_data_is_stopped(data));

    pas_heap_lock_assert_held();

    data->kind = pas_local_allocator_decommitted_kind;
}

#endif /* LIBPAS_ENABLED */


