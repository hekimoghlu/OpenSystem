/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 12, 2022.
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

#include "pas_compute_summary_object_callbacks.h"

#include "pas_heap_lock.h"
#include "pas_heap_summary.h"
#include "pas_large_sharing_pool.h"

bool pas_compute_summary_live_object_callback(uintptr_t begin,
                                              uintptr_t end,
                                              void* arg)
{
    pas_heap_summary* summary_ptr;
    pas_heap_summary my_summary;
    
    pas_heap_lock_assert_held();
    
    summary_ptr = arg;
    
    my_summary = pas_large_sharing_pool_compute_summary(
        pas_range_create(begin, end),
        pas_large_sharing_pool_compute_summary_known_allocated,
        pas_lock_is_held);
    PAS_ASSERT(!my_summary.free);
    PAS_ASSERT(my_summary.allocated == end - begin);
    
    *summary_ptr = pas_heap_summary_add(*summary_ptr, my_summary);
    
    return true;
}

bool pas_compute_summary_live_object_callback_without_physical_sharing(uintptr_t begin,
                                                                       uintptr_t end,
                                                                       void* arg)
{
    pas_heap_summary* summary_ptr;
    pas_heap_summary my_summary;
    
    pas_heap_lock_assert_held();
    
    summary_ptr = arg;
    
    my_summary = pas_heap_summary_create_empty();
    my_summary.allocated = end - begin;
    my_summary.committed = end - begin;
    
    *summary_ptr = pas_heap_summary_add(*summary_ptr, my_summary);
    
    return true;
}

bool (*pas_compute_summary_live_object_callback_for_config(const pas_heap_config* config))(
    uintptr_t begin,
    uintptr_t end,
    void* arg)
{
    if (config->aligned_allocator_talks_to_sharing_pool)
        return pas_compute_summary_live_object_callback;
    return pas_compute_summary_live_object_callback_without_physical_sharing;
}

bool pas_compute_summary_dead_object_callback(pas_large_free free,
                                              void* arg)
{
    pas_heap_summary* summary_ptr;
    pas_heap_summary my_summary;

    pas_heap_lock_assert_held();
    
    summary_ptr = arg;
    
    my_summary = pas_large_sharing_pool_compute_summary(
        pas_range_create(free.begin, free.end),
        pas_large_sharing_pool_compute_summary_known_free,
        pas_lock_is_held);
    PAS_ASSERT(!my_summary.allocated);
    PAS_ASSERT(my_summary.free == free.end - free.begin);
    
    *summary_ptr = pas_heap_summary_add(*summary_ptr, my_summary);
    
    return true;
}

bool pas_compute_summary_dead_object_callback_without_physical_sharing(pas_large_free free,
                                                                       void* arg)
{
    pas_heap_summary* summary_ptr;
    pas_heap_summary my_summary;

    pas_heap_lock_assert_held();
    
    summary_ptr = arg;
    
    my_summary = pas_heap_summary_create_empty();
    my_summary.free = free.end - free.begin;
    my_summary.committed = free.end - free.begin;
    my_summary.free_ineligible_for_decommit = free.end - free.begin;
    
    *summary_ptr = pas_heap_summary_add(*summary_ptr, my_summary);
    
    return true;
}

bool (*pas_compute_summary_dead_object_callback_for_config(const pas_heap_config* config))(
    pas_large_free free,
    void* arg)
{
    if (config->aligned_allocator_talks_to_sharing_pool)
        return pas_compute_summary_dead_object_callback;
    return pas_compute_summary_dead_object_callback_without_physical_sharing;
}

#endif /* LIBPAS_ENABLED */
