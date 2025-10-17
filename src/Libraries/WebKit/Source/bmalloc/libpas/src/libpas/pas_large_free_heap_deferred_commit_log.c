/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 11, 2022.
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

#include "pas_large_free_heap_deferred_commit_log.h"

#include "pas_bootstrap_free_heap.h"
#include "pas_debug_spectrum.h"
#include "pas_page_malloc.h"
#include "pas_physical_memory_transaction.h"
#include "pas_stream.h"
#include "pas_virtual_range.h"

void pas_large_free_heap_deferred_commit_log_construct(
    pas_large_free_heap_deferred_commit_log* log)
{
    pas_large_virtual_range_min_heap_construct(&log->impl);
    log->total = 0;
}

void pas_large_free_heap_deferred_commit_log_destruct(
    pas_large_free_heap_deferred_commit_log* log)
{
    pas_allocation_config allocation_config;
    
    PAS_ASSERT(!log->total);
    PAS_ASSERT(!log->impl.size);
    
    pas_bootstrap_free_heap_allocation_config_construct(&allocation_config, pas_lock_is_held);
    pas_large_virtual_range_min_heap_destruct(&log->impl, &allocation_config);
}

bool pas_large_free_heap_deferred_commit_log_add(
    pas_large_free_heap_deferred_commit_log* log,
    pas_large_virtual_range range,
    pas_physical_memory_transaction* transaction)
{
    pas_allocation_config allocation_config;
    
    if (!log->total
        && &pas_virtual_range_common_lock != transaction->lock_held
        && !pas_lock_try_lock(&pas_virtual_range_common_lock)) {
        pas_physical_memory_transaction_did_fail_to_acquire_lock(
            transaction, &pas_virtual_range_common_lock);
        return false;
    }
    
    log->total += pas_large_virtual_range_size(range);
    
    pas_bootstrap_free_heap_allocation_config_construct(&allocation_config, pas_lock_is_held);
    pas_large_virtual_range_min_heap_add(&log->impl, range, &allocation_config);
    
    return true;
}

static void dump_large_commit(pas_stream* stream, void* arg)
{
    PAS_UNUSED_PARAM(arg);
    pas_stream_printf(stream, "large deferred");
}

static void commit(pas_large_virtual_range range)
{
    static const bool verbose = false;
    
    if (pas_large_virtual_range_is_empty(range))
        return;
    
    if (verbose) {
        pas_log("Committing %p...%p.\n",
               (void*)range.begin,
               (void*)range.end);
    }
    
    pas_page_malloc_commit((void*)range.begin, pas_large_virtual_range_size(range), range.mmap_capability);

    if (PAS_DEBUG_SPECTRUM_USE_FOR_COMMIT)
        pas_debug_spectrum_add(dump_large_commit, dump_large_commit, pas_large_virtual_range_size(range));
}

static void commit_all(
    pas_large_free_heap_deferred_commit_log* log,
    pas_physical_memory_transaction* transaction,
    bool for_real)
{
    pas_large_virtual_range current_range;
    
    if (!log->total)
        return;
    
    current_range = pas_large_virtual_range_create_empty();
    
    for (;;) {
        pas_large_virtual_range next_range = pas_large_virtual_range_min_heap_take_min(&log->impl);
        if (pas_large_virtual_range_is_empty(next_range))
            break;
        
        PAS_ASSERT(!pas_large_virtual_range_overlaps(current_range, next_range));
        
        if (next_range.begin == current_range.end) {
            current_range.end = next_range.end;
            continue;
        }
        
        if (for_real)
            commit(current_range);
        current_range = next_range;
    }
    
    if (for_real)
        commit(current_range);
    
    if (&pas_virtual_range_common_lock != transaction->lock_held)
        pas_lock_unlock(&pas_virtual_range_common_lock);
    
    log->total = 0;
}

void pas_large_free_heap_deferred_commit_log_commit_all(
    pas_large_free_heap_deferred_commit_log* log,
    pas_physical_memory_transaction* transaction)
{
    bool for_real = true;
    commit_all(log, transaction, for_real);
}

void pas_large_free_heap_deferred_commit_log_pretend_to_commit_all(
    pas_large_free_heap_deferred_commit_log* log,
    pas_physical_memory_transaction* transaction)
{
    bool for_real = false;
    commit_all(log, transaction, for_real);
}

#endif /* LIBPAS_ENABLED */
