/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 16, 2023.
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

#include "pas_deferred_decommit_log.h"

#include "pas_bootstrap_free_heap.h"
#include "pas_log.h"
#include "pas_page_malloc.h"
#include "pas_physical_memory_transaction.h"
#include <stdio.h>

static const bool verbose = false;

void pas_deferred_decommit_log_construct(pas_deferred_decommit_log* log,
                                         pas_lock** locks_already_held,
                                         size_t num_locks_already_held,
                                         pas_physical_memory_transaction* transaction)
{
    if (verbose)
        pas_log("Constructing decommit log at %p\n", log);

    /* We assume that if there is a transaction, then that's the only thing that can cause
       us to already hold locks. */
    if (transaction) {
        PAS_ASSERT(!locks_already_held);
        PAS_ASSERT(!num_locks_already_held);
    }

    pas_virtual_range_min_heap_construct(&log->impl);
    log->total = 0;
    log->common_lock_hold_count = 0;
    if (transaction) {
        log->locks_already_held = &transaction->lock_held;
        log->num_locks_already_held = 1;
    } else {
        log->locks_already_held = locks_already_held;
        log->num_locks_already_held = num_locks_already_held;
    }
    log->transaction = transaction;
}

void pas_deferred_decommit_log_destruct(pas_deferred_decommit_log* log,
                                        pas_lock_hold_mode heap_lock_hold_mode)
{
    pas_allocation_config allocation_config;
    
    if (verbose)
        pas_log("Destructing decommit log at %p\n", log);
    
    PAS_ASSERT(!log->impl.size);
    PAS_ASSERT(!log->common_lock_hold_count);
    
    pas_bootstrap_free_heap_allocation_config_construct(&allocation_config,
                                                        heap_lock_hold_mode);
    pas_virtual_range_min_heap_destruct(&log->impl, &allocation_config);
}

static bool already_holds_lock(pas_deferred_decommit_log* log,
                               pas_lock* lock_ptr)
{
    size_t index;
    for (index = log->num_locks_already_held; index--;) {
        if (log->locks_already_held[index] == lock_ptr)
            return true;
    }
    return false;
}

bool pas_deferred_decommit_log_lock_for_adding(pas_deferred_decommit_log* log,
                                               pas_lock* lock_ptr,
                                               pas_lock_hold_mode heap_lock_hold_mode)
{
    if (already_holds_lock(log, lock_ptr))
        return true;
    
    if (verbose)
        pas_log("Adding range with lock %p to %p\n", lock_ptr, log);
    
    if (lock_ptr == &pas_virtual_range_common_lock
        && log->common_lock_hold_count)
        log->common_lock_hold_count++;
    else {
        if (heap_lock_hold_mode == pas_lock_is_not_held
            && !log->num_locks_already_held
            && !log->impl.size)
            pas_lock_lock(lock_ptr);
        else {
            if (!pas_lock_try_lock(lock_ptr)) {
                if (log->transaction) {
                    pas_physical_memory_transaction_did_fail_to_acquire_lock(
                        log->transaction, lock_ptr);
                }
                return false;
            }
        }
        
        if (lock_ptr == &pas_virtual_range_common_lock)
            log->common_lock_hold_count++;
    }

    return true;
}

bool pas_deferred_decommit_log_add(pas_deferred_decommit_log* log,
                                   pas_virtual_range range,
                                   pas_lock_hold_mode heap_lock_hold_mode)
{
    if (range.lock_ptr) {
        pas_lock* lock_ptr;
        
        lock_ptr = range.lock_ptr;

        if (!pas_deferred_decommit_log_lock_for_adding(log, lock_ptr, heap_lock_hold_mode))
            return false;
    }

    pas_deferred_decommit_log_add_already_locked(log, range, heap_lock_hold_mode);
    
    return true;
}

void pas_deferred_decommit_log_add_already_locked(pas_deferred_decommit_log* log,
                                                  pas_virtual_range range,
                                                  pas_lock_hold_mode heap_lock_hold_mode)
{
    pas_allocation_config allocation_config;

    if (verbose)
        pas_log("Log %p adding range %p...%p.\n", log, (void*)range.begin, (void*)range.end);
    
    log->total += pas_virtual_range_size(range);
    
    pas_bootstrap_free_heap_allocation_config_construct(&allocation_config,
                                                        heap_lock_hold_mode);
    pas_virtual_range_min_heap_add(&log->impl, range, &allocation_config);
}

bool pas_deferred_decommit_log_add_maybe_locked(pas_deferred_decommit_log* log,
                                                pas_virtual_range range,
                                                pas_range_locked_mode range_locked_mode,
                                                pas_lock_hold_mode heap_lock_hold_mode)
{
    switch (range_locked_mode) {
    case pas_range_is_locked:
        pas_deferred_decommit_log_add_already_locked(log, range, heap_lock_hold_mode);
        return true;
    case pas_range_is_not_locked:
        return pas_deferred_decommit_log_add(log, range, heap_lock_hold_mode);
    }
    PAS_ASSERT(!"Should not be reached");
    return false;
}

void pas_deferred_decommit_log_unlock_after_aborted_add(pas_deferred_decommit_log* log,
                                                        pas_lock* lock_ptr)
{
    if (already_holds_lock(log, lock_ptr))
        return;

    /* We could support using this with the common lock - but it so happens that nobody would need
       that functionality right now. So we save ourselves the trouble and just assert it won't
       happen.
    
       This won't happen because this function is only used by the shared page directory, which is
       going to use this to unlock a shared view commit lock. */
    PAS_ASSERT(lock_ptr != &pas_virtual_range_common_lock);

    pas_lock_unlock(lock_ptr);
}

static void decommit_all(pas_deferred_decommit_log* log,
                         bool for_real)
{
    size_t start_index;
    size_t index;
    
    pas_virtual_range_min_heap_sort_descending(&log->impl);
    
    for (start_index = log->impl.size; start_index;) {
        pas_virtual_range range;
        size_t end_index;

        range = *pas_virtual_range_min_heap_get_ptr_by_index(&log->impl, start_index);
        
        end_index = start_index;
        for (;;) {
            size_t next_index;
            pas_virtual_range next_range;
            
            next_index = end_index - 1;
            if (!next_index)
                break;
            
            next_range = *pas_virtual_range_min_heap_get_ptr_by_index(&log->impl, next_index);
            
            PAS_ASSERT(!pas_virtual_range_overlaps(range, next_range));
            PAS_ASSERT(next_range.begin >= range.end);
            
            if (next_range.begin != range.end)
                break;

            range.end = next_range.end;
            end_index = next_index;
        }
        
        if (for_real) {
            if (verbose) {
                pas_log("Decommitting %p...%p.\n",
                        (void*)range.begin,
                        (void*)range.end);
            }
            
            pas_page_malloc_decommit((void*)range.begin, pas_virtual_range_size(range), range.mmap_capability);
        }
        
        PAS_ASSERT(end_index);
        PAS_ASSERT(end_index <= start_index);
        
        start_index = end_index - 1;
    }

    for (index = log->impl.size; index; index--) {
        pas_virtual_range range;
        pas_lock* lock_ptr;
        
        range = *pas_virtual_range_min_heap_get_ptr_by_index(&log->impl, index);
        
        lock_ptr = range.lock_ptr;
        if (!lock_ptr)
            continue;
        
        if (already_holds_lock(log, lock_ptr))
            continue;
        
        if (verbose)
            pas_log("Unlocking range with lock %p to %p\n", lock_ptr, log);
        
        if (lock_ptr == &pas_virtual_range_common_lock) {
            PAS_ASSERT(log->common_lock_hold_count);
            if (--log->common_lock_hold_count)
                continue;
        }
        
        pas_lock_unlock(lock_ptr);
    }
    
    log->impl.size = 0;
    log->total = 0;
}

void pas_deferred_decommit_log_decommit_all(pas_deferred_decommit_log* log)
{
    bool for_real = true;
    decommit_all(log, for_real);
}

void pas_deferred_decommit_log_pretend_to_decommit_all(pas_deferred_decommit_log* log)
{
    bool for_real = false;
    decommit_all(log, for_real);
}

#endif /* LIBPAS_ENABLED */
