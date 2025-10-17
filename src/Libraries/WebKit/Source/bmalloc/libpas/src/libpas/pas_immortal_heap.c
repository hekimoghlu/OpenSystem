/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 16, 2025.
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

#include "pas_immortal_heap.h"

#include "pas_allocation_callbacks.h"
#include "pas_compact_heap_reservation.h"
#include "pas_heap_lock.h"

uintptr_t pas_immortal_heap_current;
uintptr_t pas_immortal_heap_end;
size_t pas_immortal_heap_allocated_external = 0;
size_t pas_immortal_heap_allocated_internal = 0;
size_t pas_immortal_heap_allocation_granule = 65536;

static bool bump_is_ok(uintptr_t bump,
                       size_t size)
{
    return bump <= pas_immortal_heap_end
        && bump >= pas_immortal_heap_current
        && pas_immortal_heap_end - bump >= size;
}

void* pas_immortal_heap_allocate_with_manual_alignment(size_t size,
                                                       size_t alignment,
                                                       const char* name,
                                                       pas_allocation_kind allocation_kind)
{
    static const bool verbose = PAS_SHOULD_LOG(PAS_LOG_IMMORTAL_HEAPS);
    static const unsigned verbosity = 0;

    uintptr_t aligned_bump;

    pas_heap_lock_assert_held();
    
    aligned_bump = pas_round_up_to_power_of_2(pas_immortal_heap_current, alignment);
    if (!bump_is_ok(aligned_bump, size)) {
        size_t allocation_size;
        pas_aligned_allocation_result allocation_result;

        allocation_size = size + pas_immortal_heap_allocation_granule;

        allocation_result = pas_compact_heap_reservation_try_allocate(allocation_size, alignment);
        PAS_ASSERT(allocation_result.result);
        PAS_ASSERT(allocation_result.result_size == allocation_size);
        PAS_ASSERT(!allocation_result.right_padding_size);
        
        pas_immortal_heap_current = (uintptr_t)allocation_result.result;
        pas_immortal_heap_end = pas_immortal_heap_current + allocation_size;

        pas_immortal_heap_allocated_external += allocation_size + allocation_result.left_padding_size;

        aligned_bump = pas_immortal_heap_current;

        PAS_ASSERT(bump_is_ok(aligned_bump, size));
        PAS_ASSERT(pas_is_aligned(aligned_bump, alignment));
    }

    pas_immortal_heap_current = aligned_bump + size;

    pas_did_allocate((void*)aligned_bump, size, pas_immortal_heap_kind, name, allocation_kind);
    pas_immortal_heap_allocated_internal += size;

    if (verbose) {
        pas_log("pas_immortal_heap allocated %zu for %s at %p.\n", size, name, (void*)aligned_bump);
        if (verbose && verbosity >= 2) {
            pas_log("immortal heap internal size: %zu.\n", pas_immortal_heap_allocated_internal);
            pas_log("immortal heap external size: %zu.\n", pas_immortal_heap_allocated_external);
        }
    }

    PAS_PROFILE(IMMORTAL_HEAP_ALLOCATION, aligned_bump, size);

    return (void*)aligned_bump;
}

void* pas_immortal_heap_allocate_with_alignment(size_t size,
                                                size_t alignment,
                                                const char* name,
                                                pas_allocation_kind allocation_kind)
{
    static const bool verbose = PAS_SHOULD_LOG(PAS_LOG_IMMORTAL_HEAPS);

    void* result;
    result = pas_immortal_heap_allocate_with_manual_alignment(
        size, PAS_MAX(alignment, PAS_INTERNAL_MIN_ALIGN), name, allocation_kind);
    if (verbose)
        pas_log("immortal allocated = %p.\n", result);
    PAS_ASSERT(pas_is_aligned((uintptr_t)result, PAS_INTERNAL_MIN_ALIGN));
    return result;
}

void* pas_immortal_heap_allocate(size_t size,
                                 const char* name,
                                 pas_allocation_kind allocation_kind)
{
    return pas_immortal_heap_allocate_with_alignment(size, 1, name, allocation_kind);
}

void* pas_immortal_heap_hold_lock_and_allocate(size_t size,
                                               const char* name,
                                               pas_allocation_kind allocation_kind)
{
    void* result;
    pas_heap_lock_lock();
    result = pas_immortal_heap_allocate(size, name, allocation_kind);
    pas_heap_lock_unlock();
    return result;
}

void* pas_immortal_heap_allocate_with_heap_lock_hold_mode(size_t size,
                                                          const char* name,
                                                          pas_allocation_kind allocation_kind,
                                                          pas_lock_hold_mode heap_lock_hold_mode)
{
    void* result;
    pas_heap_lock_lock_conditionally(heap_lock_hold_mode);
    result = pas_immortal_heap_allocate(size, name, allocation_kind);
    pas_heap_lock_unlock_conditionally(heap_lock_hold_mode);
    return result;
}

void* pas_immortal_heap_allocate_with_alignment_and_heap_lock_hold_mode(
    size_t size,
    size_t alignment,
    const char* name,
    pas_allocation_kind allocation_kind,
    pas_lock_hold_mode heap_lock_hold_mode)
{
    void* result;
    pas_heap_lock_lock_conditionally(heap_lock_hold_mode);
    result = pas_immortal_heap_allocate_with_alignment(size, alignment, name, allocation_kind);
    pas_heap_lock_unlock_conditionally(heap_lock_hold_mode);
    return result;
}

#endif /* LIBPAS_ENABLED */
