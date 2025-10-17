/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 4, 2022.
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

#include "pas_large_free_heap_helpers.h"

#include "pas_allocation_callbacks.h"
#include "pas_compute_summary_object_callbacks.h"
#include "pas_fast_large_free_heap.h"
#include "pas_large_sharing_pool.h"
#include "pas_page_malloc.h"
#include "pas_page_sharing_pool.h"

bool pas_large_utility_free_heap_talks_to_large_sharing_pool = true;

static const bool verbose = PAS_SHOULD_LOG(PAS_LOG_LARGE_HEAPS);

static pas_aligned_allocation_result large_utility_aligned_allocator(size_t size,
                                                                     pas_alignment alignment,
                                                                     void* arg)
{
    size_t page_size;
    size_t aligned_size;
    pas_alignment aligned_alignment;
    pas_allocation_result allocation_result;
    pas_aligned_allocation_result result;
    pas_large_free_heap_helpers_memory_source memory_source;

    memory_source = arg;

    /* This logic is kinda like pas_large_heap_physical_page_sharing_cache, except that this
       assumes that we have to manage the physical memory synchronously (holding the heap lock). */

    page_size = pas_page_malloc_alignment();
    aligned_size = pas_round_up_to_power_of_2(size, page_size);
    aligned_alignment = pas_alignment_round_up(alignment, page_size);

    pas_zero_memory(&result, sizeof(result));

    if (pas_large_utility_free_heap_talks_to_large_sharing_pool)
        pas_physical_page_sharing_pool_take_later(aligned_size);

    allocation_result = memory_source(
        aligned_size, aligned_alignment,
        "pas_large_utility_free_heap/chunk",
        pas_delegate_allocation);

    if (!allocation_result.did_succeed) {
        pas_physical_page_sharing_pool_give_back(aligned_size);
        return result;
    }

    if (pas_large_utility_free_heap_talks_to_large_sharing_pool) {
        if (verbose) {
            pas_log("booting free %p...%p.\n",
                    (void*)allocation_result.begin,
                    (void*)(allocation_result.begin + aligned_size));
        }
        
        pas_large_sharing_pool_boot_free(
            pas_range_create(allocation_result.begin, allocation_result.begin + aligned_size),
            pas_physical_memory_is_locked_by_heap_lock,
            pas_may_mmap);
    }

    result.result = (void*)allocation_result.begin;
    result.result_size = size;
    result.left_padding = (void*)allocation_result.begin;
    result.left_padding_size = 0;
    result.right_padding = (char*)(void*)allocation_result.begin + size;
    result.right_padding_size = aligned_size - size;
    result.zero_mode = allocation_result.zero_mode;
    
    return result;
}

static void initialize_config(pas_large_free_heap_config* config,
                              pas_large_free_heap_helpers_memory_source memory_source)
{
    config->type_size = 1;
    config->min_alignment = 1;
    config->aligned_allocator = large_utility_aligned_allocator;
    config->aligned_allocator_arg = memory_source;
    config->deallocator = NULL;
    config->deallocator_arg = NULL;
}

void* pas_large_free_heap_helpers_try_allocate_with_alignment(
    pas_fast_large_free_heap* heap,
    pas_large_free_heap_helpers_memory_source memory_source,
    size_t* num_allocated_object_bytes_ptr,
    size_t* num_allocated_object_bytes_peak_ptr,
    size_t size,
    pas_alignment alignment,
    const char* name)
{
    pas_large_free_heap_config config;
    pas_allocation_result result;
    pas_heap_lock_assert_held();
    initialize_config(&config, memory_source);
    alignment = pas_alignment_round_up(alignment, PAS_INTERNAL_MIN_ALIGN);
    result = pas_fast_large_free_heap_try_allocate(heap, size, alignment, &config);
    pas_did_allocate(
        (void*)result.begin, size, pas_large_utility_free_heap_kind, name, pas_object_allocation);
    if (result.did_succeed) {
        if (pas_large_utility_free_heap_talks_to_large_sharing_pool) {
            bool commit_result;
            if (verbose) {
                pas_log("allocated and commit %p...%p.\n",
                        (void*)result.begin,
                        (void*)(result.begin + size));
            }
            commit_result = pas_large_sharing_pool_allocate_and_commit(
                pas_range_create(result.begin, result.begin + size),
                NULL,
                pas_physical_memory_is_locked_by_heap_lock,
                pas_may_mmap);
            PAS_ASSERT(commit_result);
        }
        (*num_allocated_object_bytes_ptr) += size;
        (*num_allocated_object_bytes_peak_ptr) = PAS_MAX(
            *num_allocated_object_bytes_ptr, *num_allocated_object_bytes_peak_ptr);
    }
    return (void*)result.begin;
}

void pas_large_free_heap_helpers_deallocate(
    pas_fast_large_free_heap* heap,
    pas_large_free_heap_helpers_memory_source memory_source,
    size_t* num_allocated_object_bytes_ptr,
    void* ptr,
    size_t size)
{
    pas_large_free_heap_config config;
    pas_heap_lock_assert_held();
    if (!size)
        return;
    initialize_config(&config, memory_source);
    pas_will_deallocate(ptr, size, pas_large_utility_free_heap_kind, pas_object_allocation);
    if (pas_large_utility_free_heap_talks_to_large_sharing_pool) {
        if (verbose) {
            pas_log("large free heap freeing %p...%p.\n",
                    (void*)ptr,
                    (char*)ptr + size);
        }
        pas_large_sharing_pool_free(
            pas_range_create((uintptr_t)ptr, (uintptr_t)ptr + size),
            pas_physical_memory_is_locked_by_heap_lock,
            pas_may_mmap);
    }
    pas_fast_large_free_heap_deallocate(heap,
                                        (uintptr_t)ptr, (uintptr_t)ptr + size,
                                        pas_zero_mode_may_have_non_zero,
                                        &config);
    (*num_allocated_object_bytes_ptr) -= size;
}

pas_heap_summary pas_large_free_heap_helpers_compute_summary(
    pas_fast_large_free_heap* heap,
    size_t* num_allocated_object_bytes_ptr)
{
    pas_heap_summary result;

    pas_heap_lock_assert_held();

    result = pas_heap_summary_create_empty();
    
    pas_fast_large_free_heap_for_each_free(
        heap,
        pas_large_utility_free_heap_talks_to_large_sharing_pool
        ? pas_compute_summary_dead_object_callback
        : pas_compute_summary_dead_object_callback_without_physical_sharing,
        &result);

    result.allocated += *num_allocated_object_bytes_ptr;
    result.committed += *num_allocated_object_bytes_ptr;

    return result;
}

#endif /* LIBPAS_ENABLED */
