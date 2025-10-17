/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 14, 2022.
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
#ifndef PAS_LARGE_FREE_INLINES_H
#define PAS_LARGE_FREE_INLINES_H

#include "pas_coalign.h"
#include "pas_internal_config.h"
#include "pas_large_free.h"
#include "pas_log.h"

PAS_BEGIN_EXTERN_C;

static
#ifndef WK_WORKAROUND_RDAR_87613908_ASAN_STACK_USE_AFTER_SCOPE
PAS_ALWAYS_INLINE
#endif
pas_large_free
pas_large_free_create_merged(pas_large_free left,
                             pas_large_free right,
                             pas_large_free_heap_config* config)
{
    pas_large_free result;

    PAS_TESTING_ASSERT(left.offset_in_type < config->type_size);
    PAS_TESTING_ASSERT(right.offset_in_type < config->type_size);

    for (;;) {
        if (left.begin == right.end) {
            PAS_SWAP(left, right);
            continue;
        }
        
        result = pas_large_free_create_empty();
        
        if (left.end != right.begin)
            return result;
        
        if (pas_large_free_offset_in_type_at_end(left, config)
            != right.offset_in_type)
            return result;
        
        result.begin = left.begin;
        result.end = right.end;
        result.offset_in_type = left.offset_in_type;
        result.zero_mode = pas_zero_mode_merge(left.zero_mode, right.zero_mode);
        return result;
    }
}

static PAS_ALWAYS_INLINE pas_large_split
pas_large_free_split(pas_large_free free,
                     size_t left_size,
                     pas_large_free_heap_config* config)
{
    pas_large_split result;
    
    PAS_TESTING_ASSERT(left_size <= pas_large_free_size(free));
    PAS_TESTING_ASSERT(free.offset_in_type < config->type_size);
    
    PAS_TESTING_ASSERT(free.begin);
    PAS_TESTING_ASSERT(free.end > free.begin);
    
    result.left = pas_large_free_create_empty();
    result.right = pas_large_free_create_empty();
    
    result.left.begin = free.begin;
    result.left.end = free.begin + left_size;
    result.left.offset_in_type = free.offset_in_type;
    result.left.zero_mode = free.zero_mode;
    
    result.right.begin = free.begin + left_size;
    result.right.end = free.end;
    result.right.offset_in_type =
        (free.offset_in_type + left_size) % config->type_size;
    result.right.zero_mode = free.zero_mode;

    PAS_TESTING_ASSERT(result.left.offset_in_type < config->type_size);
    PAS_TESTING_ASSERT(result.right.offset_in_type < config->type_size);
    
    PAS_TESTING_ASSERT(result.left.begin >= free.begin);
    PAS_TESTING_ASSERT(result.left.end >= result.left.begin);
    PAS_TESTING_ASSERT(result.right.begin >= result.left.end);
    PAS_TESTING_ASSERT(result.right.end >= result.right.begin);
    PAS_TESTING_ASSERT(result.right.end <= free.end);

    return result;
}

static PAS_ALWAYS_INLINE pas_coalign_result
pas_large_free_allocation_candidate(pas_large_free free,
                                    pas_alignment alignment,
                                    pas_large_free_heap_config* config)
{
    uintptr_t search_start;
    pas_coalign_result result;

    PAS_TESTING_ASSERT(free.begin);
    PAS_TESTING_ASSERT(free.end >= free.begin);
    
    if (free.begin == free.end)
        return pas_coalign_empty_result();

    search_start = free.begin;
    
    if (PAS_LIKELY(!free.offset_in_type)) {
        if (PAS_LIKELY(pas_alignment_is_ptr_aligned(alignment, search_start)))
            return pas_coalign_result_create(free.begin);
    } else {
        search_start += config->type_size - free.offset_in_type;
    
        if (search_start >= free.end)
            return pas_coalign_empty_result();
    }
    
    result = pas_coalign(search_start, config->type_size,
                         alignment.alignment_begin, alignment.alignment);
    if (!result.has_result)
        return pas_coalign_empty_result();

    PAS_TESTING_ASSERT(result.result >= free.begin);
    
    if (result.result >= free.end)
        return pas_coalign_empty_result();
    
    return result;
}

static PAS_ALWAYS_INLINE size_t
pas_large_free_usable_space(pas_large_free free,
                            pas_large_free_heap_config* config)
{
    pas_coalign_result candidate;
    
    PAS_TESTING_ASSERT(config->min_alignment >= 1);
    PAS_TESTING_ASSERT(pas_is_power_of_2(config->min_alignment));
    PAS_TESTING_ASSERT(config->type_size >= 1);
    
    candidate = pas_large_free_allocation_candidate(
        free,
        pas_alignment_create_traditional(config->min_alignment),
        config);
    
    if (!candidate.has_result)
        return 0;

    if (PAS_LIKELY(candidate.result == free.begin))
        return free.end - free.begin;
    
    return (free.end - candidate.result)
        / config->min_alignment * config->min_alignment
        / config->type_size * config->type_size;
}

static inline pas_large_allocation_result
pas_large_allocation_result_create_empty(void)
{
    pas_large_allocation_result result;
    result.left_free = pas_large_free_create_empty();
    result.allocation = pas_large_free_create_empty();
    result.right_free = pas_large_free_create_empty();
    return result;
}

/* This is somewhat expensive, so it's a good idea to do a simple eligibility
   filter before calling it, like checking if the requested size is smaller
   than the free's size. */
static
#ifndef WK_WORKAROUND_RDAR_87613908_ASAN_STACK_USE_AFTER_SCOPE
PAS_ALWAYS_INLINE
#endif
pas_large_allocation_result
pas_large_free_allocate(pas_large_free free,
                        size_t size,
                        pas_alignment alignment,
                        pas_large_free_heap_config* config)
{
    static const bool verbose = false;
    
    pas_large_allocation_result result;
    pas_large_split split;

    if (verbose)
        pas_log("Doing pas_large_free_allocate.\n");
    
    PAS_TESTING_ASSERT(free.begin);
    PAS_TESTING_ASSERT(free.end > free.begin);
    
    pas_alignment_validate(alignment);
    
    result = pas_large_allocation_result_create_empty();

    if (!free.offset_in_type && pas_alignment_is_ptr_aligned(alignment, free.begin)
        && pas_is_aligned(size, config->min_alignment)) {
        split = pas_large_free_split(free, size, config);
        result.left_free = free;
        result.left_free.end = free.begin;
        result.allocation = split.left;
        result.right_free = split.right;
    } else {
        pas_coalign_result candidate;
        uintptr_t usage;

        candidate = pas_large_free_allocation_candidate(free, alignment, config);
        if (!candidate.has_result)
            return pas_large_allocation_result_create_empty();
    
        if (free.end - candidate.result < size)
            return pas_large_allocation_result_create_empty();

        split = pas_large_free_split(free,
                                     candidate.result - free.begin,
                                     config);
        result.left_free = split.left;
    
        split = pas_large_free_split(split.right,
                                     candidate.result + size - split.right.begin,
                                     config);
        result.allocation = split.left;
        result.right_free = split.right;
    
        usage =
            pas_large_free_usable_space(result.left_free, config) +
            size +
            pas_large_free_usable_space(result.right_free, config);
    
        if ((double)pas_large_free_size(free) / (double)usage
            > PAS_MAX_LARGE_ALIGNMENT_WASTEAGE) {
            if (verbose)
                pas_log("Wasteage says fail.\n");
            return pas_large_allocation_result_create_empty();
        }
    }
    
    PAS_TESTING_ASSERT(pas_large_free_size(result.allocation) == size);
    
    PAS_TESTING_ASSERT(!result.allocation.offset_in_type);
    
    PAS_TESTING_ASSERT(result.left_free.begin >= free.begin);
    PAS_TESTING_ASSERT(result.left_free.end >= result.left_free.begin);
    PAS_TESTING_ASSERT(result.allocation.begin >= result.left_free.end);
    PAS_TESTING_ASSERT(result.allocation.end >= result.allocation.begin);
    PAS_TESTING_ASSERT(result.right_free.begin >= result.allocation.end);
    PAS_TESTING_ASSERT(result.right_free.end >= result.right_free.begin);
    PAS_TESTING_ASSERT(result.right_free.end <= free.end);
    
    return result;
}

static inline pas_large_free
pas_large_free_create_merged_for_sure(pas_large_free left,
                                      pas_large_free right,
                                      pas_large_free_heap_config* config)
{
    pas_large_free result;
    
    result = pas_large_free_create_merged(left, right, config);
    
    PAS_ASSERT(!pas_large_free_is_empty(result));
    
    return result;
}

static inline bool pas_large_free_can_merge(pas_large_free left,
                                            pas_large_free right,
                                            pas_large_free_heap_config* config)
{
    if (left.begin != right.end
        && left.end != right.begin)
        return false;
    
    return !pas_large_free_is_empty(
        pas_large_free_create_merged(left, right, config));
}

static inline pas_allocation_result
pas_large_allocation_result_as_allocation_result(
    pas_large_allocation_result result)
{
    if (pas_large_free_is_empty(result.allocation))
        return pas_allocation_result_create_failure();

    return pas_allocation_result_create_success_with_zero_mode(
        result.allocation.begin,
        result.allocation.zero_mode);
}

PAS_END_EXTERN_C;

#endif /* PAS_LARGE_FREE_INLINES_H */

