/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 15, 2024.
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
#ifndef PAS_SEGREGATED_SHARED_VIEW_H
#define PAS_SEGREGATED_SHARED_VIEW_H

#include "pas_segregated_page_config.h"
#include "pas_segregated_view.h"
#include "pas_shared_handle_or_page_boundary.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_extended_segregated_shared_view;
struct pas_segregated_shared_view;
typedef struct pas_extended_segregated_shared_view pas_extended_segregated_shared_view;
typedef struct pas_segregated_shared_view pas_segregated_shared_view;

PAS_API extern size_t pas_segregated_shared_view_count;

struct pas_segregated_shared_view {
    pas_shared_handle_or_page_boundary shared_handle_or_page_boundary;
    pas_lock commit_lock;
    pas_lock ownership_lock;
    unsigned index : 31;

    /* This has roughly the same meaning as in pas_segregated_exclusive_view. */
    bool is_owned : 1;
    
    unsigned bump_offset;
    unsigned is_in_use_for_allocation_count;
};

static inline pas_segregated_view pas_segregated_shared_view_as_view(pas_segregated_shared_view* view)
{
    return pas_segregated_view_create(view, pas_segregated_shared_view_kind);
}

static inline pas_segregated_view pas_segregated_shared_view_as_view_non_null(pas_segregated_shared_view* view)
{
    return pas_segregated_view_create_non_null(view, pas_segregated_shared_view_kind);
}

PAS_API pas_segregated_shared_view* pas_segregated_shared_view_create(size_t index);

typedef struct {
    unsigned old_bump; /* The start of the allocation. It's _not_ the old value of bump_offset,
                          since old_bump includes the alignment bump. */
    unsigned new_bump; /* What to set bump_offset to. */
    unsigned end_bump; /* The end of the allocation (end_bump <= new_bump). */
    unsigned num_objects; /* Number of objects in the range. */
} pas_shared_view_computed_bump_result;

static inline pas_shared_view_computed_bump_result
pas_shared_view_computed_bump_result_create_empty(void)
{
    pas_shared_view_computed_bump_result result;
    result.old_bump = 0;
    result.new_bump = 0;
    result.end_bump = 0;
    result.num_objects = 0;
    return result;
}

static PAS_ALWAYS_INLINE pas_shared_view_computed_bump_result
pas_segregated_shared_view_compute_initial_new_bump(
    pas_segregated_shared_view* view,
    unsigned size,
    unsigned alignment,
    pas_segregated_page_config page_config)
{
    pas_shared_view_computed_bump_result result;

    PAS_UNUSED_PARAM(page_config);

    /* NOTE: This math cannot overflow because size and alignment are bounded to page size, and
       page size is much smaller than 1 << 32. It's not up to this function to assert that bound
       since we expect this to be a hot path. Moreover it's the kind of hot path where it's hard
       to reduce the amount of work done anywhere else so we might as well not assert too much. */

    result.old_bump = view->bump_offset;
    result.old_bump = (unsigned)pas_round_up_to_power_of_2(result.old_bump, alignment);
    result.new_bump = result.old_bump;
    result.new_bump += size;
    result.end_bump = result.new_bump;
    result.num_objects = 1;

    return result;
}

static PAS_ALWAYS_INLINE bool
pas_segregated_shared_view_can_bump(
    pas_segregated_shared_view* view,
    unsigned size,
    unsigned alignment,
    pas_segregated_page_config page_config)
{
    pas_shared_view_computed_bump_result result;
    unsigned bump_limit;

    bump_limit = (unsigned)
        pas_segregated_page_config_payload_end_offset_for_role(page_config, pas_segregated_page_shared_role);
    result = pas_segregated_shared_view_compute_initial_new_bump(view, size, alignment, page_config);

    return result.new_bump <= bump_limit;
}

static PAS_ALWAYS_INLINE pas_shared_view_computed_bump_result
pas_segregated_shared_view_compute_new_bump(
    pas_segregated_shared_view* view,
    unsigned size,
    unsigned alignment,
    pas_segregated_page_config page_config)
{
    pas_shared_view_computed_bump_result result;
    unsigned total_shift;
    unsigned bump_limit;

    total_shift = page_config.base.min_align_shift + page_config.sharing_shift;
    bump_limit = (unsigned)
        pas_segregated_page_config_payload_end_offset_for_role(page_config, pas_segregated_page_shared_role);
    result = pas_segregated_shared_view_compute_initial_new_bump(view, size, alignment, page_config);

    if (result.new_bump > bump_limit)
        return pas_shared_view_computed_bump_result_create_empty();
    
    while ((result.new_bump >> total_shift) == (result.old_bump >> total_shift)) {
        result.new_bump += size;

        if (result.new_bump > bump_limit) {
            result.new_bump = bump_limit;
            break;
        }

        result.end_bump = result.new_bump;
        result.num_objects++;
    }

    return result;
}

static PAS_ALWAYS_INLINE pas_shared_view_computed_bump_result
pas_segregated_shared_view_bump(
    pas_segregated_shared_view* view,
    unsigned size,
    unsigned alignment,
    pas_segregated_page_config page_config)
{
    pas_shared_view_computed_bump_result result;
    
    result = pas_segregated_shared_view_compute_new_bump(
        view, size, alignment, page_config);
    
    if (result.num_objects)
        view->bump_offset = result.new_bump;
    
    return result;
}

PAS_API pas_heap_summary
pas_segregated_shared_view_compute_summary(pas_segregated_shared_view* view,
                                           const pas_segregated_page_config* page_config);

PAS_API bool pas_segregated_shared_view_is_empty(pas_segregated_shared_view* view);

PAS_END_EXTERN_C;

#endif /* PAS_SEGREGATED_SHARED_VIEW_H */

