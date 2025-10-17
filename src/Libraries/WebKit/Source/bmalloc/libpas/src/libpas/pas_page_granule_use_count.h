/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 16, 2025.
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
#ifndef PAS_PAGE_GRANULE_USE_COUNT_H
#define PAS_PAGE_GRANULE_USE_COUNT_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

typedef uint8_t pas_page_granule_use_count;
#define PAS_PAGE_GRANULE_DECOMMITTED 255u /* Has to be the max value. */

static PAS_ALWAYS_INLINE void pas_page_granule_get_indices(
    uintptr_t begin,
    uintptr_t end,
    uintptr_t page_size,
    uintptr_t granule_size,
    uintptr_t* index_of_first_granule,
    uintptr_t* index_of_last_granule)
{
    *index_of_first_granule = begin / granule_size;
    *index_of_last_granule = (end - 1) / granule_size;

    PAS_ASSERT(*index_of_last_granule < page_size / granule_size);
}

static PAS_ALWAYS_INLINE void pas_page_granule_for_each_use_in_range(
    pas_page_granule_use_count* use_counts,
    uintptr_t begin,
    uintptr_t end,
    uintptr_t page_size,
    uintptr_t granule_size,
    void (*action)(pas_page_granule_use_count* use_count, void* arg),
    void* arg)
{
    uintptr_t index_of_first_granule;
    uintptr_t index_of_last_granule;
    uintptr_t granule_index;

    if (begin == end)
        return;

    pas_page_granule_get_indices(
        begin, end, page_size, granule_size, &index_of_first_granule, &index_of_last_granule);

    for (granule_index = index_of_first_granule;
         granule_index <= index_of_last_granule;
         ++granule_index)
        action(use_counts + granule_index, arg);
}

static PAS_ALWAYS_INLINE void pas_page_granule_use_count_increment(
    pas_page_granule_use_count* use_count_ptr,
    void* arg)
{
    pas_page_granule_use_count use_count;

    PAS_UNUSED_PARAM(arg);

    use_count = *use_count_ptr;

    PAS_ASSERT(use_count != PAS_PAGE_GRANULE_DECOMMITTED);

    use_count++;

    PAS_ASSERT(use_count != PAS_PAGE_GRANULE_DECOMMITTED);

    *use_count_ptr = use_count;
}

static PAS_ALWAYS_INLINE void pas_page_granule_increment_uses_for_range(
    pas_page_granule_use_count* use_counts,
    uintptr_t begin,
    uintptr_t end,
    uintptr_t page_size,
    uintptr_t granule_size)
{
    pas_page_granule_for_each_use_in_range(
        use_counts, begin, end, page_size, granule_size,
        pas_page_granule_use_count_increment, NULL);
}

PAS_END_EXTERN_C;

#endif /* PAS_PAGE_GRANULE_USE_COUNT_H */

