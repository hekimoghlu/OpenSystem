/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 23, 2025.
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
#ifndef PAS_RANGE_BEGIN_MIN_HEAP_H
#define PAS_RANGE_BEGIN_MIN_HEAP_H

#include "pas_min_heap.h"
#include "pas_range.h"

PAS_BEGIN_EXTERN_C;

static inline int pas_range_begin_min_heap_compare(pas_range* left_ptr, pas_range* right_ptr)
{
    pas_range left;
    pas_range right;

    left = *left_ptr;
    right = *right_ptr;

    if (left.begin < right.begin)
        return -1;
    if (left.begin == right.begin)
        return 0;
    return 1;
}

static inline size_t pas_range_begin_min_heap_get_index(pas_range* element_ptr)
{
    PAS_UNUSED_PARAM(element_ptr);
    PAS_ASSERT(!"Should not be reached");
    return 0;
}

static inline void pas_range_begin_min_heap_set_index(pas_range* element_ptr, size_t index)
{
    PAS_UNUSED_PARAM(element_ptr);
    PAS_UNUSED_PARAM(index);
}

PAS_CREATE_MIN_HEAP(pas_range_begin_min_heap,
                    pas_range,
                    10,
                    .compare = pas_range_begin_min_heap_compare,
                    .get_index = pas_range_begin_min_heap_get_index,
                    .set_index = pas_range_begin_min_heap_set_index);

PAS_END_EXTERN_C;

#endif /* PAS_RANGE_BEGIN_MIN_HEAP_H */

