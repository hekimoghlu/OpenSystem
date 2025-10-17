/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 7, 2025.
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
#ifndef PAS_RANGE_H
#define PAS_RANGE_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_range;
typedef struct pas_range pas_range;

struct pas_range {
    uintptr_t begin;
    uintptr_t end;
};

static inline pas_range pas_range_create(uintptr_t begin, uintptr_t end)
{
    pas_range result;
    PAS_ASSERT(end >= begin);
    result.begin = begin;
    result.end = end;
    return result;
}

static inline pas_range pas_range_create_empty(void)
{
    return pas_range_create(0, 0);
}

static inline pas_range pas_range_create_forgiving(uintptr_t begin, uintptr_t end)
{
    if (end < begin)
        return pas_range_create_empty();
    return pas_range_create(begin, end);
}

static inline bool pas_range_is_empty(pas_range range)
{
    return range.begin == range.end;
}

static inline size_t pas_range_size(pas_range range)
{
    PAS_ASSERT(range.end >= range.begin);
    return range.end - range.begin;
}

static inline bool pas_range_contains(pas_range left, uintptr_t right)
{
    return right >= left.begin && right < left.end;
}

static inline bool pas_range_subsumes(pas_range left, pas_range right)
{
    if (pas_range_is_empty(right))
        return true;
    return right.begin >= left.begin && right.end <= left.end;
}

static inline bool pas_range_overlaps(pas_range left, pas_range right)
{
    return pas_ranges_overlap(left.begin, left.end,
                              right.begin, right.end);
}

static inline pas_range pas_range_create_intersection(pas_range left, pas_range right)
{
    if (!pas_range_overlaps(left, right))
        return pas_range_create_empty();
    
    return pas_range_create(PAS_MAX(left.begin, right.begin),
                            PAS_MIN(left.end, right.end));
}

static inline int pas_range_compare(pas_range left, pas_range right)
{
    if (pas_range_overlaps(left, right))
        return 0;
    
    if (left.begin < right.begin)
        return -1;
    return 1;
}

PAS_END_EXTERN_C;

#endif /* PAS_RANGE_H */

