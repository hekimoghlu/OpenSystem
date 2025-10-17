/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 21, 2024.
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
#ifndef PAS_VIRTUAL_RANGE_H
#define PAS_VIRTUAL_RANGE_H

#include "pas_lock.h"
#include "pas_mmap_capability.h"
#include "pas_range.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_virtual_range;
typedef struct pas_virtual_range pas_virtual_range;

PAS_API extern pas_lock pas_virtual_range_common_lock;

struct pas_virtual_range {
    uintptr_t begin;
    uintptr_t end;
    pas_lock* lock_ptr;
    pas_mmap_capability mmap_capability;
};

static inline pas_virtual_range pas_virtual_range_create(uintptr_t begin,
                                                         uintptr_t end,
                                                         pas_lock* lock_ptr,
                                                         pas_mmap_capability mmap_capability)
{
    pas_virtual_range result;
    result.begin = begin;
    result.end = end;
    result.lock_ptr = lock_ptr;
    result.mmap_capability = mmap_capability;
    return result;
}

static inline pas_virtual_range pas_virtual_range_create_empty(void)
{
    return pas_virtual_range_create(0, 0, NULL, pas_may_not_mmap);
}

static inline pas_range pas_virtual_range_get_range(pas_virtual_range range)
{
    return pas_range_create(range.begin, range.end);
}

static inline bool pas_virtual_range_is_empty(pas_virtual_range range)
{
    return pas_range_is_empty(pas_virtual_range_get_range(range));
}

static inline size_t pas_virtual_range_size(pas_virtual_range range)
{
    return pas_range_size(pas_virtual_range_get_range(range));
}

static inline bool pas_virtual_range_overlaps(pas_virtual_range left,
                                              pas_virtual_range right)
{
    return pas_range_overlaps(pas_virtual_range_get_range(left),
                              pas_virtual_range_get_range(right));
}

PAS_END_EXTERN_C;

#endif /* PAS_VIRTUAL_RANGE_H */

