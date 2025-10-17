/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 7, 2025.
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
#ifndef PAS_ALIGNMENT_H
#define PAS_ALIGNMENT_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_alignment;
struct pas_stream;
typedef struct pas_alignment pas_alignment;
typedef struct pas_stream pas_stream;

struct pas_alignment {
    size_t alignment;
    uintptr_t alignment_begin;
};

static inline pas_alignment pas_alignment_create(size_t alignment, uintptr_t alignment_begin)
{
    PAS_ASSERT(pas_is_power_of_2(alignment));
    PAS_ASSERT(alignment_begin < alignment);
    
    pas_alignment result;
    result.alignment = alignment;
    result.alignment_begin = alignment_begin;
    return result;
}

static inline pas_alignment pas_alignment_create_traditional(size_t alignment)
{
    return pas_alignment_create(alignment, 0);
}

static inline pas_alignment pas_alignment_create_trivial(void)
{
    return pas_alignment_create(1, 0);
}

static inline void pas_alignment_validate(pas_alignment alignment)
{
    PAS_ASSERT(pas_is_power_of_2(alignment.alignment));
    PAS_ASSERT(alignment.alignment_begin < alignment.alignment);
}

static inline bool pas_alignment_is_ptr_aligned(pas_alignment alignment,
                                                uintptr_t ptr)
{
    return pas_is_aligned(ptr - alignment.alignment_begin, alignment.alignment);
}

static inline pas_alignment pas_alignment_round_up(pas_alignment alignment,
                                                   uintptr_t possibly_bigger_alignment)
{
    pas_alignment result;
    pas_alignment_validate(alignment);
    PAS_ASSERT(pas_is_power_of_2(possibly_bigger_alignment));
    
    /* This creates a new alignment that is either:
       
       - Exactly the same as the old one, or
       
       - Has a bigger alignment but the same offset. That's trivially correct, since that is
         the exact first-fit solution to the question of how to allocate with the input alignment
         constraint from an allocator that has the bigger alignment capability. */
    result = pas_alignment_create(
        PAS_MAX(alignment.alignment, possibly_bigger_alignment),
        alignment.alignment_begin);
    
    pas_alignment_validate(result);
    return result;
}

static inline bool pas_alignment_is_equal(pas_alignment left, pas_alignment right)
{
    return left.alignment == right.alignment
        && left.alignment_begin == right.alignment_begin;
}

PAS_API void pas_alignment_dump(pas_alignment alignment, pas_stream* stream);

PAS_END_EXTERN_C;

#endif /* PAS_ALIGNMENT_H */
