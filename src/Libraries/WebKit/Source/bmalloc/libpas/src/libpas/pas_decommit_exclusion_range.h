/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 9, 2023.
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
#ifndef PAS_DECOMMIT_EXCLUSION_RANGE_H
#define PAS_DECOMMIT_EXCLUSION_RANGE_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_decommit_exclusion_range;
typedef struct pas_decommit_exclusion_range pas_decommit_exclusion_range;

struct pas_decommit_exclusion_range {
    uintptr_t start_of_possible_decommit;
    uintptr_t end_of_possible_decommit;
};

static inline bool pas_decommit_exclusion_range_is_empty(pas_decommit_exclusion_range range)
{
    return range.start_of_possible_decommit == range.end_of_possible_decommit;
}

static inline bool pas_decommit_exclusion_range_is_contiguous(pas_decommit_exclusion_range range)
{
    return range.start_of_possible_decommit < range.end_of_possible_decommit;
}

/* This means that the set of things that could be decommitted is anything before end_of_possible_decommit
   (exclusive) and after start_of_possible_decommit (inclusive). */
static inline bool pas_decommit_exclusion_range_is_inverted(pas_decommit_exclusion_range range)
{
    return range.start_of_possible_decommit > range.end_of_possible_decommit;
}

static inline pas_decommit_exclusion_range
pas_decommit_exclusion_range_create_inverted(pas_decommit_exclusion_range range)
{
    pas_decommit_exclusion_range result;
    result.start_of_possible_decommit = range.end_of_possible_decommit;
    result.end_of_possible_decommit = range.start_of_possible_decommit;
    return result;
}

PAS_END_EXTERN_C;

#endif /* PAS_DECOMMIT_EXCLUSION_RANGE_H */

