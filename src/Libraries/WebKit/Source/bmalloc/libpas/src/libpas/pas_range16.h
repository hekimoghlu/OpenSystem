/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 7, 2022.
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
#ifndef PAS_RANGE16_H
#define PAS_RANGE16_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_range16;
typedef struct pas_range16 pas_range16;

struct pas_range16 {
    uint16_t begin;
    uint16_t end;
};

static inline pas_range16 pas_range16_create(uintptr_t begin, uintptr_t end)
{
    pas_range16 result;
    PAS_ASSERT(end >= begin);
    result.begin = (uint16_t)begin;
    result.end = (uint16_t)end;
    PAS_ASSERT(result.begin == begin);
    PAS_ASSERT(result.end == end);
    return result;
}

PAS_END_EXTERN_C;

#endif /* PAS_RANGE16_H */

