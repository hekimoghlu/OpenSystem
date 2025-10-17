/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 1, 2025.
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
#ifndef PAS_ENUMERATOR_REGION_H
#define PAS_ENUMERATOR_REGION_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_enumerator_region;
typedef struct pas_enumerator_region pas_enumerator_region;

struct pas_enumerator_region {
    pas_enumerator_region* previous;
    size_t size;
    size_t offset;
    PAS_ALIGNED(PAS_INTERNAL_MIN_ALIGN) uint64_t payload[1];
};

PAS_API void* pas_enumerator_region_allocate(pas_enumerator_region** region_ptr,
                                             size_t size);

PAS_API void pas_enumerator_region_destroy(pas_enumerator_region* region);

PAS_END_EXTERN_C;

#endif /* PAS_ENUMERATOR_REGION_H */

