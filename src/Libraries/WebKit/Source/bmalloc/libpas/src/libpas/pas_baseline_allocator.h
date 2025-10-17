/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 16, 2024.
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
#ifndef PAS_BASELINE_ALLOCATOR_H
#define PAS_BASELINE_ALLOCATOR_H

#include "pas_internal_config.h"
#include "pas_local_allocator.h"
#include "pas_lock.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_baseline_allocator;
typedef struct pas_baseline_allocator pas_baseline_allocator;

#define PAS_BASELINE_LOCAL_ALLOCATOR_SIZE \
    PAS_LOCAL_ALLOCATOR_SIZE(PAS_MAX_OBJECTS_PER_PAGE)

struct pas_baseline_allocator {
    pas_lock lock; /* Can hold this before getting the heap lock. */
    union {
        pas_local_allocator allocator;
        char fake_field_to_force_size[PAS_BASELINE_LOCAL_ALLOCATOR_SIZE];
    } u;
};

#define PAS_BASELINE_ALLOCATOR_INITIALIZER ((pas_baseline_allocator){ \
        .lock = PAS_LOCK_INITIALIZER, \
        .u = { \
            .allocator = PAS_LOCAL_ALLOCATOR_NULL_INITIALIZER \
        } \
    })

PAS_API void pas_baseline_allocator_attach_directory(pas_baseline_allocator* allocator,
                                                     pas_segregated_size_directory* directory);

PAS_API void pas_baseline_allocator_detach_directory(pas_baseline_allocator* allocator);

PAS_END_EXTERN_C;

#endif /* PAS_BASELINE_ALLOCATOR_H */

