/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 9, 2023.
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
#ifndef PAS_LOCAL_ALLOCATOR_RESULT_H
#define PAS_LOCAL_ALLOCATOR_RESULT_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_local_allocator_result;
typedef struct pas_local_allocator_result pas_local_allocator_result;

/* This exists to allow jump-threading of success. */
struct pas_local_allocator_result {
    bool did_succeed;
    void* allocator; /* Could really be a local_allocator or a local_view_cache. */
};

static inline pas_local_allocator_result pas_local_allocator_result_create_failure(void)
{
    pas_local_allocator_result result;
    result.did_succeed = false;
    result.allocator = NULL;
    return result;
}

static inline pas_local_allocator_result pas_local_allocator_result_create_success(void* allocator)
{
    pas_local_allocator_result result;
    result.did_succeed = true;
    result.allocator = allocator;
    return result;
}

PAS_END_EXTERN_C;

#endif /* PAS_LOCAL_ALLOCATOR_RESULT_H */
