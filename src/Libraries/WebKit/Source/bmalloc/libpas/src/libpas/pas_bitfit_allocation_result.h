/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 10, 2023.
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
#ifndef PAS_BITFIT_ALLOCATION_RESULT_H
#define PAS_BITFIT_ALLOCATION_RESULT_H

#include "pas_range.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_bitfit_allocation_result;
typedef struct pas_bitfit_allocation_result pas_bitfit_allocation_result;

struct pas_bitfit_allocation_result {
    bool did_succeed;
    unsigned pages_to_commit_on_reloop;
    union {
        uintptr_t result;
        uintptr_t largest_available;
    } u;
};

static PAS_ALWAYS_INLINE pas_bitfit_allocation_result
pas_bitfit_allocation_result_create_success(uintptr_t result_ptr)
{
    pas_bitfit_allocation_result result;
    result.did_succeed = true;
    result.pages_to_commit_on_reloop = 0;
    result.u.result = result_ptr;
    return result;
}

static PAS_ALWAYS_INLINE pas_bitfit_allocation_result
pas_bitfit_allocation_result_create_failure(uintptr_t largest_available)
{
    pas_bitfit_allocation_result result;
    result.did_succeed = false;
    result.pages_to_commit_on_reloop = 0;
    result.u.largest_available = largest_available;
    return result;
}

static PAS_ALWAYS_INLINE pas_bitfit_allocation_result
pas_bitfit_allocation_result_create_empty(void)
{
    return pas_bitfit_allocation_result_create_failure(0);
}

static PAS_ALWAYS_INLINE pas_bitfit_allocation_result
pas_bitfit_allocation_result_create_need_to_lock_commit_lock(unsigned pages_to_commit_on_reloop)
{
    pas_bitfit_allocation_result result;
    result.did_succeed = false;
    result.pages_to_commit_on_reloop = pages_to_commit_on_reloop;
    result.u.result = 0;
    return result;
}

PAS_END_EXTERN_C;

#endif /* PAS_BITFIT_ALLOCATION_RESULT_H */

