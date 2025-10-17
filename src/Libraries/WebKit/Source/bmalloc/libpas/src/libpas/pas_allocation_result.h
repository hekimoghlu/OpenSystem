/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 3, 2022.
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
#ifndef PAS_ALLOCATION_RESULT_H
#define PAS_ALLOCATION_RESULT_H

#include <errno.h>
#include "pas_internal_config.h"
#include "pas_utils.h"
#include "pas_zero_mode.h"

PAS_BEGIN_EXTERN_C;

struct pas_allocation_result;
typedef struct pas_allocation_result pas_allocation_result;

/* Normally, we just return NULL to mean that allocation failed. But in some cases it's better
   to separate out the allocation result status from the pointer to make compiler transformations
   on the allocation fast path work. */
struct pas_allocation_result {
    uintptr_t begin;
    bool did_succeed;
    pas_zero_mode zero_mode;
};

typedef pas_allocation_result (*pas_allocation_result_filter)(pas_allocation_result result);

static inline pas_allocation_result pas_allocation_result_create_failure(void)
{
    pas_allocation_result result;
    result.did_succeed = false;
    result.zero_mode = pas_zero_mode_may_have_non_zero;
    result.begin = 0;
    return result;
}

static inline pas_allocation_result
pas_allocation_result_create_success_with_zero_mode(uintptr_t begin,
                                                    pas_zero_mode zero_mode)
{
    pas_allocation_result result;
    result.did_succeed = true;
    result.zero_mode = zero_mode;
    result.begin = begin;
    return result;
}

static inline pas_allocation_result pas_allocation_result_create_success(uintptr_t begin)
{
    return pas_allocation_result_create_success_with_zero_mode(
        begin, pas_zero_mode_may_have_non_zero);
}

static PAS_ALWAYS_INLINE pas_allocation_result
pas_allocation_result_identity(pas_allocation_result result)
{
    return result;
}

PAS_API pas_allocation_result pas_allocation_result_zero_large_slow(pas_allocation_result result, size_t size);

static PAS_ALWAYS_INLINE pas_allocation_result
pas_allocation_result_zero(pas_allocation_result result,
                           size_t size)
{
    if (PAS_UNLIKELY(!result.did_succeed))
        return result;
    if (result.zero_mode == pas_zero_mode_is_all_zero)
        return result;

    if (size >= (1ULL << PAS_VA_BASED_ZERO_MEMORY_SHIFT))
        return pas_allocation_result_zero_large_slow(result, size);

    PAS_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    PAS_PROFILE(ZERO_ALLOCATION_RESULT, result.begin);
    PAS_ALLOW_UNSAFE_BUFFER_USAGE_END

    void* memory = (void*)result.begin;
    pas_zero_memory(memory, size);

    return pas_allocation_result_create_success_with_zero_mode(result.begin, pas_zero_mode_is_all_zero);
}

static PAS_ALWAYS_INLINE pas_allocation_result
pas_allocation_result_set_errno(pas_allocation_result result)
{
    if (!result.did_succeed)
        errno = ENOMEM;
    return result;
}

static PAS_ALWAYS_INLINE pas_allocation_result
pas_allocation_result_crash_on_error(pas_allocation_result result)
{
    if (PAS_UNLIKELY(!result.did_succeed))
        pas_panic_on_out_of_memory_error();
    return result;
}

PAS_END_EXTERN_C;

#endif /* PAS_ALLOCATION_RESULT_H */
