/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 1, 2022.
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
#ifndef PAS_FAST_PATH_ALLOCATION_RESULT_H
#define PAS_FAST_PATH_ALLOCATION_RESULT_H

#include "pas_allocation_result.h"
#include "pas_fast_path_allocation_result_kind.h"

PAS_BEGIN_EXTERN_C;

struct pas_fast_path_allocation_result;
typedef struct pas_fast_path_allocation_result pas_fast_path_allocation_result;

struct pas_fast_path_allocation_result {
    pas_fast_path_allocation_result_kind kind;
    uintptr_t begin;
};

static inline pas_fast_path_allocation_result
pas_fast_path_allocation_result_create(pas_fast_path_allocation_result_kind kind)
{
    pas_fast_path_allocation_result result;
    PAS_ASSERT(kind == pas_fast_path_allocation_result_need_slow
               || kind == pas_fast_path_allocation_result_out_of_memory);
    result.kind = kind;
    result.begin = 0;
    return result;
}

static inline pas_fast_path_allocation_result
pas_fast_path_allocation_result_create_need_slow(void)
{
    return pas_fast_path_allocation_result_create(pas_fast_path_allocation_result_need_slow);
}

static inline pas_fast_path_allocation_result
pas_fast_path_allocation_result_create_out_of_memory(void)
{
    return pas_fast_path_allocation_result_create(pas_fast_path_allocation_result_out_of_memory);
}

static inline pas_fast_path_allocation_result
pas_fast_path_allocation_result_create_success(uintptr_t begin)
{
    pas_fast_path_allocation_result result;
    result.kind = pas_fast_path_allocation_result_success;
    result.begin = begin;
    return result;
}

static inline pas_fast_path_allocation_result
pas_fast_path_allocation_result_from_allocation_result(
    pas_allocation_result allocation_result,
    pas_fast_path_allocation_result_kind failure_kind)
{
    pas_fast_path_allocation_result result;
    PAS_ASSERT(failure_kind != pas_fast_path_allocation_result_success);
    result.kind = allocation_result.did_succeed
        ? pas_fast_path_allocation_result_success
        : failure_kind;
    result.begin = allocation_result.begin;
    return result;
}

static inline pas_allocation_result
pas_fast_path_allocation_result_to_allocation_result(pas_fast_path_allocation_result fast_result)
{
    if (fast_result.kind == pas_fast_path_allocation_result_success)
        return pas_allocation_result_create_success(fast_result.begin);
    return pas_allocation_result_create_failure();
}

PAS_END_EXTERN_C;

#endif /* PAS_FAST_PATH_ALLOCATION_RESULT_H */

