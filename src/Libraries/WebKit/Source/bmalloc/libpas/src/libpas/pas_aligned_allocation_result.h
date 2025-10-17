/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 17, 2022.
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
#ifndef PAS_ALIGNED_ALLOCATION_RESULT_H
#define PAS_ALIGNED_ALLOCATION_RESULT_H

#include "pas_allocation_result.h"
#include "pas_utils.h"
#include "pas_zero_mode.h"

PAS_BEGIN_EXTERN_C;

struct pas_aligned_allocation_result;
typedef struct pas_aligned_allocation_result pas_aligned_allocation_result;

struct pas_aligned_allocation_result {
    void* result;
    size_t result_size;
    void* left_padding;
    size_t left_padding_size;
    void* right_padding;
    size_t right_padding_size;
    pas_zero_mode zero_mode;
};

static inline pas_aligned_allocation_result pas_aligned_allocation_result_create_empty(void)
{
    pas_aligned_allocation_result result;
    result.result = NULL;
    result.result_size = 0;
    result.left_padding = NULL;
    result.left_padding_size = 0;
    result.right_padding = NULL;
    result.right_padding_size = 0;
    result.zero_mode = pas_zero_mode_may_have_non_zero;
    return result;
}

static inline pas_allocation_result
pas_aligned_allocation_result_as_allocation_result(pas_aligned_allocation_result result)
{
    if (!result.result)
        return pas_allocation_result_create_failure();
    
    return pas_allocation_result_create_success_with_zero_mode((uintptr_t)result.result,
                                                               result.zero_mode);
}


PAS_END_EXTERN_C;

#endif /* PAS_ALIGNED_ALLOCATION_RESULT_H */

