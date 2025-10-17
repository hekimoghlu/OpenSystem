/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 23, 2024.
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
#ifndef PAS_FAST_PATH_ALLOCATION_RESULT_KIND_H
#define PAS_FAST_PATH_ALLOCATION_RESULT_KIND_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

enum pas_fast_path_allocation_result_kind {
    pas_fast_path_allocation_result_success,
    pas_fast_path_allocation_result_need_slow,
    pas_fast_path_allocation_result_out_of_memory
};

typedef enum pas_fast_path_allocation_result_kind pas_fast_path_allocation_result_kind;

static inline const char*
pas_fast_path_allocation_result_kind_get_string(pas_fast_path_allocation_result_kind kind)
{
    switch (kind) {
    case pas_fast_path_allocation_result_success:
        return "success";
    case pas_fast_path_allocation_result_need_slow:
        return "need_slow";
    case pas_fast_path_allocation_result_out_of_memory:
        return "out_of_memory";
    }
}

PAS_END_EXTERN_C;

#endif /* PAS_FAST_PATH_ALLOCATION_RESULT_KIND_H */

