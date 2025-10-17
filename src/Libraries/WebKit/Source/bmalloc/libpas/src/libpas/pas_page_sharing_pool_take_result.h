/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 15, 2024.
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
#ifndef PAS_PAGE_SHARING_POOL_TAKE_RESULT_H
#define PAS_PAGE_SHARING_POOL_TAKE_RESULT_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

enum pas_page_sharing_pool_take_result {
    pas_page_sharing_pool_take_none_available,
    pas_page_sharing_pool_take_none_within_max_epoch,
    pas_page_sharing_pool_take_locks_unavailable,
    pas_page_sharing_pool_take_success,
};

typedef enum pas_page_sharing_pool_take_result pas_page_sharing_pool_take_result;

static inline const char*
pas_page_sharing_pool_take_result_get_string(pas_page_sharing_pool_take_result result)
{
    switch (result) {
    case pas_page_sharing_pool_take_none_available:
        return "none_available";
    case pas_page_sharing_pool_take_none_within_max_epoch:
        return "none_within_max_epoch";
    case pas_page_sharing_pool_take_locks_unavailable:
        return "locks_unavailable";
    case pas_page_sharing_pool_take_success:
        return "success";
    }
    PAS_ASSERT(!"Bad take result");
    return NULL;
}

PAS_END_EXTERN_C;

#endif /* PAS_PAGE_SHARING_POOL_TAKE_RESULT_H */

