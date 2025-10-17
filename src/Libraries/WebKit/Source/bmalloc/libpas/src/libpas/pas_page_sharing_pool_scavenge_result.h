/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 16, 2024.
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
#ifndef PAS_PAGE_SHARING_POOL_SCAVENGE_RESULT_H
#define PAS_PAGE_SHARING_POOL_SCAVENGE_RESULT_H

#include "pas_page_sharing_pool_take_result.h"

PAS_BEGIN_EXTERN_C;

struct pas_page_sharing_pool_scavenge_result;
typedef struct pas_page_sharing_pool_scavenge_result pas_page_sharing_pool_scavenge_result;

struct pas_page_sharing_pool_scavenge_result {
    pas_page_sharing_pool_take_result take_result;
    size_t total_bytes;
};

static inline pas_page_sharing_pool_scavenge_result pas_page_sharing_pool_scavenge_result_create(
    pas_page_sharing_pool_take_result take_result,
    size_t total_bytes)
{
    pas_page_sharing_pool_scavenge_result result;
    result.take_result = take_result;
    result.total_bytes = total_bytes;
    return result;
}

PAS_END_EXTERN_C;

#endif /* PAS_PAGE_SHARING_POOL_SCAVENGE_RESULT_H */

