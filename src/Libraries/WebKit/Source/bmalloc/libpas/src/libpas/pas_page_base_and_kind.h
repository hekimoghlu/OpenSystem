/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 1, 2024.
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
#ifndef PAS_PAGE_BASE_AND_KIND_H
#define PAS_PAGE_BASE_AND_KIND_H

#include "pas_page_base.h"

PAS_BEGIN_EXTERN_C;

struct pas_page_base;
struct pas_page_base_and_kind;
typedef struct pas_page_base pas_page_base;
typedef struct pas_page_base_and_kind pas_page_base_and_kind;

struct pas_page_base_and_kind {
    pas_page_base* page_base;
    pas_page_kind page_kind;
};

static inline pas_page_base_and_kind pas_page_base_and_kind_create(pas_page_base* page_base,
                                                                   pas_page_kind page_kind)
{
    pas_page_base_and_kind result;
    PAS_TESTING_ASSERT(page_base);
    PAS_TESTING_ASSERT(pas_page_base_get_kind(page_base) == page_kind);
    result.page_base = page_base;
    result.page_kind = page_kind;
    return result;
}

static inline pas_page_base_and_kind pas_page_base_and_kind_create_empty(void)
{
    pas_page_base_and_kind result;
    pas_zero_memory(&result, sizeof(result));
    return result;
}

PAS_END_EXTERN_C;

#endif /* PAS_PAGE_BASE_AND_KIND_H */

