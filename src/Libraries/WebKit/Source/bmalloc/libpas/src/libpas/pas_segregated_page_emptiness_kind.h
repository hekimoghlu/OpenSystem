/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 9, 2023.
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
#ifndef PAS_SEGREGATED_PAGE_EMPTINESS_KIND_H
#define PAS_SEGREGATED_PAGE_EMPTINESS_KIND_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

enum pas_page_emptiness_kind {
    pas_page_partial_emptiness,
    pas_page_full_emptiness
};

typedef enum pas_page_emptiness_kind pas_page_emptiness_kind;

static inline pas_page_emptiness_kind
pas_page_emptiness_kind_get_inverted(pas_page_emptiness_kind kind)
{
    switch (kind) {
    case pas_page_partial_emptiness:
        return pas_page_full_emptiness;
    case pas_page_full_emptiness:
        return pas_page_partial_emptiness;
    }
    PAS_ASSERT(!"Should not be reached");
    return pas_page_partial_emptiness;
}

static inline const char*
pas_page_emptiness_kind_get_string(pas_page_emptiness_kind kind)
{
    switch (kind) {
    case pas_page_partial_emptiness:
        return "partial";
    case pas_page_full_emptiness:
        return "full";
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

PAS_END_EXTERN_C;

#endif /* PAS_SEGREGATED_PAGE_EMPTINESS_KIND_H */

