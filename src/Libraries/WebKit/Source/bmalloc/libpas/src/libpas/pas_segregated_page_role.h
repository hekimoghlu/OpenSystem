/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 24, 2025.
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
#ifndef PAS_SEGREGATED_PAGE_ROLE_H
#define PAS_SEGREGATED_PAGE_ROLE_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

enum pas_segregated_page_role {
    pas_segregated_page_shared_role,
    pas_segregated_page_exclusive_role
};

typedef enum pas_segregated_page_role pas_segregated_page_role;

static inline const char* pas_segregated_page_role_get_string(pas_segregated_page_role role)
{
    switch (role) {
    case pas_segregated_page_shared_role:
        return "shared";
    case pas_segregated_page_exclusive_role:
        return "exclusive";
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

PAS_END_EXTERN_C;

#endif /* PAS_SEGREGATED_PAGE_ROLE_H */

