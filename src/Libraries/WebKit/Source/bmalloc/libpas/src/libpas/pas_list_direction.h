/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 6, 2023.
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
#ifndef PAS_LIST_DIRECTION_H
#define PAS_LIST_DIRECTION_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

enum pas_list_direction {
    pas_list_direction_prev,
    pas_list_direction_next
};

typedef enum pas_list_direction pas_list_direction;

static inline const char* pas_list_direction_get_string(pas_list_direction direction)
{
    switch (direction) {
    case pas_list_direction_prev:
        return "prev";
    case pas_list_direction_next:
        return "next";
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

static inline pas_list_direction pas_list_direction_invert(pas_list_direction direction)
{
    switch (direction) {
    case pas_list_direction_prev:
        return pas_list_direction_next;
    case pas_list_direction_next:
        return pas_list_direction_prev;
    }
    PAS_ASSERT(!"Should not be reached");
    return pas_list_direction_prev;
}

PAS_END_EXTERN_C;

#endif /* PAS_LIST_DIRECTION_H */

