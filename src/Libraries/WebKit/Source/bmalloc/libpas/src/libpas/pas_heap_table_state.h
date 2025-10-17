/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 20, 2024.
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
#ifndef PAS_HEAP_TABLE_STATE_H
#define PAS_HEAP_TABLE_STATE_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

enum pas_heap_table_state {
    pas_heap_table_state_uninitialized,
    pas_heap_table_state_failed,
    pas_heap_table_state_has_index
};

typedef enum pas_heap_table_state pas_heap_table_state;

static inline const char* pas_heap_table_state_get_string(pas_heap_table_state state)
{
    switch (state) {
    case pas_heap_table_state_uninitialized:
        return "uninitialized";
    case pas_heap_table_state_failed:
        return "failed";
    case pas_heap_table_state_has_index:
        return "has_index";
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

PAS_END_EXTERN_C;

#endif /* PAS_HEAP_TABLE_STATE_H */

