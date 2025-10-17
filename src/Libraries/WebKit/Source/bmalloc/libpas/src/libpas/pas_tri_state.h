/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 7, 2024.
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
#ifndef PAS_TRI_STATE_H
#define PAS_TRI_STATE_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

enum pas_tri_state {
    pas_tri_state_no,
    pas_tri_state_maybe,
    pas_tri_state_yes
};

typedef enum pas_tri_state pas_tri_state;

static inline const char* pas_tri_state_get_string(pas_tri_state tri_state)
{
    switch (tri_state) {
    case pas_tri_state_no:
        return "no";
    case pas_tri_state_maybe:
        return "maybe";
    case pas_tri_state_yes:
        return "yes";
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

static inline bool pas_tri_state_equals_boolean(pas_tri_state tri_state, bool boolean)
{
    switch (tri_state) {
    case pas_tri_state_no:
        return !boolean;
    case pas_tri_state_maybe:
        return true;
    case pas_tri_state_yes:
        return boolean;
    }
    PAS_ASSERT(!"Should not be reached");
    return false;
}

PAS_END_EXTERN_C;

#endif /* PAS_TRI_STATE_H */

