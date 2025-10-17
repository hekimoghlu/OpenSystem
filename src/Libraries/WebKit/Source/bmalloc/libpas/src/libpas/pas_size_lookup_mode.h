/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 19, 2022.
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
#ifndef PAS_SIZE_LOOKUP_MODE_H
#define PAS_SIZE_LOOKUP_MODE_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

enum pas_size_lookup_mode {
    pas_avoid_size_lookup,
    pas_force_size_lookup
};

typedef enum pas_size_lookup_mode pas_size_lookup_mode;

static inline const char* pas_size_lookup_mode_get_string(pas_size_lookup_mode mode)
{
    switch (mode) {
    case pas_avoid_size_lookup:
        return "avoid_size_lookup";
    case pas_force_size_lookup:
        return "force_size_lookup";
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

PAS_END_EXTERN_C;

#endif /* PAS_SIZE_LOOKUP_MODE_H */

