/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 21, 2024.
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
#ifndef PAS_ALLOCATION_MODE_H
#define PAS_ALLOCATION_MODE_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

enum pas_allocation_mode {
    /* We are allocating an object from ordinary memory and don't plan on
       compacting its address. */
    pas_non_compact_allocation_mode,

    /* We are allocating an object from ordinary memory and expect to
       be able to compact its address, but don't expect all addresses in
       that memory to be trivially compactible. */
    pas_maybe_compact_allocation_mode,

    /* We are allocating an object from memory where all addresses within
       that memory are trivially compactible, like in the immortal heap. */
    pas_always_compact_allocation_mode,
};

typedef enum pas_allocation_mode pas_allocation_mode;
typedef enum pas_allocation_mode __pas_allocation_mode;

static inline const char* pas_allocation_mode_get_string(pas_allocation_mode allocation_mode)
{
    switch (allocation_mode) {
    case pas_non_compact_allocation_mode:
        return "non-compact";
    case pas_maybe_compact_allocation_mode:
        return "maybe compact";
    case pas_always_compact_allocation_mode:
        return "always compact";
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

PAS_END_EXTERN_C;

#endif /* PAS_ALLOCATION_MODE_H */
