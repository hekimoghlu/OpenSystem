/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 5, 2022.
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
#ifndef PAS_ALLOCATION_KIND_H
#define PAS_ALLOCATION_KIND_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

enum pas_allocation_kind {
    /* We are using the heap to allocate a normal object or array. If it's a free heap, we will
       remember the size and later call free on this object with exactly that size. */
    pas_object_allocation,
    
    /* We are using the free heap as a source of memory or some other heap. The size allocated
       doesn't have to match the size freed. */
    pas_delegate_allocation,
};

typedef enum pas_allocation_kind pas_allocation_kind;

static inline const char* pas_allocation_kind_get_string(pas_allocation_kind kind)
{
    switch (kind) {
    case pas_object_allocation:
        return "object";
    case pas_delegate_allocation:
        return "delegate";
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

PAS_END_EXTERN_C;

#endif /* PAS_ALLOCATION_KIND_H */
