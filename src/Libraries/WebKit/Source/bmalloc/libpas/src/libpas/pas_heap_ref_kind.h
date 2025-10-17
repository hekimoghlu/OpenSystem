/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 17, 2024.
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
#ifndef PAS_HEAP_REF_KIND_H
#define PAS_HEAP_REF_KIND_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

enum pas_heap_ref_kind {
    /* This means we have a pas_heap_ref. */
    pas_normal_heap_ref_kind,

    /* This means we have a pas_primitive_heap_ref. */
    pas_primitive_heap_ref_kind,

    /* This means that we created a fake pas_heap_ref just to carry the heap and type around. */
    pas_fake_heap_ref_kind
};

typedef enum pas_heap_ref_kind pas_heap_ref_kind;

static inline const char* pas_heap_ref_kind_get_string(pas_heap_ref_kind kind)
{
    switch (kind) {
    case pas_normal_heap_ref_kind:
        return "normal";
    case pas_primitive_heap_ref_kind:
        return "primitive";
    case pas_fake_heap_ref_kind:
        return "fake";
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

PAS_END_EXTERN_C;

#endif /* PAS_HEAP_REF_KIND_H */

