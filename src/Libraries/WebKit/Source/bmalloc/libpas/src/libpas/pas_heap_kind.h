/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 25, 2023.
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
#ifndef PAS_HEAP_KIND_H
#define PAS_HEAP_KIND_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

/* This is used for heap iteration and callbacks. It only includes heaps that are currently
   part of that. */
enum pas_heap_kind {
    pas_bootstrap_free_heap_kind,
    pas_small_medium_bootstrap_free_heap_kind,
    pas_compact_bootstrap_free_heap_kind,
    pas_large_utility_free_heap_kind,
    pas_immortal_heap_kind,
    pas_utility_heap_kind,
    pas_compact_expendable_heap_kind,
    pas_large_expendable_heap_kind
};

typedef enum pas_heap_kind pas_heap_kind;

#define PAS_NUM_HEAP_KINDS 8

static inline const char* pas_heap_kind_get_string(pas_heap_kind kind)
{
    switch (kind) {
    case pas_bootstrap_free_heap_kind:
        return "bootstrap_free_heap";
    case pas_small_medium_bootstrap_free_heap_kind:
        return "small_medium_bootstrap_free_heap";
    case pas_compact_bootstrap_free_heap_kind:
        return "compact_bootstrap_free_heap";
    case pas_large_utility_free_heap_kind:
        return "large_utility_free_heap";
    case pas_immortal_heap_kind:
        return "immortal_heap";
    case pas_utility_heap_kind:
        return "utility_heap";
    case pas_compact_expendable_heap_kind:
        return "compact_expendable_heap";
    case pas_large_expendable_heap_kind:
        return "large_expendable_heap";
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

PAS_END_EXTERN_C;

#endif /* PAS_HEAP_KIND_H */

