/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 7, 2024.
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
#ifndef BMALLOC_HEAP_REF_H
#define BMALLOC_HEAP_REF_H

#include "bmalloc_type.h"
#include "pas_heap_ref.h"

PAS_BEGIN_EXTERN_C;

#define BMALLOC_HEAP_REF_INITIALIZER(passed_type) \
    ((pas_heap_ref){ \
         .type = (const pas_heap_type*)(passed_type), \
         .heap = NULL, \
         .allocator_index = 0 \
     })

#define BMALLOC_PRIMITIVE_HEAP_REF_INITIALIZER_IMPL(passed_type) \
    ((pas_primitive_heap_ref){ \
         .base = BMALLOC_HEAP_REF_INITIALIZER(passed_type), \
         .cached_index = UINT_MAX \
     })

#define BMALLOC_FLEX_HEAP_REF_INITIALIZER(passed_type) \
    BMALLOC_PRIMITIVE_HEAP_REF_INITIALIZER_IMPL(passed_type)

#define BMALLOC_AUXILIARY_HEAP_REF_INITIALIZER(passed_type) \
    BMALLOC_PRIMITIVE_HEAP_REF_INITIALIZER_IMPL(passed_type)

PAS_END_EXTERN_C;

#endif /* BMALLOC_HEAP_REF_H */

