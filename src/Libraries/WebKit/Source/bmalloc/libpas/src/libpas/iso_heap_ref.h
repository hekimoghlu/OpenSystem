/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 5, 2022.
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
#ifndef ISO_HEAP_REF_H
#define ISO_HEAP_REF_H

#include "pas_heap_ref.h"
#include "pas_simple_type.h"

PAS_BEGIN_EXTERN_C;

#define ISO_HEAP_REF_INITIALIZER_TLC_PART \
    .allocator_index = 0

#define ISO_HEAP_REF_INITIALIZER_WITH_ALIGNMENT(type_size, alignment) \
    ((pas_heap_ref){ \
         .type = (const pas_heap_type*)PAS_SIMPLE_TYPE_CREATE((type_size), (alignment)), \
         .heap = NULL, \
         ISO_HEAP_REF_INITIALIZER_TLC_PART \
     })

#define ISO_HEAP_REF_INITIALIZER(type_size) \
    ISO_HEAP_REF_INITIALIZER_WITH_ALIGNMENT((type_size), 1)

#define ISO_PRIMITIVE_HEAP_REF_INITIALIZER \
    ((pas_primitive_heap_ref){ \
         .base = ISO_HEAP_REF_INITIALIZER(1), \
         .cached_index = UINT_MAX \
     })

PAS_END_EXTERN_C;

#endif /* ISO_HEAP_REF_H */

