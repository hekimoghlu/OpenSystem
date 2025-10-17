/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 14, 2022.
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
#ifndef PAS_PRIMITIVE_HEAP_REF_H
#define PAS_PRIMITIVE_HEAP_REF_H

#include "pas_heap_ref.h"

PAS_BEGIN_EXTERN_C;

struct pas_primitive_heap_ref;
typedef struct pas_primitive_heap_ref pas_primitive_heap_ref;

struct pas_primitive_heap_ref {
    pas_heap_ref base;
    unsigned cached_index; /* Initialize this to UINT_MAX */
};

PAS_API extern uint64_t pas_primitive_heap_ref_allocate_slow_path_count;

PAS_END_EXTERN_C;

#endif /* PAS_PRIMITIVE_HEAP_REF_H */

