/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 6, 2024.
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
#ifndef PAS_DESIGNATED_INTRINSIC_HEAP_H
#define PAS_DESIGNATED_INTRINSIC_HEAP_H

#include "pas_allocator_index.h"
#include "pas_heap_config.h"

PAS_BEGIN_EXTERN_C;

struct pas_segregated_heap;
typedef struct pas_segregated_heap pas_segregated_heap;

enum pas_intrinsic_heap_designation_mode {
    pas_intrinsic_heap_is_not_designated,
    pas_intrinsic_heap_is_designated
};

typedef enum pas_intrinsic_heap_designation_mode pas_intrinsic_heap_designation_mode;

/* This can only be done once ever. Once you do it for an intrinsic heap you cannot do it for
   any other intrinsic heaps. It sets up the ability to use the fast size class lookup function. */
PAS_API void pas_designated_intrinsic_heap_initialize(pas_segregated_heap* heap,
                                                      const pas_heap_config* config);

PAS_END_EXTERN_C;

#endif /* PAS_DESIGNATED_INTRINSIC_HEAP_H */

