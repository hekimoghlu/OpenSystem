/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 14, 2022.
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
#ifndef PAS_INTRINSIC_HEAP_SUPPORT_H
#define PAS_INTRINSIC_HEAP_SUPPORT_H

#include "pas_allocator_index.h"
#include "pas_compact_atomic_segregated_size_directory_ptr.h"
#include "pas_internal_config.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_intrinsic_heap_support;
typedef struct pas_intrinsic_heap_support pas_intrinsic_heap_support;

struct pas_intrinsic_heap_support {
    pas_compact_atomic_segregated_size_directory_ptr index_to_size_directory[
        PAS_NUM_INTRINSIC_SIZE_CLASSES];
    pas_allocator_index index_to_allocator_index[PAS_NUM_INTRINSIC_SIZE_CLASSES];
#ifdef __cplusplus
    constexpr pas_intrinsic_heap_support(cpp_initialization_t)
        : index_to_size_directory { }
        , index_to_allocator_index { }
    {
        for (unsigned i = 0; i < PAS_NUM_INTRINSIC_SIZE_CLASSES; ++i) {
            index_to_size_directory[i] = PAS_COMPACT_ATOMIC_PTR_INITIALIZER;
            index_to_allocator_index[i] = 0;
        }
    }
#endif
};

#ifdef __cplusplus
#define PAS_INTRINSIC_HEAP_SUPPORT_INITIALIZER { cpp_initialization }
#else
#define PAS_INTRINSIC_HEAP_SUPPORT_INITIALIZER { \
        .index_to_size_directory = {[0 ... PAS_NUM_INTRINSIC_SIZE_CLASSES - 1] = \
                                        PAS_COMPACT_ATOMIC_PTR_INITIALIZER}, \
        .index_to_allocator_index = {[0 ... PAS_NUM_INTRINSIC_SIZE_CLASSES - 1] = 0}, \
    }
#endif

PAS_END_EXTERN_C;

#endif /* PAS_INTRINSIC_HEAP_SUPPORT_H */

