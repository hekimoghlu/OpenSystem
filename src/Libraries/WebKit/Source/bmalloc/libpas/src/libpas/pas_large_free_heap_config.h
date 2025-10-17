/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 1, 2025.
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
#ifndef PAS_LARGE_FREE_HEAP_CONFIG_H
#define PAS_LARGE_FREE_HEAP_CONFIG_H

#include "pas_aligned_allocator.h"
#include "pas_deallocator.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_large_free_heap_config;
typedef struct pas_large_free_heap_config pas_large_free_heap_config;

struct pas_large_free_heap_config {
    size_t type_size;
    
    /* This is the smallest alignment to which the sizes of objects are
       aligned. This isn't meaningful if type_size is 1. In that case, this
       should be 1 also.
       
       You don't actually have to request alignment that is at least as big
       as this. */
    size_t min_alignment;
    
    pas_aligned_allocator aligned_allocator;
    void* aligned_allocator_arg;
    
    pas_deallocator deallocator;
    void* deallocator_arg;
};

PAS_END_EXTERN_C;

#endif /* PAS_LARGE_FREE_HEAP_CONFIG */

