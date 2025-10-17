/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 24, 2023.
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
#ifndef PAS_COMMITTED_PAGES_VECTOR_H
#define PAS_COMMITTED_PAGES_VECTOR_H

#include "pas_utils.h"
#include <sys/types.h>
#include <sys/mman.h>

PAS_BEGIN_EXTERN_C;

struct pas_allocation_config;
struct pas_committed_pages_vector;
typedef struct pas_allocation_config pas_allocation_config;
typedef struct pas_committed_pages_vector pas_committed_pages_vector;

struct pas_committed_pages_vector {
    char* raw_data;
    size_t size;
};

PAS_API void pas_committed_pages_vector_construct(pas_committed_pages_vector* vector,
                                                  void* object,
                                                  size_t size,
                                                  const pas_allocation_config* allocation_config);

PAS_API void pas_committed_pages_vector_destruct(pas_committed_pages_vector* vector,
                                                 const pas_allocation_config* allocation_config);

static inline bool pas_committed_pages_vector_is_committed(pas_committed_pages_vector* vector,
                                                           size_t page_index)
{
    PAS_ASSERT(page_index < vector->size);
#if PAS_OS(LINUX)
    return vector->raw_data[page_index];
#else
    return vector->raw_data[page_index] & (MINCORE_REFERENCED |
                                           MINCORE_REFERENCED_OTHER |
                                           MINCORE_MODIFIED_OTHER |
                                           MINCORE_MODIFIED);
#endif
}

PAS_API size_t pas_committed_pages_vector_count_committed(pas_committed_pages_vector* vector);

PAS_API size_t pas_count_committed_pages(void* object,
                                         size_t size,
                                         const pas_allocation_config* allocation_config);

PAS_END_EXTERN_C;

#endif /* PAS_COMMITTED_PAGES_VECTOR_H */

