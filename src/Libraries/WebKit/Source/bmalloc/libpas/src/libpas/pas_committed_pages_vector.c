/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 24, 2023.
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
#include "pas_config.h"

#if LIBPAS_ENABLED

#include "pas_committed_pages_vector.h"

#include "pas_allocation_config.h"
#include "pas_page_malloc.h"

PAS_BEGIN_EXTERN_C;

void pas_committed_pages_vector_construct(pas_committed_pages_vector* vector,
                                          void* object,
                                          size_t size,
                                          const pas_allocation_config* allocation_config)
{
    size_t page_size;
    size_t page_size_shift;
    size_t num_pages;

    page_size = pas_page_malloc_alignment();
    page_size_shift = pas_page_malloc_alignment_shift();

    PAS_ASSERT(pas_is_aligned((uintptr_t)object, page_size));
    PAS_ASSERT(pas_is_aligned(size, page_size));

    num_pages = size >> page_size_shift;

    vector->raw_data = allocation_config->allocate(
        num_pages, "pas_committed_pages_vector/raw_data", pas_object_allocation, allocation_config->arg);
    vector->size = num_pages;

#if PAS_OS(LINUX)
    PAS_SYSCALL(mincore(object, size, (unsigned char*)vector->raw_data));
#else
    PAS_SYSCALL(mincore(object, size, vector->raw_data));
#endif
}

void pas_committed_pages_vector_destruct(pas_committed_pages_vector* vector,
                                         const pas_allocation_config* allocation_config)
{
    allocation_config->deallocate(
        vector->raw_data, vector->size, pas_object_allocation, allocation_config->arg);
}

size_t pas_committed_pages_vector_count_committed(pas_committed_pages_vector* vector)
{
    size_t result;
    size_t index;
    result = 0;
    for (index = vector->size; index--;)
        result += (size_t)pas_committed_pages_vector_is_committed(vector, index);
    return result;
}

size_t pas_count_committed_pages(void* object,
                                 size_t size,
                                 const pas_allocation_config* allocation_config)
{
    size_t result;
    pas_committed_pages_vector vector;
    pas_committed_pages_vector_construct(&vector, object, size, allocation_config);
    result = pas_committed_pages_vector_count_committed(&vector);
    pas_committed_pages_vector_destruct(&vector, allocation_config);
    return result;
}

PAS_END_EXTERN_C;

#endif /* LIBPAS_ENABLED */

