/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 18, 2023.
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
#ifndef PAS_RESERVED_MEMORY_PROVIDER_H
#define PAS_RESERVED_MEMORY_PROVIDER_H

#include "pas_heap_page_provider.h"
#include "pas_simple_large_free_heap.h"

PAS_BEGIN_EXTERN_C;

struct pas_reserved_memory_provider;
typedef struct pas_reserved_memory_provider pas_reserved_memory_provider;

struct pas_reserved_memory_provider {
    pas_simple_large_free_heap free_heap;
};

PAS_API void pas_reserved_memory_provider_construct(
    pas_reserved_memory_provider* provider,
    uintptr_t begin,
    uintptr_t end);

PAS_API pas_allocation_result pas_reserved_memory_provider_try_allocate(
    size_t size,
    pas_alignment alignment,
    const char* name,
    pas_heap* heap,
    pas_physical_memory_transaction* transaction,
    void* arg);

PAS_END_EXTERN_C;

#endif /* PAS_RESERVED_MEMORY_PROVIDER_H */

