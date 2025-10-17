/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 7, 2023.
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

#include "pas_reserved_memory_provider.h"

static pas_aligned_allocation_result null_aligned_allocator(size_t size,
                                                            pas_alignment alignment,
                                                            void* arg)
{
    PAS_UNUSED_PARAM(size);
    PAS_UNUSED_PARAM(alignment);
    PAS_UNUSED_PARAM(arg);
    return pas_aligned_allocation_result_create_empty();
}

static void initialize_config(pas_large_free_heap_config* config)
{
    config->type_size = 1;
    config->min_alignment = 1;
    config->aligned_allocator = null_aligned_allocator;
    config->aligned_allocator_arg = NULL;
    config->deallocator = NULL;
    config->deallocator_arg = NULL;
}

void pas_reserved_memory_provider_construct(
    pas_reserved_memory_provider* provider,
    uintptr_t begin,
    uintptr_t end)
{
    pas_large_free_heap_config config;

    initialize_config(&config);
    
    pas_simple_large_free_heap_construct(&provider->free_heap);

    pas_simple_large_free_heap_deallocate(
        &provider->free_heap, begin, end, pas_zero_mode_is_all_zero, &config);
}

pas_allocation_result pas_reserved_memory_provider_try_allocate(
    size_t size,
    pas_alignment alignment,
    const char* name,
    pas_heap* heap,
    pas_physical_memory_transaction* transaction,
    void* arg)
{
    pas_reserved_memory_provider* provider;
    pas_large_free_heap_config config;

    PAS_UNUSED_PARAM(name);
    PAS_UNUSED_PARAM(heap);
    PAS_UNUSED_PARAM(transaction);

    provider = arg;

    initialize_config(&config);

    return pas_simple_large_free_heap_try_allocate(
        &provider->free_heap, size, alignment, &config);
}

#endif /* LIBPAS_ENABLED */
