/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 23, 2022.
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

#include "pas_create_basic_heap_page_caches_with_reserved_memory.h"

#include "pas_basic_heap_page_caches.h"
#include "pas_basic_heap_runtime_config.h"
#include "pas_immortal_heap.h"
#include "pas_large_heap_physical_page_sharing_cache.h"
#include "pas_megapage_cache.h"
#include "pas_reserved_memory_provider.h"
#include "pas_segregated_shared_page_directory.h"

static pas_allocation_result allocate_from_megapages(
    size_t size,
    pas_alignment alignment,
    const char* name,
    pas_heap* heap,
    pas_physical_memory_transaction* transaction,
    void* arg)
{
    const pas_heap_config* heap_config;

    PAS_UNUSED_PARAM(name);
    PAS_ASSERT(heap);
    PAS_ASSERT(transaction);
    PAS_ASSERT(!arg);
    PAS_ASSERT(!alignment.alignment_begin);

    heap_config = pas_heap_config_kind_get_config(heap->config_kind);

    PAS_PROFILE(MEGAPAGES_ALLOCATION, heap, size, alignment.alignment, heap_config);

    return pas_large_heap_try_allocate_and_forget(
        &heap->large_heap, size, alignment.alignment, pas_non_compact_allocation_mode,
        heap_config, transaction);
}

/* Warning: This creates caches that allow type confusion. Only use this for primitive heaps! */
pas_basic_heap_page_caches* pas_create_basic_heap_page_caches_with_reserved_memory(
    pas_basic_heap_runtime_config* template_runtime_config,
    uintptr_t begin,
    uintptr_t end)
{
    pas_reserved_memory_provider* provider;
    pas_basic_heap_page_caches* caches;
    pas_segregated_page_config_variant segregated_variant;

    pas_heap_lock_lock();

    provider = pas_immortal_heap_allocate(
        sizeof(pas_reserved_memory_provider),
        "pas_reserved_memory_provider",
        pas_object_allocation);

    pas_reserved_memory_provider_construct(provider, begin, end);

    caches = pas_immortal_heap_allocate(
        sizeof(pas_basic_heap_page_caches),
        "pas_basic_heap_page_caches",
        pas_object_allocation);

    pas_large_heap_physical_page_sharing_cache_construct(
        &caches->megapage_large_heap_cache,
        pas_reserved_memory_provider_try_allocate,
        provider);

    pas_large_heap_physical_page_sharing_cache_construct(
        &caches->large_heap_cache,
        pas_reserved_memory_provider_try_allocate,
        provider);
    
    pas_megapage_cache_construct(
        &caches->small_exclusive_segregated_megapage_cache,
        allocate_from_megapages,
        NULL);

    pas_megapage_cache_construct(
        &caches->small_other_megapage_cache,
        allocate_from_megapages,
        NULL);

    pas_megapage_cache_construct(
        &caches->medium_megapage_cache,
        allocate_from_megapages,
        NULL);

    for (PAS_EACH_SEGREGATED_PAGE_CONFIG_VARIANT_ASCENDING(segregated_variant)) {
        pas_shared_page_directory_by_size* directories;

        directories = pas_basic_heap_page_caches_get_shared_page_directories(caches,
                                                                             segregated_variant);

        *directories = PAS_SHARED_PAGE_DIRECTORY_BY_SIZE_INITIALIZER(
            pas_basic_heap_page_caches_get_shared_page_directories(
                template_runtime_config->page_caches,
                segregated_variant)->log_shift,
            pas_share_pages);
    }

    pas_heap_lock_unlock();

    return caches;
}

#endif /* LIBPAS_ENABLED */
