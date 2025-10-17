/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 31, 2022.
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

#include "pas_heap_config_utils.h"

#include "pas_basic_heap_config_enumerator_data.h"
#include "pas_config.h"
#include "pas_heap_config_utils_inlines.h"
#include "pas_large_heap_physical_page_sharing_cache.h"
#include "pas_root.h"
#include "pas_segregated_page.h"

void pas_heap_config_utils_null_activate(void)
{
}

bool pas_heap_config_utils_for_each_shared_page_directory(
    pas_segregated_heap* heap,
    bool (*callback)(pas_segregated_shared_page_directory* directory,
                     void* arg),
    void* arg)
{
    pas_segregated_page_config_variant variant;
    pas_basic_heap_runtime_config* runtime_config;

    runtime_config = (pas_basic_heap_runtime_config*)heap->runtime_config;

    for (PAS_EACH_SEGREGATED_PAGE_CONFIG_VARIANT_ASCENDING(variant)) {
        if (!pas_shared_page_directory_by_size_for_each(
                pas_basic_heap_page_caches_get_shared_page_directories(
                    runtime_config->page_caches, variant),
                callback, arg))
            return false;
    }

    return true;
}

bool pas_heap_config_utils_for_each_shared_page_directory_remote(
    pas_enumerator* enumerator,
    pas_segregated_heap* heap,
    bool (*callback)(pas_enumerator* enumerator,
                     pas_segregated_shared_page_directory* directory,
                     void* arg),
    void* arg)
{
    pas_basic_heap_runtime_config* runtime_config;
    pas_basic_heap_page_caches* page_caches;
    pas_segregated_page_config_variant variant;

    runtime_config = pas_enumerator_read(
        enumerator, heap->runtime_config, sizeof(pas_basic_heap_runtime_config));
    if (!runtime_config)
        return false;

    page_caches = pas_enumerator_read(
        enumerator, runtime_config->page_caches, sizeof(pas_basic_heap_page_caches));
    if (!page_caches)
        return false;

    for (PAS_EACH_SEGREGATED_PAGE_CONFIG_VARIANT_ASCENDING(variant)) {
        if (!pas_shared_page_directory_by_size_for_each_remote(
                pas_basic_heap_page_caches_get_shared_page_directories(page_caches, variant),
                enumerator, callback, arg))
            return false;
    }

    return true;
}

pas_aligned_allocation_result
pas_heap_config_utils_allocate_aligned(
    size_t size,
    pas_alignment alignment,
    pas_large_heap* large_heap,
    const pas_heap_config* config,
    bool should_zero)
{
    static const bool verbose = false;
    
    pas_large_heap_physical_page_sharing_cache* cache;
    pas_aligned_allocation_result result;
    pas_allocation_result allocation_result;
    pas_zero_mode zero_mode;
    size_t aligned_size;
    pas_basic_heap_runtime_config* runtime_config;

    PAS_UNUSED_PARAM(config);

    pas_zero_memory(&result, sizeof(result));
    
    aligned_size = pas_round_up_to_power_of_2(size, alignment.alignment);

    runtime_config = (pas_basic_heap_runtime_config*)
        pas_heap_for_large_heap(large_heap)->segregated_heap.runtime_config;
    if (large_heap->is_megapage_heap)
        cache = &runtime_config->page_caches->megapage_large_heap_cache;
    else
        cache = &runtime_config->page_caches->large_heap_cache;
    
    allocation_result =
        pas_large_heap_physical_page_sharing_cache_try_allocate_with_alignment(
            cache, aligned_size, alignment, config, should_zero);
    if (!allocation_result.did_succeed)
        return result;

    if (verbose) {
        pas_log("Got allocation %p...%p\n",
                (void*)allocation_result.begin,
                (char*)allocation_result.begin + aligned_size);
    }
    
    zero_mode = allocation_result.zero_mode;

    if (should_zero)
        PAS_ASSERT(zero_mode);

    result.result = (void*)allocation_result.begin;
    result.result_size = size;
    result.left_padding = (void*)allocation_result.begin;
    result.left_padding_size = 0;
    result.right_padding = (char*)(void*)allocation_result.begin + size;
    result.right_padding_size = aligned_size - size;
    result.zero_mode = zero_mode;
    
    return result;
}

void* pas_heap_config_utils_prepare_to_enumerate(pas_enumerator* enumerator,
                                                 const pas_heap_config* my_config)
{
    pas_basic_heap_config_enumerator_data* result;
    const pas_heap_config** configs;
    const pas_heap_config* config;
    pas_basic_heap_config_root_data* root_data;

    configs = pas_enumerator_read(
        enumerator, enumerator->root->heap_configs,
        sizeof(const pas_heap_config*) * pas_heap_config_kind_num_kinds);
    if (!configs)
        return NULL;
    
    config = pas_enumerator_read(enumerator, (void*)(uintptr_t)configs[my_config->kind], sizeof(pas_heap_config));
    if (!config)
        return NULL;

    root_data = pas_enumerator_read(
        enumerator, config->root_data, sizeof(pas_basic_heap_config_root_data));
    if (!root_data)
        return NULL;

    result = pas_enumerator_allocate(enumerator, sizeof(pas_basic_heap_config_enumerator_data));
    
    pas_ptr_hash_map_construct(&result->page_header_table);

    if (!pas_basic_heap_config_enumerator_data_add_page_header_table(
            result,
            enumerator,
            pas_enumerator_read(
                enumerator, root_data->medium_page_header_table, sizeof(pas_page_header_table))))
        return NULL;
    
    if (!pas_basic_heap_config_enumerator_data_add_page_header_table(
            result,
            enumerator,
            pas_enumerator_read(
                enumerator, root_data->marge_page_header_table, sizeof(pas_page_header_table))))
        return NULL;
    
    return result;
}

#endif /* LIBPAS_ENABLED */
