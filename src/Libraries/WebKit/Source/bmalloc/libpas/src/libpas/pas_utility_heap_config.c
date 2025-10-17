/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 13, 2023.
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

#include "pas_utility_heap_config.h"

#include "pas_compact_bootstrap_free_heap.h"
#include "pas_heap_config_utils_inlines.h"
#include "pas_stream.h"
#include "pas_utility_heap.h"

PAS_BEGIN_EXTERN_C;

const pas_heap_config pas_utility_heap_config = PAS_UTILITY_HEAP_CONFIG;

PAS_SEGREGATED_PAGE_CONFIG_SPECIALIZATION_DEFINITIONS(
    pas_utility_heap_page_config, PAS_UTILITY_HEAP_CONFIG.small_segregated_config);
PAS_HEAP_CONFIG_SPECIALIZATION_DEFINITIONS(
    pas_utility_heap_config, PAS_UTILITY_HEAP_CONFIG);

void* pas_utility_heap_allocate_page(
    pas_segregated_heap* heap, pas_physical_memory_transaction* transaction, pas_segregated_page_role role)
{
    PAS_UNUSED_PARAM(heap);
    PAS_ASSERT(!transaction);
    PAS_ASSERT(role == pas_segregated_page_exclusive_role);
    return (void*)pas_compact_bootstrap_free_heap_try_allocate_with_alignment(
        PAS_SMALL_PAGE_DEFAULT_SIZE,
        pas_alignment_create_traditional(PAS_SMALL_PAGE_DEFAULT_SIZE),
        "pas_utility_heap/page",
        pas_delegate_allocation).begin;
}

pas_segregated_shared_page_directory*
pas_utility_heap_shared_page_directory_selector(pas_segregated_heap* heap,
                                                pas_segregated_size_directory* directory)
{
    PAS_UNUSED_PARAM(heap);
    PAS_UNUSED_PARAM(directory);
    PAS_ASSERT(!"Not implemented");
    return NULL;
}

bool pas_utility_heap_config_for_each_shared_page_directory(
    pas_segregated_heap* heap,
    bool (*callback)(pas_segregated_shared_page_directory* directory,
                     void* arg),
    void* arg)
{
    PAS_ASSERT(heap == &pas_utility_segregated_heap);
    PAS_UNUSED_PARAM(callback);
    PAS_UNUSED_PARAM(arg);
    return true;
}

void pas_utility_heap_config_dump_shared_page_directory_arg(
    pas_stream* stream, pas_segregated_shared_page_directory* directory)
{
    PAS_UNUSED_PARAM(stream);
    PAS_UNUSED_PARAM(directory);
    PAS_ASSERT(!"Should not be reached");
}

PAS_END_EXTERN_C;

#endif /* LIBPAS_ENABLED */
