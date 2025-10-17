/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 15, 2025.
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
#ifndef PAS_MEGAPAGE_CACHE_H
#define PAS_MEGAPAGE_CACHE_H

#include "pas_bootstrap_heap_page_provider.h"
#include "pas_simple_large_free_heap.h"
#include "pas_small_medium_bootstrap_heap_page_provider.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_megapage_cache;
struct pas_megapage_cache_config;
typedef struct pas_megapage_cache pas_megapage_cache;
typedef struct pas_megapage_cache_config pas_megapage_cache_config;

typedef void (*pas_megapage_cache_table_set_by_index)(size_t index, void* arg);

struct pas_megapage_cache {
    pas_simple_large_free_heap free_heap;
    pas_heap_page_provider provider;
    void* provider_arg;
};

struct pas_megapage_cache_config {
    size_t megapage_size;
    size_t allocation_size;
    pas_alignment allocation_alignment;
    size_t excluded_size; /* FIXME: Remove this, we don't use it anymore. It used to be used for putting
                             medium page headers at the start of the megapage. That was before we had the
                             page header table. */
    pas_megapage_cache_table_set_by_index table_set_by_index;
    void* table_set_by_index_arg;
    bool should_zero;
};

#define PAS_MEGAPAGE_CACHE_INITIALIZER { \
        .free_heap = PAS_SIMPLE_LARGE_FREE_HEAP_INITIALIZER, \
        .provider = pas_small_medium_bootstrap_heap_page_provider, \
        .provider_arg = NULL \
    }

PAS_API void pas_megapage_cache_construct(pas_megapage_cache* cache,
                                          pas_heap_page_provider provider,
                                          void* provider_arg);

PAS_API void* pas_megapage_cache_try_allocate(pas_megapage_cache* cache,
                                              pas_megapage_cache_config* cache_config,
                                              pas_heap* heap,
                                              pas_physical_memory_transaction* transaction);

PAS_END_EXTERN_C;

#endif /* PAS_MEGAPAGE_CACHE_H */

