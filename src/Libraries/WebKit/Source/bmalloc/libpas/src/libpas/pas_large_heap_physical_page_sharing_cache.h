/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 17, 2025.
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
#ifndef PAS_LARGE_HEAP_PHYSICAL_PAGE_SHARING_CACHE_H
#define PAS_LARGE_HEAP_PHYSICAL_PAGE_SHARING_CACHE_H

#include "pas_bootstrap_heap_page_provider.h"
#include "pas_enumerable_range_list.h"
#include "pas_simple_large_free_heap.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_heap_config;
struct pas_large_heap_physical_page_sharing_cache;
typedef struct pas_heap_config pas_heap_config;
typedef struct pas_large_heap_physical_page_sharing_cache pas_large_heap_physical_page_sharing_cache;

struct pas_large_heap_physical_page_sharing_cache {
    pas_simple_large_free_heap free_heap;
    pas_heap_page_provider provider;
    void* provider_arg;
};

#define PAS_MEGAPAGE_LARGE_FREE_HEAP_PHYSICAL_PAGE_SHARING_CACHE_INITIALIZER \
    ((pas_large_heap_physical_page_sharing_cache){ \
         .free_heap = PAS_SIMPLE_LARGE_FREE_HEAP_INITIALIZER, \
         .provider = pas_small_medium_bootstrap_heap_page_provider, \
         .provider_arg = NULL \
     })

#define PAS_LARGE_FREE_HEAP_PHYSICAL_PAGE_SHARING_CACHE_INITIALIZER \
    ((pas_large_heap_physical_page_sharing_cache){ \
         .free_heap = PAS_SIMPLE_LARGE_FREE_HEAP_INITIALIZER, \
         .provider = pas_bootstrap_heap_page_provider, \
         .provider_arg = NULL \
     })

PAS_API extern pas_enumerable_range_list pas_large_heap_physical_page_sharing_cache_page_list;

PAS_API void
pas_large_heap_physical_page_sharing_cache_construct(
    pas_large_heap_physical_page_sharing_cache* cache,
    pas_heap_page_provider provider,
    void* provider_arg);

/* NOTE: should_zero should have a consistent value for all calls to try_allocate for a given
   cache. */
PAS_API pas_allocation_result
pas_large_heap_physical_page_sharing_cache_try_allocate_with_alignment(
    pas_large_heap_physical_page_sharing_cache* cache,
    size_t size,
    pas_alignment alignment,
    const pas_heap_config* config,
    bool should_zero);

PAS_END_EXTERN_C;

#endif /* PAS_LARGE_HEAP_PHYSICAL_PAGE_SHARING_CACHE_H */

