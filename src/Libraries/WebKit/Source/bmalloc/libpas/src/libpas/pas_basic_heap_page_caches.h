/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 10, 2022.
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
#ifndef PAS_BASIC_HEAP_PAGE_CACHES_H
#define PAS_BASIC_HEAP_PAGE_CACHES_H

#include "pas_large_heap_physical_page_sharing_cache.h"
#include "pas_megapage_cache.h"
#include "pas_segregated_page_config_variant.h"
#include "pas_shared_page_directory_by_size.h"

PAS_BEGIN_EXTERN_C;

struct pas_basic_heap_page_caches;
typedef struct pas_basic_heap_page_caches pas_basic_heap_page_caches;

struct pas_basic_heap_page_caches {
    pas_large_heap_physical_page_sharing_cache megapage_large_heap_cache;
    pas_large_heap_physical_page_sharing_cache large_heap_cache;
    pas_megapage_cache small_exclusive_segregated_megapage_cache;
    pas_shared_page_directory_by_size small_shared_page_directories;
    pas_megapage_cache small_other_megapage_cache;
    pas_megapage_cache medium_megapage_cache; /* The purpose of this is not for the fast megapage
                                                 table, but to make sure that medium pages are not
                                                 allocated one-at-a-time from bootstrap, since that
                                                 would create unnecessary fragmentation in the large
                                                 heap. */
    pas_shared_page_directory_by_size medium_shared_page_directories;
};

#define PAS_BASIC_HEAP_PAGE_CACHES_INITIALIZER(small_log_shift, medium_log_shift) \
    ((pas_basic_heap_page_caches){ \
        .megapage_large_heap_cache = PAS_MEGAPAGE_LARGE_FREE_HEAP_PHYSICAL_PAGE_SHARING_CACHE_INITIALIZER, \
        .large_heap_cache = PAS_LARGE_FREE_HEAP_PHYSICAL_PAGE_SHARING_CACHE_INITIALIZER, \
        .small_exclusive_segregated_megapage_cache = PAS_MEGAPAGE_CACHE_INITIALIZER, \
        .small_shared_page_directories = PAS_SHARED_PAGE_DIRECTORY_BY_SIZE_INITIALIZER( \
            (small_log_shift), pas_share_pages), \
        .small_other_megapage_cache = PAS_MEGAPAGE_CACHE_INITIALIZER, \
        .medium_megapage_cache = PAS_MEGAPAGE_CACHE_INITIALIZER, \
        .medium_shared_page_directories = PAS_SHARED_PAGE_DIRECTORY_BY_SIZE_INITIALIZER( \
            (medium_log_shift), pas_share_pages) \
    })

static inline pas_shared_page_directory_by_size*
pas_basic_heap_page_caches_get_shared_page_directories(
    pas_basic_heap_page_caches* caches,
    pas_segregated_page_config_variant variant)
{
    switch (variant) {
    case pas_medium_segregated_page_config_variant:
        return &caches->medium_shared_page_directories;
    case pas_small_segregated_page_config_variant:
        return &caches->small_shared_page_directories;
    }
    PAS_ASSERT(!"Should not be reached");
    return NULL;
}

PAS_END_EXTERN_C;

#endif /* PAS_BASIC_HEAP_PAGE_CACHES_H */

