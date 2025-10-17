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
#ifndef PAS_GET_ALLOCATION_SIZE_H
#define PAS_GET_ALLOCATION_SIZE_H

#include "pas_get_page_base_and_kind_for_small_other_in_fast_megapage.h"
#include "pas_heap_config.h"
#include "pas_heap_lock.h"
#include "pas_large_map.h"
#include "pas_segregated_page_inlines.h"
#include "pas_segregated_size_directory.h"

PAS_BEGIN_EXTERN_C;

/* Returns zero if we don't own this object. */
static PAS_ALWAYS_INLINE size_t pas_get_allocation_size(void* ptr,
                                                        pas_heap_config config)
{
    uintptr_t begin;
    
    begin = (uintptr_t)ptr;

    switch (config.fast_megapage_kind_func(begin)) {
    case pas_small_exclusive_segregated_fast_megapage_kind:
        return pas_segregated_page_get_object_size_for_address_and_page_config(
            begin, config.small_segregated_config, pas_segregated_page_exclusive_role);
    case pas_small_other_fast_megapage_kind: {
        pas_page_base_and_kind page_and_kind;
        page_and_kind = pas_get_page_base_and_kind_for_small_other_in_fast_megapage(begin, config);
        switch (page_and_kind.page_kind) {
        case pas_small_shared_segregated_page_kind:
            return pas_segregated_page_get_object_size_for_address_in_page(
                pas_page_base_get_segregated(page_and_kind.page_base),
                begin,
                config.small_segregated_config,
                pas_segregated_page_shared_role);
        case pas_small_bitfit_page_kind:
            return config.small_bitfit_config.specialized_page_get_allocation_size_with_page(
                pas_page_base_get_bitfit(page_and_kind.page_base),
                begin);
        default:
            PAS_ASSERT(!"Should not be reached");
            return 0;
        }
    }
    case pas_not_a_fast_megapage_kind: {
        pas_page_base* page_base;
        pas_large_map_entry entry;
        size_t result;

        page_base = config.page_header_func(begin);
        if (page_base) {
            switch (pas_page_base_get_kind(page_base)) {
            case pas_small_shared_segregated_page_kind:
                PAS_ASSERT(!config.small_segregated_is_in_megapage);
                return pas_segregated_page_get_object_size_for_address_in_page(
                    pas_page_base_get_segregated(page_base),
                    begin,
                    config.small_segregated_config,
                    pas_segregated_page_shared_role);
            case pas_small_exclusive_segregated_page_kind:
                PAS_ASSERT(!config.small_segregated_is_in_megapage);
                return pas_segregated_page_get_object_size_for_address_in_page(
                    pas_page_base_get_segregated(page_base),
                    begin,
                    config.small_segregated_config,
                    pas_segregated_page_exclusive_role);
            case pas_small_bitfit_page_kind:
                PAS_ASSERT(!config.small_bitfit_is_in_megapage);
                return config.small_bitfit_config.specialized_page_get_allocation_size_with_page(
                    pas_page_base_get_bitfit(page_base),
                    begin);
            case pas_medium_shared_segregated_page_kind:
                return pas_segregated_page_get_object_size_for_address_in_page(
                    pas_page_base_get_segregated(page_base),
                    begin,
                    config.medium_segregated_config,
                    pas_segregated_page_shared_role);
            case pas_medium_exclusive_segregated_page_kind:
                return pas_segregated_page_get_object_size_for_address_in_page(
                    pas_page_base_get_segregated(page_base),
                    begin,
                    config.medium_segregated_config,
                    pas_segregated_page_exclusive_role);
            case pas_medium_bitfit_page_kind:
                return config.medium_bitfit_config.specialized_page_get_allocation_size_with_page(
                    pas_page_base_get_bitfit(page_base),
                    begin);
            case pas_marge_bitfit_page_kind:
                return config.marge_bitfit_config.specialized_page_get_allocation_size_with_page(
                    pas_page_base_get_bitfit(page_base),
                    begin);
            }
            PAS_ASSERT(!"Bad page kind");
            return 0;
        }
        
        pas_heap_lock_lock();
        
        entry = pas_large_map_find(begin);
        
        if (!pas_large_map_entry_is_empty(entry)) {
            PAS_PROFILE(LARGE_MAP_FOUND_ENTRY, &config, entry.begin, entry.end);
            PAS_ASSERT(entry.begin == begin);
            PAS_ASSERT(entry.end > begin);
            
            result = entry.end - begin;
        } else
            result = 0;
        
        pas_heap_lock_unlock();
        
        return result;
    } }
    
    PAS_ASSERT(!"Should not be reached");
    return 0;
}

PAS_END_EXTERN_C;

#endif /* PAS_GET_ALLOCATION_SIZE */

