/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 23, 2025.
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
#ifndef PAS_GET_HEAP_H
#define PAS_GET_HEAP_H

#include "pas_bitfit_directory.h"
#include "pas_get_page_base_and_kind_for_small_other_in_fast_megapage.h"
#include "pas_heap.h"
#include "pas_heap_config.h"
#include "pas_large_map.h"
#include "pas_segregated_page_inlines.h"
#include "pas_segregated_size_directory.h"

PAS_BEGIN_EXTERN_C;

static PAS_ALWAYS_INLINE pas_heap* pas_get_heap(void* ptr,
                                                pas_heap_config config)
{
    uintptr_t begin;
    pas_page_base* page_base;
    
    begin = (uintptr_t)ptr;
    
    switch (config.fast_megapage_kind_func(begin)) {
    case pas_small_exclusive_segregated_fast_megapage_kind:
        return pas_heap_for_segregated_heap(
            pas_segregated_page_get_directory_for_address_and_page_config(
                begin, config.small_segregated_config, pas_segregated_page_exclusive_role)->heap);
    case pas_small_other_fast_megapage_kind: {
        pas_page_base_and_kind page_and_kind;
        page_and_kind = pas_get_page_base_and_kind_for_small_other_in_fast_megapage(begin, config);
        switch (page_and_kind.page_kind) {
        case pas_small_shared_segregated_page_kind:
            return pas_heap_for_segregated_heap(
                pas_segregated_page_get_directory_for_address_in_page(
                    pas_page_base_get_segregated(page_and_kind.page_base),
                    begin, config.small_segregated_config, pas_segregated_page_shared_role)->heap);
        case pas_small_bitfit_page_kind:
            page_base = page_and_kind.page_base;
            goto bitfit_case_with_page_base;
        default:
            PAS_ASSERT(!"Should not be reached");
            return NULL;
        }
    }
    case pas_not_a_fast_megapage_kind: {
        pas_large_map_entry entry;
        pas_heap* result;

        page_base = config.page_header_func(begin);
        if (page_base) {
            switch (pas_page_base_get_kind(page_base)) {
            case pas_small_shared_segregated_page_kind:
                PAS_ASSERT(!config.small_segregated_is_in_megapage);
                return pas_heap_for_segregated_heap(
                    pas_segregated_page_get_directory_for_address_in_page(
                        pas_page_base_get_segregated(page_base),
                        begin, config.small_segregated_config, pas_segregated_page_shared_role)->heap);
            case pas_small_exclusive_segregated_page_kind:
                PAS_ASSERT(!config.small_segregated_is_in_megapage);
                return pas_heap_for_segregated_heap(
                    pas_segregated_page_get_directory_for_address_in_page(
                        pas_page_base_get_segregated(page_base),
                        begin, config.small_segregated_config, pas_segregated_page_exclusive_role)->heap);
            case pas_medium_shared_segregated_page_kind:
                return pas_heap_for_segregated_heap(
                    pas_segregated_page_get_directory_for_address_in_page(
                        pas_page_base_get_segregated(page_base),
                        begin, config.medium_segregated_config, pas_segregated_page_shared_role)->heap);
            case pas_medium_exclusive_segregated_page_kind:
                return pas_heap_for_segregated_heap(
                    pas_segregated_page_get_directory_for_address_in_page(
                        pas_page_base_get_segregated(page_base),
                        begin, config.medium_segregated_config, pas_segregated_page_exclusive_role)->heap);
            case pas_small_bitfit_page_kind:
            case pas_medium_bitfit_page_kind:
            case pas_marge_bitfit_page_kind:
                goto bitfit_case_with_page_base;
            }
            PAS_ASSERT(!"Bad page kind");
            return NULL;
        }
        
        pas_heap_lock_lock();
        
        entry = pas_large_map_find(begin);
        
        PAS_ASSERT(!pas_large_map_entry_is_empty(entry));
        PAS_PROFILE(LARGE_MAP_FOUND_ENTRY, &config, entry.begin, entry.end);
        PAS_ASSERT(entry.begin == begin);
        PAS_ASSERT(entry.end > begin);
        
        result = pas_heap_for_large_heap(entry.heap);
        
        pas_heap_lock_unlock();
        
        return result;
    } }
    
    PAS_ASSERT(!"Should not be reached");
    return NULL;

bitfit_case_with_page_base:
    return pas_heap_for_segregated_heap(
        pas_compact_bitfit_directory_ptr_load_non_null(
            &pas_compact_atomic_bitfit_view_ptr_load_non_null(
                &pas_page_base_get_bitfit(page_base)->owner)->directory)->heap);
}

PAS_END_EXTERN_C;

#endif /* PAS_GET_HEAP */

