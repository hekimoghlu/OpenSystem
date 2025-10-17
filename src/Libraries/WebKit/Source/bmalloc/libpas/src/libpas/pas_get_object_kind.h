/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 22, 2024.
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
#ifndef PAS_GET_OBJECT_KIND_H
#define PAS_GET_OBJECT_KIND_H

#include "pas_get_page_base_and_kind_for_small_other_in_fast_megapage.h"
#include "pas_heap_config.h"
#include "pas_object_kind.h"
#include "pas_page_base.h"
#include "pas_large_map.h"

PAS_BEGIN_EXTERN_C;

static PAS_ALWAYS_INLINE pas_object_kind pas_get_object_kind(void* ptr,
                                                             pas_heap_config config)
{
    uintptr_t begin;
    pas_page_kind page_kind;
    
    begin = (uintptr_t)ptr;
    
    switch (config.fast_megapage_kind_func(begin)) {
    case pas_small_exclusive_segregated_fast_megapage_kind:
        return pas_small_segregated_object_kind;
    case pas_small_other_fast_megapage_kind:
        page_kind = pas_get_page_base_and_kind_for_small_other_in_fast_megapage(begin, config).page_kind;
        goto use_page_kind;
    case pas_not_a_fast_megapage_kind: {
        pas_large_map_entry entry;
        pas_page_base* page_base;

        page_base = config.page_header_func(begin);
        if (page_base) {
            page_kind = pas_page_base_get_kind(page_base);
            goto use_page_kind;
        }
        
        pas_heap_lock_lock();
        
        entry = pas_large_map_find(begin);
        
        pas_heap_lock_unlock();
        
        if (pas_large_map_entry_is_empty(entry))
            return pas_not_an_object_kind;
        return pas_large_object_kind;
    } }
    
    PAS_ASSERT(!"Should not be reached");
    return pas_not_an_object_kind;

use_page_kind:
    return pas_object_kind_for_page_kind(page_kind);
}

PAS_END_EXTERN_C;

#endif /* PAS_GET_OBJECT_KIND */

