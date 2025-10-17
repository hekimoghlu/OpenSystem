/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 20, 2024.
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
#ifndef PAS_HAS_OBJECT_H
#define PAS_HAS_OBJECT_H

#include "pas_heap_config.h"
#include "pas_large_map.h"

PAS_BEGIN_EXTERN_C;

static PAS_ALWAYS_INLINE bool pas_has_object(void* ptr,
                                             pas_heap_config config)
{
    uintptr_t begin;
    
    begin = (uintptr_t)ptr;
    
    switch (config.fast_megapage_kind_func(begin)) {
    case pas_small_exclusive_segregated_fast_megapage_kind:
    case pas_small_other_fast_megapage_kind:
        return true;
    case pas_not_a_fast_megapage_kind: {
        pas_large_map_entry entry;

        if (config.page_header_func(begin))
            return true;
        
        pas_heap_lock_lock();
        
        entry = pas_large_map_find(begin);
        
        pas_heap_lock_unlock();
        
        return !pas_large_map_entry_is_empty(entry);
    } }
    
    PAS_ASSERT(!"Should not be reached");
    return false;
}

PAS_END_EXTERN_C;

#endif /* PAS_HAS_OBJECT */

