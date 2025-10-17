/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 31, 2022.
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
#ifndef PAS_PAGE_BASE_INLINES_H
#define PAS_PAGE_BASE_INLINES_H

#include "pas_log.h"
#include "pas_page_base.h"

PAS_BEGIN_EXTERN_C;

typedef struct {
    bool did_find_empty_granule;
} pas_page_base_free_granule_uses_in_range_data;

static PAS_ALWAYS_INLINE void pas_page_base_free_granule_uses_in_range_action(
    pas_page_granule_use_count* use_count_ptr,
    void* arg)
{
    static const bool verbose = false;
    
    pas_page_base_free_granule_uses_in_range_data* data;
    pas_page_granule_use_count use_count;

    data = (pas_page_base_free_granule_uses_in_range_data*)arg;

    if (verbose)
        pas_log("Decrementing use count at %p\n", use_count_ptr);
    
    use_count = *use_count_ptr;
    
    /* I'm assuming that we do have available cycles for asserts here. */
    PAS_ASSERT(use_count);
    PAS_ASSERT(use_count != PAS_PAGE_GRANULE_DECOMMITTED);
    
    use_count--;
    
    *use_count_ptr = use_count;
    
    if (!use_count)
        data->did_find_empty_granule = true;
}

/* Returns true if we found an empty granule. */
static PAS_ALWAYS_INLINE bool pas_page_base_free_granule_uses_in_range(
    pas_page_granule_use_count* use_count_ptr,
    uintptr_t begin_offset,
    uintptr_t end_offset,
    pas_page_base_config page_config)
{
    pas_page_base_free_granule_uses_in_range_data free_uses_in_range_data;
    free_uses_in_range_data.did_find_empty_granule = false;
    
    pas_page_granule_for_each_use_in_range(
        use_count_ptr,
        begin_offset,
        end_offset,
        page_config.page_size,
        page_config.granule_size,
        pas_page_base_free_granule_uses_in_range_action,
        &free_uses_in_range_data);
    
    return free_uses_in_range_data.did_find_empty_granule;
}

PAS_END_EXTERN_C;

#endif /* PAS_PAGE_BASE_INLINES_H */

