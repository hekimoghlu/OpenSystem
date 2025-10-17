/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 11, 2024.
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
#ifndef PAS_GET_PAGE_BASE_AND_KIND_FOR_SMALL_OTHER_IN_FAST_MEGAPAGE_H
#define PAS_GET_PAGE_BASE_AND_KIND_FOR_SMALL_OTHER_IN_FAST_MEGAPAGE_H

#include "pas_heap_config.h"
#include "pas_page_base_and_kind.h"

PAS_BEGIN_EXTERN_C;

static PAS_ALWAYS_INLINE pas_page_base_and_kind
pas_get_page_base_and_kind_for_small_other_in_fast_megapage(uintptr_t begin, pas_heap_config config)
{
    if (config.small_bitfit_config.base.is_enabled
        && config.small_bitfit_is_in_megapage) {
        pas_page_base* page_base;
        page_base = pas_page_base_for_address_and_page_config(begin, config.small_bitfit_config.base);
        if (config.small_segregated_config.base.is_enabled
            && config.small_segregated_is_in_megapage) {
            PAS_ASSERT(config.small_bitfit_config.base.page_size
                       == config.small_segregated_config.base.page_size);
            PAS_ASSERT(
                pas_page_base_for_address_and_page_config(begin, config.small_segregated_config.base)
                == page_base);
            return pas_page_base_and_kind_create(page_base, pas_page_base_get_kind(page_base));
        }
        return pas_page_base_and_kind_create(page_base, pas_small_bitfit_page_kind);
    }

    /* We shouldn't get here unless we think that either small bitfit or small segregated is in megapage.
       So, if small bitfit isn't, then small segregated must be. */
    PAS_ASSERT(config.small_segregated_config.base.is_enabled
               && config.small_segregated_is_in_megapage);
    
    return pas_page_base_and_kind_create(
        pas_page_base_for_address_and_page_config(begin, config.small_segregated_config.base),
        pas_small_shared_segregated_page_kind);
}

PAS_END_EXTERN_C;

#endif /* PAS_GET_PAGE_BASE_AND_KIND_FOR_SMALL_OTHER_IN_FAST_MEGAPAGE_H */

