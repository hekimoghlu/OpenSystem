/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 24, 2023.
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
#ifndef PAS_TRY_SHRINK_H
#define PAS_TRY_SHRINK_H

/* Returns true if the object was actually shrunk. You'll get false if the heap that the object is
   in doesn't support shrinking. However, if you try to shrink something that isn't even allocated
   then you'll get a trap.

   Shrinking is supported by the bitfit and large heaps. It's not supported by the segregated heap. */
static PAS_ALWAYS_INLINE bool pas_try_shrink(void* ptr,
                                             size_t new_size,
                                             pas_heap_config config)
{
    uintptr_t begin;

    begin = (uintptr_t)ptr;

    switch (config.fast_megapage_kind_func(begin)) {
    case pas_small_exclusive_segregated_fast_megapage_kind:
        return false;
    case pas_small_other_fast_megapage_kind: {
        pas_page_base_and_kind page_and_kind;
        page_and_kind = pas_get_page_base_and_kind_for_small_other_in_fast_megapage(begin, config);
        switch (page_and_kind.page_kind) {
        case pas_small_shared_segregated_page_kind:
            return false;
        case pas_small_bitfit_page_kind:
            config.small_bitfit_config.specialized_page_shrink_with_page(
                pas_page_base_get_bitfit(page_and_kind.page_base),
                begin, new_size);
            return true;
        default:
            PAS_ASSERT(!"Should not be reached");
            return 0;
        }
    }
    case pas_not_a_fast_megapage_kind: {
        pas_page_base* page_base;
        bool shrink_result;

        page_base = config.page_header_func(begin);
        if (page_base) {
            switch (pas_page_base_get_kind(page_base)) {
            case pas_small_exclusive_segregated_page_kind:
            case pas_small_shared_segregated_page_kind:
                PAS_ASSERT(!config.small_segregated_is_in_megapage);
                return false;
            case pas_small_bitfit_page_kind:
                PAS_ASSERT(!config.small_bitfit_is_in_megapage);
                config.small_bitfit_config.specialized_page_shrink_with_page(
                    pas_page_base_get_bitfit(page_base),
                    begin, new_size);
                return true;
            case pas_medium_exclusive_segregated_page_kind:
            case pas_medium_shared_segregated_page_kind:
                return false;
            case pas_medium_bitfit_page_kind:
                config.medium_bitfit_config.specialized_page_shrink_with_page(
                    pas_page_base_get_bitfit(page_base),
                    begin, new_size);
                return true;
            case pas_marge_bitfit_page_kind:
                config.marge_bitfit_config.specialized_page_shrink_with_page(
                    pas_page_base_get_bitfit(page_base),
                    begin, new_size);
                return true;
            }
            PAS_ASSERT(!"Bad page kind");
            return false;
        }

        pas_heap_lock_lock();
        shrink_result = pas_large_heap_try_shrink(begin, new_size, config.config_ptr);
        pas_heap_lock_unlock();
        if (!shrink_result)
            pas_deallocation_did_fail("Object not allocated", begin);
        return true;
    } }

    PAS_ASSERT(!"Should not be reached");
    return false;
}

#endif /* PAS_TRY_SHRINK_H */

