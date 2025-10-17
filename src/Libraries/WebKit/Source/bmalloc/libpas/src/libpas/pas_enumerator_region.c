/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 25, 2024.
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
#include "pas_config.h"

#if LIBPAS_ENABLED

#include "pas_enumerator_region.h"

#include "pas_internal_config.h"
#include "pas_page_malloc.h"

void* pas_enumerator_region_allocate(pas_enumerator_region** region_ptr,
                                     size_t size)
{
    pas_enumerator_region* region;
    void* result;

    size = pas_round_up_to_power_of_2(size, PAS_INTERNAL_MIN_ALIGN);

    region = *region_ptr;

    if (!region || region->size - region->offset < size) {
        pas_enumerator_region* new_region;
        size_t allocation_size;
        pas_aligned_allocation_result allocation_result;

        allocation_size = PAS_OFFSETOF(pas_enumerator_region, payload) + size;

        PAS_ASSERT_WITH_DETAIL(pas_is_aligned(allocation_size, PAS_INTERNAL_MIN_ALIGN));

        allocation_result = pas_page_malloc_try_allocate_without_deallocating_padding(
            allocation_size, pas_alignment_create_trivial(), false);

        PAS_ASSERT_WITH_DETAIL(allocation_result.result);
        PAS_ASSERT_WITH_DETAIL(allocation_result.result == allocation_result.left_padding);
        PAS_ASSERT_WITH_DETAIL(!allocation_result.left_padding_size);

        new_region = allocation_result.result;
        new_region->previous = region;
        new_region->size =
            allocation_result.result_size + allocation_result.right_padding_size -
            PAS_OFFSETOF(pas_enumerator_region, payload);
        new_region->offset = 0;

        *region_ptr = new_region;
        region = new_region;
    }

    PAS_ASSERT_WITH_DETAIL(region);
    PAS_ASSERT_WITH_DETAIL(region->size - region->offset >= size);

    result = (char*)region->payload + region->offset;
    region->offset += size;

    return result;
}

void pas_enumerator_region_destroy(pas_enumerator_region* region)
{
    while (region) {
        pas_enumerator_region* previous;

        previous = region->previous;

        pas_page_malloc_deallocate(region, region->size + PAS_OFFSETOF(pas_enumerator_region, payload));

        region = previous;
    }
}

#endif /* LIBPAS_ENABLED */
