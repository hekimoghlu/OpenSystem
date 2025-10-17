/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 31, 2024.
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

#include "pas_enumerable_page_malloc.h"

#include "pas_page_malloc.h"

pas_enumerable_range_list pas_enumerable_page_malloc_page_list;

pas_aligned_allocation_result
pas_enumerable_page_malloc_try_allocate_without_deallocating_padding(
    size_t size, pas_alignment alignment, bool may_contain_small_or_medium)
{
    pas_aligned_allocation_result result;

    result = pas_page_malloc_try_allocate_without_deallocating_padding(size, alignment, may_contain_small_or_medium);

    if (result.result) {
        pas_enumerable_range_list_append(
            &pas_enumerable_page_malloc_page_list,
            pas_range_create(
                (uintptr_t)result.left_padding,
                (uintptr_t)result.right_padding + result.right_padding_size));
    }

    return result;
}

#endif /* LIBPAS_ENABLED */
