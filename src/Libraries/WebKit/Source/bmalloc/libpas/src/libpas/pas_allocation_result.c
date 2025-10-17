/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 2, 2025.
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

#include "pas_allocation_result.h"
#include "pas_page_malloc.h"

pas_allocation_result pas_allocation_result_zero_large_slow(pas_allocation_result result, size_t size)
{
    size_t page_size;

    PAS_PROFILE(ZERO_ALLOCATION_RESULT, result.begin);

    page_size = pas_page_malloc_alignment();
    if (pas_is_aligned(size, page_size) && pas_is_aligned(result.begin, page_size))
        pas_page_malloc_zero_fill((void*)result.begin, size);
    else
        pas_zero_memory((void*)result.begin, size);
    return pas_allocation_result_create_success_with_zero_mode(result.begin, pas_zero_mode_is_all_zero);
}

#endif /* LIBPAS_ENABLED */
