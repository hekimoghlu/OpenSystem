/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 18, 2023.
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

#include "pas_heap_runtime_config.h"

#include "pas_designated_intrinsic_heap_inlines.h"

uint8_t pas_heap_runtime_config_view_cache_capacity_for_object_size(
    pas_heap_runtime_config* config,
    size_t object_size,
    const pas_segregated_page_config* page_config)
{
    size_t result;

    result = config->view_cache_capacity_for_object_size(object_size, page_config);

    PAS_ASSERT((uint8_t)result == result);
    return (uint8_t)result;
}

size_t pas_heap_runtime_config_zero_view_cache_capacity(
    size_t object_size, const pas_segregated_page_config* page_config)
{
    PAS_UNUSED_PARAM(object_size);
    PAS_UNUSED_PARAM(page_config);
    return 0;
}

size_t pas_heap_runtime_config_aggressive_view_cache_capacity(
    size_t object_size, const pas_segregated_page_config* page_config)
{
    static const size_t cache_size = 1638400;

    PAS_UNUSED_PARAM(object_size);

    PAS_ASSERT(page_config->base.page_size < cache_size);

    return cache_size / page_config->base.page_size;
}

#endif /* LIBPAS_ENABLED */

