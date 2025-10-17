/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 21, 2024.
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

#include "minalign32_heap_config.h"

#if PAS_ENABLE_MINALIGN32

#include "minalign32_heap.h"
#include "pas_designated_intrinsic_heap.h"
#include "pas_heap_config_utils_inlines.h"

const pas_heap_config minalign32_heap_config = MINALIGN32_HEAP_CONFIG;

PAS_BASIC_HEAP_CONFIG_DEFINITIONS(
    minalign32, MINALIGN32,
    .allocate_page_should_zero = false,
    .intrinsic_view_cache_capacity = pas_heap_runtime_config_aggressive_view_cache_capacity);

void minalign32_heap_config_activate(void)
{
    pas_designated_intrinsic_heap_initialize(&minalign32_common_primitive_heap.segregated_heap,
                                             &minalign32_heap_config);
}

#endif /* PAS_ENABLE_MINALIGN32 */

#endif /* LIBPAS_ENABLED */
