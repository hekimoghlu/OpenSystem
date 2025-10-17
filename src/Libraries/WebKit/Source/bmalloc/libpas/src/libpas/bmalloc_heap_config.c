/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 6, 2025.
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

#include "bmalloc_heap_config.h"

#if PAS_ENABLE_BMALLOC

#include "bmalloc_heap_innards.h"
#include "pas_designated_intrinsic_heap.h"
#include "pas_heap_config_utils_inlines.h"
#include "pas_root.h"

PAS_BEGIN_EXTERN_C;

const pas_heap_config bmalloc_heap_config = BMALLOC_HEAP_CONFIG;

PAS_BASIC_HEAP_CONFIG_DEFINITIONS(
    bmalloc, BMALLOC,
    .allocate_page_should_zero = false,
    .intrinsic_view_cache_capacity = pas_heap_runtime_config_aggressive_view_cache_capacity);

void bmalloc_heap_config_activate(void)
{
#if PAS_OS(DARWIN)
    static const bool register_with_libmalloc = true;
#endif
    
    pas_designated_intrinsic_heap_initialize(&bmalloc_common_primitive_heap.segregated_heap,
                                             &bmalloc_heap_config);

#if PAS_OS(DARWIN)
    if (register_with_libmalloc && !pas_debug_heap_is_enabled(pas_heap_config_kind_bmalloc))
        pas_root_ensure_for_libmalloc_enumeration();
#endif
}

PAS_END_EXTERN_C;

#endif /* PAS_ENABLE_BMALLOC */

#endif /* LIBPAS_ENABLED */
