/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 1, 2024.
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

#include "pas_ensure_heap_forced_into_reserved_memory.h"

#include "pas_create_basic_heap_page_caches_with_reserved_memory.h"
#include "pas_ensure_heap_with_page_caches.h"

/* Warning: This creates caches that allow type confusion. Only use this for primitive heaps! */
pas_heap* pas_ensure_heap_forced_into_reserved_memory(
    pas_heap_ref* heap_ref,
    pas_heap_ref_kind heap_ref_kind,
    const pas_heap_config* config,
    pas_heap_runtime_config* template_runtime_config,
    uintptr_t begin,
    uintptr_t end)
{
    return pas_ensure_heap_with_page_caches(
        heap_ref, heap_ref_kind, config, template_runtime_config,
        pas_create_basic_heap_page_caches_with_reserved_memory(
            (pas_basic_heap_runtime_config*)template_runtime_config, begin, end));
}

#endif /* LIBPAS_ENABLED */
