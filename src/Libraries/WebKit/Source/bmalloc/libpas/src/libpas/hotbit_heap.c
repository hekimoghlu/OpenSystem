/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 24, 2023.
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

#include "hotbit_heap.h"

#if PAS_ENABLE_HOTBIT

#include "hotbit_heap_inlines.h"
#include "pas_deallocate.h"

pas_intrinsic_heap_support hotbit_common_primitive_heap_support =
    PAS_INTRINSIC_HEAP_SUPPORT_INITIALIZER;

pas_heap hotbit_common_primitive_heap =
    PAS_INTRINSIC_HEAP_INITIALIZER(
        &hotbit_common_primitive_heap,
        PAS_SIMPLE_TYPE_CREATE(1, 1),
        hotbit_common_primitive_heap_support,
        HOTBIT_HEAP_CONFIG,
        &hotbit_intrinsic_runtime_config.base);

pas_allocator_counts hotbit_allocator_counts;

void* hotbit_try_allocate(size_t size, pas_allocation_mode allocation_mode)
{
    return hotbit_try_allocate_inline(size, allocation_mode);
}

void* hotbit_try_allocate_with_alignment(size_t size, size_t alignment, pas_allocation_mode allocation_mode)
{
    return hotbit_try_allocate_with_alignment_inline(size, alignment, allocation_mode);
}

void* hotbit_try_reallocate(void* old_ptr, size_t new_size,
                             pas_reallocate_free_mode free_mode, pas_allocation_mode allocation_mode)
{
    return hotbit_try_reallocate_inline(old_ptr, new_size, free_mode, allocation_mode);
}

void hotbit_deallocate(void* ptr)
{
    hotbit_deallocate_inline(ptr);
}

#endif /* PAS_ENABLE_HOTBIT */

#endif /* LIBPAS_ENABLED */
