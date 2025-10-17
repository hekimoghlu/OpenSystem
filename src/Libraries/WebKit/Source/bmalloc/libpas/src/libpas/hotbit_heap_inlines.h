/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 29, 2025.
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
#ifndef HOTBIT_HEAP_INLINES_H
#define HOTBIT_HEAP_INLINES_H

#include "hotbit_heap.h"
#include "hotbit_heap_config.h"
#include "hotbit_heap_innards.h"
#include "pas_deallocate.h"
#include "pas_try_allocate_intrinsic.h"
#include "pas_try_reallocate.h"

#if PAS_ENABLE_HOTBIT

PAS_BEGIN_EXTERN_C;

PAS_CREATE_TRY_ALLOCATE_INTRINSIC(
    hotbit_try_allocate_impl,
    HOTBIT_HEAP_CONFIG,
    &hotbit_intrinsic_runtime_config.base,
    &hotbit_allocator_counts,
    pas_allocation_result_identity,
    &hotbit_common_primitive_heap,
    &hotbit_common_primitive_heap_support,
    pas_intrinsic_heap_is_designated);

/* Need to create a different set of allocation functions if we want to pass nontrivial alignment,
   since in that case we do not want to use the fancy lookup path. */
PAS_CREATE_TRY_ALLOCATE_INTRINSIC(
    hotbit_try_allocate_with_alignment_impl,
    HOTBIT_HEAP_CONFIG,
    &hotbit_intrinsic_runtime_config.base,
    &hotbit_allocator_counts,
    pas_allocation_result_identity,
    &hotbit_common_primitive_heap,
    &hotbit_common_primitive_heap_support,
    pas_intrinsic_heap_is_not_designated);

static PAS_ALWAYS_INLINE void* hotbit_try_allocate_inline(size_t size, pas_allocation_mode allocation_mode)
{
    return (void*)hotbit_try_allocate_impl(size, 1, allocation_mode).begin;
}

static PAS_ALWAYS_INLINE void*
hotbit_try_allocate_with_alignment_inline(size_t size, size_t alignment, pas_allocation_mode allocation_mode)
{
    return (void*)hotbit_try_allocate_with_alignment_impl(size, alignment, allocation_mode).begin;
}

static PAS_ALWAYS_INLINE void*
hotbit_try_reallocate_inline(void* old_ptr, size_t new_size,
                              pas_reallocate_free_mode free_mode, pas_allocation_mode allocation_mode)
{
    return (void*)pas_try_reallocate_intrinsic(
        old_ptr,
        &hotbit_common_primitive_heap,
        new_size,
        allocation_mode,
        HOTBIT_HEAP_CONFIG,
        hotbit_try_allocate_impl_for_realloc,
        pas_reallocate_allow_heap_teleport,
        free_mode).begin;
}

static PAS_ALWAYS_INLINE void hotbit_deallocate_inline(void* ptr)
{
    pas_deallocate(ptr, HOTBIT_HEAP_CONFIG);
}

PAS_END_EXTERN_C;

#endif /* PAS_ENABLE_HOTBIT */

#endif /* HOTBIT_HEAP_INLINES_H */

