/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 31, 2021.
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
#ifndef BMALLOC_HEAP_INLINES_H
#define BMALLOC_HEAP_INLINES_H

#include "pas_platform.h"

PAS_IGNORE_WARNINGS_BEGIN("missing-field-initializers")

#include "bmalloc_heap.h"
#include "bmalloc_heap_config.h"
#include "bmalloc_heap_innards.h"
#include "pas_deallocate.h"
#include "pas_try_allocate.h"
#include "pas_try_allocate_array.h"
#include "pas_try_allocate_intrinsic.h"
#include "pas_try_allocate_primitive.h"
#include "pas_try_reallocate.h"

#if PAS_ENABLE_BMALLOC

PAS_BEGIN_EXTERN_C;

PAS_CREATE_TRY_ALLOCATE_INTRINSIC(
    bmalloc_try_allocate_impl,
    BMALLOC_HEAP_CONFIG,
    &bmalloc_intrinsic_runtime_config.base,
    &bmalloc_allocator_counts,
    pas_allocation_result_identity,
    &bmalloc_common_primitive_heap,
    &bmalloc_common_primitive_heap_support,
    pas_intrinsic_heap_is_designated);

/* Need to create a different set of allocation functions if we want to pass nontrivial alignment,
   since in that case we do not want to use the fancy lookup path. */
PAS_CREATE_TRY_ALLOCATE_INTRINSIC(
    bmalloc_try_allocate_with_alignment_impl,
    BMALLOC_HEAP_CONFIG,
    &bmalloc_intrinsic_runtime_config.base,
    &bmalloc_allocator_counts,
    pas_allocation_result_identity,
    &bmalloc_common_primitive_heap,
    &bmalloc_common_primitive_heap_support,
    pas_intrinsic_heap_is_not_designated);

PAS_CREATE_TRY_ALLOCATE_INTRINSIC(
    bmalloc_allocate_impl,
    BMALLOC_HEAP_CONFIG,
    &bmalloc_intrinsic_runtime_config.base,
    &bmalloc_allocator_counts,
    pas_allocation_result_crash_on_error,
    &bmalloc_common_primitive_heap,
    &bmalloc_common_primitive_heap_support,
    pas_intrinsic_heap_is_designated);

PAS_CREATE_TRY_ALLOCATE_INTRINSIC(
    bmalloc_allocate_with_alignment_impl,
    BMALLOC_HEAP_CONFIG,
    &bmalloc_intrinsic_runtime_config.base,
    &bmalloc_allocator_counts,
    pas_allocation_result_crash_on_error,
    &bmalloc_common_primitive_heap,
    &bmalloc_common_primitive_heap_support,
    pas_intrinsic_heap_is_not_designated);

PAS_API void* bmalloc_try_allocate_casual(size_t size, pas_allocation_mode allocation_mode);
PAS_API void* bmalloc_allocate_casual(size_t size, pas_allocation_mode allocation_mode);

static PAS_ALWAYS_INLINE void* bmalloc_try_allocate_inline(size_t size, pas_allocation_mode allocation_mode)
{
    pas_allocation_result result;
    result = bmalloc_try_allocate_impl_inline_only(size, 1, allocation_mode);
    if (PAS_LIKELY(result.did_succeed))
        return (void*)result.begin;
    return bmalloc_try_allocate_casual(size, allocation_mode);
}

static PAS_ALWAYS_INLINE void*
bmalloc_try_allocate_with_alignment_inline(size_t size, size_t alignment, pas_allocation_mode allocation_mode)
{
    return (void*)bmalloc_try_allocate_with_alignment_impl(size, alignment, allocation_mode).begin;
}

static PAS_ALWAYS_INLINE void* bmalloc_try_allocate_zeroed_inline(size_t size, pas_allocation_mode allocation_mode)
{
    return (void*)pas_allocation_result_zero(
        bmalloc_try_allocate_impl(size, 1, allocation_mode),
        size).begin;
}

static PAS_ALWAYS_INLINE void* bmalloc_allocate_inline(size_t size, pas_allocation_mode allocation_mode)
{
    pas_allocation_result result;
    result = bmalloc_allocate_impl_inline_only(size, 1, allocation_mode);
    if (PAS_LIKELY(result.did_succeed))
        return (void*)result.begin;
    return bmalloc_allocate_casual(size, allocation_mode);
}

static PAS_ALWAYS_INLINE void*
bmalloc_allocate_with_alignment_inline(size_t size, size_t alignment, pas_allocation_mode allocation_mode)
{
    return (void*)bmalloc_allocate_with_alignment_impl(size, alignment, allocation_mode).begin;
}

static PAS_ALWAYS_INLINE void* bmalloc_allocate_zeroed_inline(size_t size, pas_allocation_mode allocation_mode)
{
    return (void*)pas_allocation_result_zero(
        bmalloc_allocate_impl(size, 1, allocation_mode),
        size).begin;
}

static PAS_ALWAYS_INLINE void*
bmalloc_try_reallocate_inline(void* old_ptr, size_t new_size,
                              pas_allocation_mode allocation_mode,
                              pas_reallocate_free_mode free_mode)
{
    return (void*)pas_try_reallocate_intrinsic(
        old_ptr,
        &bmalloc_common_primitive_heap,
        new_size,
        allocation_mode,
        BMALLOC_HEAP_CONFIG,
        bmalloc_try_allocate_impl_for_realloc,
        pas_reallocate_allow_heap_teleport,
        free_mode).begin;
}

static PAS_ALWAYS_INLINE void*
bmalloc_reallocate_inline(void* old_ptr, size_t new_size,
                          pas_allocation_mode allocation_mode,
                          pas_reallocate_free_mode free_mode)
{
    return (void*)pas_try_reallocate_intrinsic(
        old_ptr,
        &bmalloc_common_primitive_heap,
        new_size,
        allocation_mode,
        BMALLOC_HEAP_CONFIG,
        bmalloc_allocate_impl_for_realloc,
        pas_reallocate_allow_heap_teleport,
        free_mode).begin;
}

PAS_CREATE_TRY_ALLOCATE(
    bmalloc_try_iso_allocate_impl,
    BMALLOC_HEAP_CONFIG,
    &bmalloc_typed_runtime_config.base,
    &bmalloc_allocator_counts,
    pas_allocation_result_identity);

PAS_CREATE_TRY_ALLOCATE(
    bmalloc_iso_allocate_impl,
    BMALLOC_HEAP_CONFIG,
    &bmalloc_typed_runtime_config.base,
    &bmalloc_allocator_counts,
    pas_allocation_result_crash_on_error);

PAS_API void* bmalloc_try_iso_allocate_casual(pas_heap_ref* heap_ref, pas_allocation_mode allocation_mode);
PAS_API void* bmalloc_iso_allocate_casual(pas_heap_ref* heap_ref, pas_allocation_mode allocation_mode);


PAS_CREATE_TRY_ALLOCATE_PRIMITIVE(
    bmalloc_try_allocate_auxiliary_impl,
    BMALLOC_HEAP_CONFIG,
    &bmalloc_primitive_runtime_config.base,
    &bmalloc_allocator_counts,
    pas_allocation_result_identity);

PAS_CREATE_TRY_ALLOCATE_PRIMITIVE(
    bmalloc_allocate_auxiliary_impl,
    BMALLOC_HEAP_CONFIG,
    &bmalloc_primitive_runtime_config.base,
    &bmalloc_allocator_counts,
    pas_allocation_result_crash_on_error);

PAS_API void* bmalloc_try_allocate_auxiliary_with_alignment_casual(
    pas_primitive_heap_ref* heap_ref, size_t size, size_t alignment, pas_allocation_mode allocation_mode);
PAS_API void* bmalloc_allocate_auxiliary_with_alignment_casual(
    pas_primitive_heap_ref* heap_ref, size_t size, size_t alignment, pas_allocation_mode allocation_mode);

static PAS_ALWAYS_INLINE void* bmalloc_try_allocate_auxiliary_inline(pas_primitive_heap_ref* heap_ref,
                                                                     size_t size,
                                                                     pas_allocation_mode allocation_mode)
{
    pas_allocation_result result;
    result = bmalloc_try_allocate_auxiliary_impl_inline_only(heap_ref, size, 1, allocation_mode);
    if (PAS_LIKELY(result.did_succeed))
        return (void*)result.begin;
    return bmalloc_try_allocate_auxiliary_with_alignment_casual(heap_ref, size, 1, allocation_mode);
}

static PAS_ALWAYS_INLINE void* bmalloc_allocate_auxiliary_inline(pas_primitive_heap_ref* heap_ref,
                                                                 size_t size,
                                                                 pas_allocation_mode allocation_mode)
{
    pas_allocation_result result;
    result = bmalloc_allocate_auxiliary_impl_inline_only(heap_ref, size, 1, allocation_mode);
    if (PAS_LIKELY(result.did_succeed))
        return (void*)result.begin;
    return bmalloc_allocate_auxiliary_with_alignment_casual(heap_ref, size, 1, allocation_mode);
}

static PAS_ALWAYS_INLINE void* bmalloc_try_allocate_auxiliary_zeroed_inline(
    pas_primitive_heap_ref* heap_ref,
    size_t size,
    pas_allocation_mode allocation_mode)
{
    return (void*)pas_allocation_result_zero(
        bmalloc_try_allocate_auxiliary_impl(heap_ref, size, 1, allocation_mode),
        size).begin;
}

static PAS_ALWAYS_INLINE void* bmalloc_allocate_auxiliary_zeroed_inline(pas_primitive_heap_ref* heap_ref,
                                                                        size_t size,
                                                                        pas_allocation_mode allocation_mode)
{
    return (void*)pas_allocation_result_zero(
        bmalloc_allocate_auxiliary_impl(heap_ref, size, 1, allocation_mode),
        size).begin;
}

static PAS_ALWAYS_INLINE void*
bmalloc_try_allocate_auxiliary_with_alignment_inline(pas_primitive_heap_ref* heap_ref,
                                                     size_t size,
                                                     size_t alignment,
                                                     pas_allocation_mode allocation_mode)
{
    pas_allocation_result result;
    result = bmalloc_try_allocate_auxiliary_impl_inline_only(heap_ref, size, alignment, allocation_mode);
    if (PAS_LIKELY(result.did_succeed))
        return (void*)result.begin;
    return bmalloc_try_allocate_auxiliary_with_alignment_casual(heap_ref, size, alignment, allocation_mode);
}

static PAS_ALWAYS_INLINE void*
bmalloc_allocate_auxiliary_with_alignment_inline(pas_primitive_heap_ref* heap_ref,
                                                 size_t size,
                                                 size_t alignment,
                                                 pas_allocation_mode allocation_mode)
{
    pas_allocation_result result;
    result = bmalloc_allocate_auxiliary_impl_inline_only(heap_ref, size, alignment, allocation_mode);
    if (PAS_LIKELY(result.did_succeed))
        return (void*)result.begin;
    return bmalloc_allocate_auxiliary_with_alignment_casual(heap_ref, size, alignment, allocation_mode);
}

static PAS_ALWAYS_INLINE void* bmalloc_try_reallocate_auxiliary_inline(
    void* old_ptr,
    pas_primitive_heap_ref* heap_ref,
    size_t new_size,
    pas_allocation_mode allocation_mode,
    pas_reallocate_free_mode free_mode)
{
    return (void*)pas_try_reallocate_primitive(
        old_ptr,
        heap_ref,
        new_size,
        allocation_mode,
        BMALLOC_HEAP_CONFIG,
        bmalloc_try_allocate_auxiliary_impl_for_realloc,
        &bmalloc_primitive_runtime_config.base,
        pas_reallocate_allow_heap_teleport,
        free_mode).begin;
}

static PAS_ALWAYS_INLINE void* bmalloc_reallocate_auxiliary_inline(void* old_ptr,
                                                                   pas_primitive_heap_ref* heap_ref,
                                                                   size_t new_size,
                                                                   pas_allocation_mode allocation_mode,
                                                                   pas_reallocate_free_mode free_mode)
{
    return (void*)pas_try_reallocate_primitive(
        old_ptr,
        heap_ref,
        new_size,
        allocation_mode,
        BMALLOC_HEAP_CONFIG,
        bmalloc_allocate_auxiliary_impl_for_realloc,
        &bmalloc_primitive_runtime_config.base,
        pas_reallocate_allow_heap_teleport,
        free_mode).begin;
}

static PAS_ALWAYS_INLINE void bmalloc_deallocate_inline(void* ptr)
{
    pas_deallocate(ptr, BMALLOC_HEAP_CONFIG);
}

PAS_END_EXTERN_C;

#endif /* PAS_ENABLE_BMALLOC */

PAS_IGNORE_WARNINGS_END

#endif /* BMALLOC_HEAP_INLINES_H */

