/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 21, 2023.
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
#ifndef BMALLOC_HEAP_INTERNAL_H
#define BMALLOC_HEAP_INTERNAL_H

#include "pas_platform.h"

#include "bmalloc_heap.h"

#if PAS_ENABLE_BMALLOC

PAS_BEGIN_EXTERN_C;

PAS_CREATE_TRY_ALLOCATE_ARRAY(
    bmalloc_try_iso_allocate_array_impl,
    BMALLOC_HEAP_CONFIG,
    &bmalloc_typed_runtime_config.base,
    &bmalloc_allocator_counts,
    pas_allocation_result_identity);

PAS_CREATE_TRY_ALLOCATE_ARRAY(
    bmalloc_iso_allocate_array_impl,
    BMALLOC_HEAP_CONFIG,
    &bmalloc_typed_runtime_config.base,
    &bmalloc_allocator_counts,
    pas_allocation_result_crash_on_error);

PAS_API void* bmalloc_try_allocate_array_by_size_with_alignment_casual(
    pas_heap_ref* heap_ref, size_t size, size_t alignment, pas_allocation_mode allocation_mode);
PAS_API void* bmalloc_allocate_array_by_size_with_alignment_casual(
    pas_heap_ref* heap_ref, size_t size, size_t alignment, pas_allocation_mode allocation_mode);

static PAS_ALWAYS_INLINE void* bmalloc_try_iso_allocate_inline(pas_heap_ref* heap_ref, pas_allocation_mode allocation_mode)
{
    pas_allocation_result result;
    result = bmalloc_try_iso_allocate_impl_inline_only(heap_ref, allocation_mode);
    if (PAS_LIKELY(result.did_succeed))
        return (void*)result.begin;
    return bmalloc_try_iso_allocate_casual(heap_ref, allocation_mode);
}

static PAS_ALWAYS_INLINE void* bmalloc_iso_allocate_inline(pas_heap_ref* heap_ref, pas_allocation_mode allocation_mode)
{
    pas_allocation_result result;
    result = bmalloc_iso_allocate_impl_inline_only(heap_ref, allocation_mode);
    if (PAS_LIKELY(result.did_succeed))
        return (void*)result.begin;
    return bmalloc_iso_allocate_casual(heap_ref, allocation_mode);
}

static PAS_ALWAYS_INLINE void* bmalloc_try_iso_allocate_array_by_size_inline(
    pas_heap_ref* heap_ref, size_t size, pas_allocation_mode allocation_mode)
{
    pas_allocation_result result;
    result = bmalloc_try_iso_allocate_array_impl_inline_only(heap_ref, size, 1, allocation_mode);
    if (PAS_LIKELY(result.did_succeed))
        return (void*)result.begin;
    return bmalloc_try_allocate_array_by_size_with_alignment_casual(heap_ref, size, 1, allocation_mode);
}

static PAS_ALWAYS_INLINE void* bmalloc_try_iso_allocate_zeroed_array_by_size_inline(
    pas_heap_ref* heap_ref, size_t size, pas_allocation_mode allocation_mode)
{
    return (void*)pas_allocation_result_zero(
        bmalloc_try_iso_allocate_array_impl_by_size(heap_ref, size, 1, allocation_mode),
        size).begin;
}

static PAS_ALWAYS_INLINE void* bmalloc_try_iso_allocate_array_by_size_with_alignment_inline(
    pas_heap_ref* heap_ref, size_t size, size_t alignment, pas_allocation_mode allocation_mode)
{
    pas_allocation_result result;
    result = bmalloc_try_iso_allocate_array_impl_inline_only(heap_ref, size, alignment, allocation_mode);
    if (PAS_LIKELY(result.did_succeed))
        return (void*)result.begin;
    return bmalloc_try_allocate_array_by_size_with_alignment_casual(heap_ref, size, alignment, allocation_mode);
}

static PAS_ALWAYS_INLINE void* bmalloc_try_iso_allocate_array_by_count_inline(
    pas_heap_ref* heap_ref, size_t count, pas_allocation_mode allocation_mode)
{
    return (void*)bmalloc_try_iso_allocate_array_impl_by_count(heap_ref, count, 1, allocation_mode).begin;
}

static PAS_ALWAYS_INLINE void* bmalloc_try_iso_allocate_array_by_count_with_alignment_inline(
    pas_heap_ref* heap_ref, size_t count, size_t alignment, pas_allocation_mode allocation_mode)
{
    return (void*)bmalloc_try_iso_allocate_array_impl_by_count(heap_ref, count, alignment, allocation_mode).begin;
}

static PAS_ALWAYS_INLINE void* bmalloc_iso_allocate_array_by_size_inline(
    pas_heap_ref* heap_ref, size_t size, pas_allocation_mode allocation_mode)
{
    pas_allocation_result result;
    result = bmalloc_iso_allocate_array_impl_inline_only(heap_ref, size, 1, allocation_mode);
    if (PAS_LIKELY(result.did_succeed))
        return (void*)result.begin;
    return bmalloc_allocate_array_by_size_with_alignment_casual(heap_ref, size, 1, allocation_mode);
}

static PAS_ALWAYS_INLINE void* bmalloc_iso_allocate_zeroed_array_by_size_inline(
    pas_heap_ref* heap_ref, size_t size, pas_allocation_mode allocation_mode)
{
    return (void*)pas_allocation_result_zero(
        bmalloc_iso_allocate_array_impl_by_size(heap_ref, size, 1, allocation_mode),
        size).begin;
}

static PAS_ALWAYS_INLINE void* bmalloc_iso_allocate_array_by_size_with_alignment_inline(
    pas_heap_ref* heap_ref, size_t size, size_t alignment, pas_allocation_mode allocation_mode)
{
    pas_allocation_result result;
    result = bmalloc_iso_allocate_array_impl_inline_only(heap_ref, size, alignment, allocation_mode);
    if (PAS_LIKELY(result.did_succeed))
        return (void*)result.begin;
    return bmalloc_allocate_array_by_size_with_alignment_casual(heap_ref, size, alignment, allocation_mode);
}

static PAS_ALWAYS_INLINE void* bmalloc_iso_allocate_array_by_count_inline(
    pas_heap_ref* heap_ref, size_t count, pas_allocation_mode allocation_mode)
{
    return (void*)bmalloc_iso_allocate_array_impl_by_count(heap_ref, count, 1, allocation_mode).begin;
}

static PAS_ALWAYS_INLINE void* bmalloc_iso_allocate_array_by_count_with_alignment_inline(
    pas_heap_ref* heap_ref, size_t count, size_t alignment, pas_allocation_mode allocation_mode)
{
    return (void*)bmalloc_iso_allocate_array_impl_by_count(heap_ref, count, alignment, allocation_mode).begin;
}


static PAS_ALWAYS_INLINE void* bmalloc_try_iso_reallocate_array_by_size_inline(
    pas_heap_ref* heap_ref, void* ptr, size_t size, pas_allocation_mode allocation_mode)
{
    return (void*)pas_try_reallocate_array_by_size(
        ptr,
        heap_ref,
        size,
        allocation_mode,
        BMALLOC_HEAP_CONFIG,
        bmalloc_try_iso_allocate_array_impl_for_realloc,
        &bmalloc_typed_runtime_config.base,
        pas_reallocate_disallow_heap_teleport,
        pas_reallocate_free_if_successful).begin;
}

static PAS_ALWAYS_INLINE void* bmalloc_iso_reallocate_array_by_size_inline(
    pas_heap_ref* heap_ref, void* ptr, size_t size, pas_allocation_mode allocation_mode)
{
    return (void*)pas_try_reallocate_array_by_size(
        ptr,
        heap_ref,
        size,
        allocation_mode,
        BMALLOC_HEAP_CONFIG,
        bmalloc_iso_allocate_array_impl_for_realloc,
        &bmalloc_typed_runtime_config.base,
        pas_reallocate_disallow_heap_teleport,
        pas_reallocate_free_if_successful).begin;
}

static PAS_ALWAYS_INLINE void* bmalloc_try_iso_reallocate_array_by_count_inline(
    pas_heap_ref* heap_ref, void* ptr, size_t count, pas_allocation_mode allocation_mode)
{
    return (void*)pas_try_reallocate_array_by_count(
        ptr,
        heap_ref,
        count,
        allocation_mode,
        BMALLOC_HEAP_CONFIG,
        bmalloc_try_iso_allocate_array_impl_for_realloc,
        &bmalloc_typed_runtime_config.base,
        pas_reallocate_disallow_heap_teleport,
        pas_reallocate_free_if_successful).begin;
}

static PAS_ALWAYS_INLINE void* bmalloc_iso_reallocate_array_by_count_inline(
    pas_heap_ref* heap_ref, void* ptr, size_t count, pas_allocation_mode allocation_mode)
{
    return (void*)pas_try_reallocate_array_by_count(
        ptr,
        heap_ref,
        count,
        allocation_mode,
        BMALLOC_HEAP_CONFIG,
        bmalloc_iso_allocate_array_impl_for_realloc,
        &bmalloc_typed_runtime_config.base,
        pas_reallocate_disallow_heap_teleport,
        pas_reallocate_free_if_successful).begin;
}

PAS_CREATE_TRY_ALLOCATE_PRIMITIVE(
    bmalloc_try_allocate_flex_impl,
    BMALLOC_HEAP_CONFIG,
    &bmalloc_flex_runtime_config.base,
    &bmalloc_allocator_counts,
    pas_allocation_result_identity);

PAS_CREATE_TRY_ALLOCATE_PRIMITIVE(
    bmalloc_allocate_flex_impl,
    BMALLOC_HEAP_CONFIG,
    &bmalloc_flex_runtime_config.base,
    &bmalloc_allocator_counts,
    pas_allocation_result_crash_on_error);

PAS_API void* bmalloc_try_allocate_flex_with_alignment_casual(
    pas_primitive_heap_ref* heap_ref, size_t size, size_t alignment, pas_allocation_mode allocation_mode);
PAS_API void* bmalloc_allocate_flex_with_alignment_casual(
    pas_primitive_heap_ref* heap_ref, size_t size, size_t alignment, pas_allocation_mode allocation_mode);

static PAS_ALWAYS_INLINE void* bmalloc_try_allocate_flex_inline(
    pas_primitive_heap_ref* heap_ref, size_t size, pas_allocation_mode allocation_mode)
{
    pas_allocation_result result;
    result = bmalloc_try_allocate_flex_impl_inline_only(heap_ref, size, 1, allocation_mode);
    if (PAS_LIKELY(result.did_succeed))
        return (void*)result.begin;
    return bmalloc_try_allocate_flex_with_alignment_casual(heap_ref, size, 1, allocation_mode);
}

static PAS_ALWAYS_INLINE void* bmalloc_allocate_flex_inline(
    pas_primitive_heap_ref* heap_ref, size_t size, pas_allocation_mode allocation_mode)
{
    pas_allocation_result result;
    result = bmalloc_allocate_flex_impl_inline_only(heap_ref, size, 1, allocation_mode);
    if (PAS_LIKELY(result.did_succeed))
        return (void*)result.begin;
    return bmalloc_allocate_flex_with_alignment_casual(heap_ref, size, 1, allocation_mode);
}

static PAS_ALWAYS_INLINE void* bmalloc_try_allocate_zeroed_flex_inline(
    pas_primitive_heap_ref* heap_ref, size_t size, pas_allocation_mode allocation_mode)
{
    return (void*)pas_allocation_result_zero(
        bmalloc_try_allocate_flex_impl(heap_ref, size, 1, allocation_mode),
        size).begin;
}

static PAS_ALWAYS_INLINE void* bmalloc_allocate_zeroed_flex_inline(
    pas_primitive_heap_ref* heap_ref, size_t size, pas_allocation_mode allocation_mode)
{
    return (void*)pas_allocation_result_zero(
        bmalloc_allocate_flex_impl(heap_ref, size, 1, allocation_mode),
        size).begin;
}

static PAS_ALWAYS_INLINE void* bmalloc_try_allocate_flex_with_alignment_inline(
    pas_primitive_heap_ref* heap_ref, size_t size, size_t alignment, pas_allocation_mode allocation_mode)
{
    pas_allocation_result result;
    result = bmalloc_try_allocate_flex_impl_inline_only(heap_ref, size, alignment, allocation_mode);
    if (PAS_LIKELY(result.did_succeed))
        return (void*)result.begin;
    return bmalloc_try_allocate_flex_with_alignment_casual(heap_ref, size, alignment, allocation_mode);
}

static PAS_ALWAYS_INLINE void* bmalloc_allocate_flex_with_alignment_inline(
    pas_primitive_heap_ref* heap_ref, size_t size, size_t alignment, pas_allocation_mode allocation_mode)
{
    pas_allocation_result result;
    result = bmalloc_allocate_flex_impl_inline_only(heap_ref, size, alignment, allocation_mode);
    if (PAS_LIKELY(result.did_succeed))
        return (void*)result.begin;
    return bmalloc_allocate_flex_with_alignment_casual(heap_ref, size, alignment, allocation_mode);
}

static PAS_ALWAYS_INLINE void* bmalloc_try_reallocate_flex_inline(
    pas_primitive_heap_ref* heap_ref, void* old_ptr, size_t new_size, pas_allocation_mode allocation_mode)
{
    return (void*)pas_try_reallocate_primitive(
        old_ptr,
        heap_ref,
        new_size,
        allocation_mode,
        BMALLOC_HEAP_CONFIG,
        bmalloc_try_allocate_flex_impl_for_realloc,
        &bmalloc_flex_runtime_config.base,
        pas_reallocate_disallow_heap_teleport,
        pas_reallocate_free_if_successful).begin;
}

static PAS_ALWAYS_INLINE void* bmalloc_reallocate_flex_inline(
    pas_primitive_heap_ref* heap_ref, void* old_ptr, size_t new_size, pas_allocation_mode allocation_mode)
{
    return (void*)pas_try_reallocate_primitive(
        old_ptr,
        heap_ref,
        new_size,
        allocation_mode,
        BMALLOC_HEAP_CONFIG,
        bmalloc_allocate_flex_impl_for_realloc,
        &bmalloc_flex_runtime_config.base,
        pas_reallocate_disallow_heap_teleport,
        pas_reallocate_free_if_successful).begin;
}

PAS_END_EXTERN_C;

#endif /* PAS_ENABLE_BMALLOC */

#endif /* BMALLOC_HEAP_INTERNAL_H */
