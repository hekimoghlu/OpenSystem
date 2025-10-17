/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 4, 2021.
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
#ifndef ISO_HEAP_INLINES_H
#define ISO_HEAP_INLINES_H

#include "iso_heap.h"
#include "iso_heap_config.h"
#include "iso_heap_innards.h"
#include "pas_deallocate.h"
#include "pas_get_allocation_size.h"
#include "pas_get_heap.h"
#include "pas_has_object.h"
#include "pas_try_allocate.h"
#include "pas_try_allocate_array.h"
#include "pas_try_allocate_intrinsic.h"
#include "pas_try_allocate_primitive.h"
#include "pas_try_reallocate.h"

#if PAS_ENABLE_ISO

PAS_BEGIN_EXTERN_C;

PAS_CREATE_TRY_ALLOCATE_INTRINSIC(
    iso_try_allocate_common_primitive_impl,
    ISO_HEAP_CONFIG,
    &iso_intrinsic_runtime_config.base,
    &iso_allocator_counts,
    pas_allocation_result_set_errno,
    &iso_common_primitive_heap,
    &iso_common_primitive_heap_support,
    pas_intrinsic_heap_is_not_designated);

/* Need to create a different set of allocation functions if we want to pass nontrivial alignment,
   since in that case we do not want to use the fancy lookup path. */
PAS_CREATE_TRY_ALLOCATE_INTRINSIC(
    iso_try_allocate_common_primitive_with_alignment_impl,
    ISO_HEAP_CONFIG,
    &iso_intrinsic_runtime_config.base,
    &iso_allocator_counts,
    pas_allocation_result_set_errno,
    &iso_common_primitive_heap,
    &iso_common_primitive_heap_support,
    pas_intrinsic_heap_is_not_designated);

PAS_CREATE_TRY_ALLOCATE_INTRINSIC(
    iso_allocate_common_primitive_impl,
    ISO_HEAP_CONFIG,
    &iso_intrinsic_runtime_config.base,
    &iso_allocator_counts,
    pas_allocation_result_crash_on_error,
    &iso_common_primitive_heap,
    &iso_common_primitive_heap_support,
    pas_intrinsic_heap_is_not_designated);

PAS_CREATE_TRY_ALLOCATE_INTRINSIC(
    iso_allocate_common_primitive_with_alignment_impl,
    ISO_HEAP_CONFIG,
    &iso_intrinsic_runtime_config.base,
    &iso_allocator_counts,
    pas_allocation_result_crash_on_error,
    &iso_common_primitive_heap,
    &iso_common_primitive_heap_support,
    pas_intrinsic_heap_is_not_designated);

static PAS_ALWAYS_INLINE void* iso_try_allocate_common_primitive_inline(size_t size, pas_allocation_mode allocation_mode)
{
    return (void*)iso_try_allocate_common_primitive_impl(size, 1, allocation_mode).begin;
}

static PAS_ALWAYS_INLINE void*
iso_try_allocate_common_primitive_with_alignment_inline(size_t size, size_t alignment, pas_allocation_mode allocation_mode)
{
    return (void*)iso_try_allocate_common_primitive_with_alignment_impl(size, alignment, allocation_mode).begin;
}

static PAS_ALWAYS_INLINE void* iso_try_allocate_common_primitive_zeroed_inline(size_t size, pas_allocation_mode allocation_mode)
{
    return (void*)pas_allocation_result_zero(
        iso_try_allocate_common_primitive_impl(size, 1, allocation_mode),
        size).begin;
}

static PAS_ALWAYS_INLINE void* iso_allocate_common_primitive_inline(size_t size, pas_allocation_mode allocation_mode)
{
    return (void*)iso_allocate_common_primitive_impl(size, 1, allocation_mode).begin;
}

static PAS_ALWAYS_INLINE void*
iso_allocate_common_primitive_with_alignment_inline(size_t size, size_t alignment, pas_allocation_mode allocation_mode)
{
    return (void*)iso_allocate_common_primitive_with_alignment_impl(size, alignment, allocation_mode).begin;
}

static PAS_ALWAYS_INLINE void* iso_allocate_common_primitive_zeroed_inline(size_t size, pas_allocation_mode allocation_mode)
{
    return (void*)pas_allocation_result_zero(
        iso_allocate_common_primitive_impl(size, 1, allocation_mode),
        size).begin;
}

static PAS_ALWAYS_INLINE void*
iso_try_reallocate_common_primitive_inline(void* old_ptr, size_t new_size,
                                           pas_reallocate_free_mode free_mode, pas_allocation_mode allocation_mode)
{
    return (void*)pas_try_reallocate_intrinsic(
        old_ptr,
        &iso_common_primitive_heap,
        new_size,
        allocation_mode,
        ISO_HEAP_CONFIG,
        iso_try_allocate_common_primitive_impl_for_realloc,
        pas_reallocate_allow_heap_teleport,
        free_mode).begin;
}

static PAS_ALWAYS_INLINE void*
iso_reallocate_common_primitive_inline(void* old_ptr, size_t new_size,
                                       pas_reallocate_free_mode free_mode, pas_allocation_mode allocation_mode)
{
    return (void*)pas_try_reallocate_intrinsic(
        old_ptr,
        &iso_common_primitive_heap,
        new_size,
        allocation_mode,
        ISO_HEAP_CONFIG,
        iso_allocate_common_primitive_impl_for_realloc,
        pas_reallocate_allow_heap_teleport,
        free_mode).begin;
}

PAS_CREATE_TRY_ALLOCATE(
    iso_try_allocate_impl,
    ISO_HEAP_CONFIG,
    &iso_typed_runtime_config.base,
    &iso_allocator_counts,
    pas_allocation_result_set_errno);

static PAS_ALWAYS_INLINE void* iso_try_allocate_inline(pas_heap_ref* heap_ref, pas_allocation_mode allocation_mode)
{
    return (void*)iso_try_allocate_impl(heap_ref, allocation_mode).begin;
}

PAS_CREATE_TRY_ALLOCATE(
    iso_allocate_impl,
    ISO_HEAP_CONFIG,
    &iso_typed_runtime_config.base,
    &iso_allocator_counts,
    pas_allocation_result_crash_on_error);

static PAS_ALWAYS_INLINE void* iso_allocate_inline(pas_heap_ref* heap_ref, pas_allocation_mode allocation_mode)
{
    return (void*)iso_allocate_impl(heap_ref, allocation_mode).begin;
}

PAS_CREATE_TRY_ALLOCATE_ARRAY(
    iso_try_allocate_array_impl,
    ISO_HEAP_CONFIG,
    &iso_typed_runtime_config.base,
    &iso_allocator_counts,
    pas_allocation_result_set_errno);

static PAS_ALWAYS_INLINE void*
iso_try_allocate_array_by_count_inline(pas_heap_ref* heap_ref, size_t count, size_t alignment, pas_allocation_mode allocation_mode)
{
    return (void*)iso_try_allocate_array_impl_by_count(heap_ref, count, alignment, allocation_mode).begin;
}

PAS_CREATE_TRY_ALLOCATE_ARRAY(
    iso_allocate_array_impl,
    ISO_HEAP_CONFIG,
    &iso_typed_runtime_config.base,
    &iso_allocator_counts,
    pas_allocation_result_crash_on_error);

static PAS_ALWAYS_INLINE void*
iso_allocate_array_by_count_inline(pas_heap_ref* heap_ref, size_t count, size_t alignment, pas_allocation_mode allocation_mode)
{
    return (void*)iso_allocate_array_impl_by_count(heap_ref, count, alignment, allocation_mode).begin;
}

static PAS_ALWAYS_INLINE void* iso_try_allocate_array_by_count_zeroed_inline(
    pas_heap_ref* heap_ref, size_t count, size_t alignment, pas_allocation_mode allocation_mode)
{
    size_t size;

    if (__builtin_mul_overflow(count, ISO_HEAP_CONFIG.get_type_size(heap_ref->type), &size)) {
        errno = ENOMEM;
        return NULL;
    }
    
    return (void*)pas_allocation_result_zero(
        iso_try_allocate_array_impl_by_size(heap_ref, size, alignment, allocation_mode), size).begin;
}

static PAS_ALWAYS_INLINE void* iso_allocate_array_by_count_zeroed_inline(
    pas_heap_ref* heap_ref, size_t count, size_t alignment, pas_allocation_mode allocation_mode)
{
    size_t size;
    bool did_overflow;

    did_overflow = __builtin_mul_overflow(count, ISO_HEAP_CONFIG.get_type_size(heap_ref->type), &size);
    PAS_ASSERT(!did_overflow);
    
    return (void*)pas_allocation_result_zero(
        iso_allocate_array_impl_by_size(heap_ref, size, alignment, allocation_mode), size).begin;
}

static PAS_ALWAYS_INLINE void* iso_try_reallocate_array_by_count_inline(void* old_ptr, pas_heap_ref* heap_ref,
                                                                        size_t new_count,
                                                                        pas_reallocate_free_mode free_mode,
                                                                        pas_allocation_mode allocation_mode)
{
    return (void*)pas_try_reallocate_array_by_count(
        old_ptr,
        heap_ref,
        new_count,
        allocation_mode,
        ISO_HEAP_CONFIG,
        iso_try_allocate_array_impl_for_realloc,
        &iso_typed_runtime_config.base,
        pas_reallocate_allow_heap_teleport,
        free_mode).begin;
}

static PAS_ALWAYS_INLINE void* iso_reallocate_array_by_count_inline(void* old_ptr, pas_heap_ref* heap_ref,
                                                                    size_t new_count,
                                                                    pas_reallocate_free_mode free_mode,
                                                                    pas_allocation_mode allocation_mode)
{
    return (void*)pas_try_reallocate_array_by_count(
        old_ptr,
        heap_ref,
        new_count,
        allocation_mode,
        ISO_HEAP_CONFIG,
        iso_allocate_array_impl_for_realloc,
        &iso_typed_runtime_config.base,
        pas_reallocate_allow_heap_teleport,
        free_mode).begin;
}

PAS_CREATE_TRY_ALLOCATE_PRIMITIVE(
    iso_try_allocate_primitive_impl,
    ISO_HEAP_CONFIG,
    &iso_primitive_runtime_config.base,
    &iso_allocator_counts,
    pas_allocation_result_set_errno);

static PAS_ALWAYS_INLINE void* iso_try_allocate_primitive_inline(pas_primitive_heap_ref* heap_ref,
                                                                 size_t size, pas_allocation_mode allocation_mode)
{
    return (void*)iso_try_allocate_primitive_impl(heap_ref, size, 1, allocation_mode).begin;
}

PAS_CREATE_TRY_ALLOCATE_PRIMITIVE(
    iso_allocate_primitive_impl,
    ISO_HEAP_CONFIG,
    &iso_primitive_runtime_config.base,
    &iso_allocator_counts,
    pas_allocation_result_crash_on_error);

static PAS_ALWAYS_INLINE void* iso_allocate_primitive_inline(pas_primitive_heap_ref* heap_ref,
                                                             size_t size, pas_allocation_mode allocation_mode)
{
    return (void*)iso_allocate_primitive_impl(heap_ref, size, 1, allocation_mode).begin;
}

static PAS_ALWAYS_INLINE void* iso_try_allocate_primitive_zeroed_inline(pas_primitive_heap_ref* heap_ref,
                                                                        size_t size, pas_allocation_mode allocation_mode)
{
    return (void*)pas_allocation_result_zero(
        iso_try_allocate_primitive_impl(heap_ref, size, 1, allocation_mode),
        size).begin;
}

static PAS_ALWAYS_INLINE void* iso_allocate_primitive_zeroed_inline(pas_primitive_heap_ref* heap_ref,
                                                                    size_t size, pas_allocation_mode allocation_mode)
{
    return (void*)pas_allocation_result_zero(
        iso_allocate_primitive_impl(heap_ref, size, 1, allocation_mode),
        size).begin;
}

static PAS_ALWAYS_INLINE void*
iso_try_allocate_primitive_with_alignment_inline(pas_primitive_heap_ref* heap_ref,
                                                 size_t size,
                                                 size_t alignment,
                                                 pas_allocation_mode allocation_mode)
{
    return (void*)iso_try_allocate_primitive_impl(heap_ref, size, alignment, allocation_mode).begin;
}

static PAS_ALWAYS_INLINE void*
iso_allocate_primitive_with_alignment_inline(pas_primitive_heap_ref* heap_ref,
                                             size_t size,
                                             size_t alignment,
                                             pas_allocation_mode allocation_mode)
{
    return (void*)iso_allocate_primitive_impl(heap_ref, size, alignment, allocation_mode).begin;
}

static PAS_ALWAYS_INLINE void* iso_try_reallocate_primitive_inline(void* old_ptr,
                                                                   pas_primitive_heap_ref* heap_ref,
                                                                   size_t new_size,
                                                                   pas_reallocate_free_mode free_mode,
                                                                   pas_allocation_mode allocation_mode)
{
    return (void*)pas_try_reallocate_primitive(
        old_ptr,
        heap_ref,
        new_size,
        allocation_mode,
        ISO_HEAP_CONFIG,
        iso_try_allocate_primitive_impl_for_realloc,
        &iso_primitive_runtime_config.base,
        pas_reallocate_allow_heap_teleport,
        free_mode).begin;
}

static PAS_ALWAYS_INLINE void* iso_reallocate_primitive_inline(void* old_ptr,
                                                               pas_primitive_heap_ref* heap_ref,
                                                               size_t new_size,
                                                               pas_reallocate_free_mode free_mode,
                                                               pas_allocation_mode allocation_mode)
{
    return (void*)pas_try_reallocate_primitive(
        old_ptr,
        heap_ref,
        new_size,
        allocation_mode,
        ISO_HEAP_CONFIG,
        iso_allocate_primitive_impl_for_realloc,
        &iso_primitive_runtime_config.base,
        pas_reallocate_allow_heap_teleport,
        free_mode).begin;
}

PAS_CREATE_TRY_ALLOCATE_PRIMITIVE(
    iso_try_allocate_for_flex_impl,
    ISO_HEAP_CONFIG,
    &iso_flex_runtime_config.base,
    &iso_allocator_counts,
    pas_allocation_result_set_errno);

static PAS_ALWAYS_INLINE void* iso_try_allocate_for_flex_inline(const void* cls, size_t size, pas_allocation_mode allocation_mode)
{
    return (void*)pas_allocation_result_zero(
        iso_try_allocate_for_flex_impl(
            pas_dynamic_primitive_heap_map_find(
                &iso_flex_dynamic_heap_map, cls, size),
            size, 1, allocation_mode),
        size).begin;
}

static PAS_ALWAYS_INLINE bool iso_has_object_inline(void* ptr)
{
    return pas_has_object(ptr, ISO_HEAP_CONFIG);
}

static PAS_ALWAYS_INLINE size_t iso_get_allocation_size_inline(void* ptr)
{
    return pas_get_allocation_size(ptr, ISO_HEAP_CONFIG);
}

static PAS_ALWAYS_INLINE pas_heap* iso_get_heap_inline(void* ptr)
{
    return pas_get_heap(ptr, ISO_HEAP_CONFIG);
}

static PAS_ALWAYS_INLINE void iso_deallocate_inline(void* ptr)
{
    pas_deallocate(ptr, ISO_HEAP_CONFIG);
}

PAS_END_EXTERN_C;

#endif /* PAS_ENABLE_ISO */

#endif /* ISO_HEAP_INLINES_H */

