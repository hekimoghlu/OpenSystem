/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 13, 2022.
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

#include "iso_heap.h"

#if PAS_ENABLE_ISO

#include "iso_heap_inlines.h"
#include "pas_deallocate.h"
#include "pas_ensure_heap_forced_into_reserved_memory.h"
#include "pas_try_allocate.h"
#include "pas_try_allocate_array.h"
#include "pas_try_allocate_intrinsic.h"
#include "pas_try_allocate_primitive.h"
#include "pas_try_reallocate.h"

pas_intrinsic_heap_support iso_common_primitive_heap_support =
    PAS_INTRINSIC_HEAP_SUPPORT_INITIALIZER;

pas_heap iso_common_primitive_heap =
    PAS_INTRINSIC_HEAP_INITIALIZER(
        &iso_common_primitive_heap,
        PAS_SIMPLE_TYPE_CREATE(1, 1),
        iso_common_primitive_heap_support,
        ISO_HEAP_CONFIG,
        &iso_intrinsic_runtime_config.base);

pas_allocator_counts iso_allocator_counts;

pas_dynamic_primitive_heap_map iso_primitive_dynamic_heap_map =
    PAS_DYNAMIC_PRIMITIVE_HEAP_MAP_INITIALIZER(iso_primitive_heap_ref_construct);
pas_dynamic_primitive_heap_map iso_flex_dynamic_heap_map =
    PAS_DYNAMIC_PRIMITIVE_HEAP_MAP_INITIALIZER(iso_primitive_heap_ref_construct);

void* iso_try_allocate_common_primitive(size_t size, pas_allocation_mode allocation_mode)
{
    return iso_try_allocate_common_primitive_inline(size, allocation_mode);
}

void* iso_try_allocate_common_primitive_with_alignment(size_t size, size_t alignment, pas_allocation_mode allocation_mode)
{
    return iso_try_allocate_common_primitive_with_alignment_inline(size, alignment, allocation_mode);
}

void* iso_try_allocate_common_primitive_zeroed(size_t size, pas_allocation_mode allocation_mode)
{
    return iso_try_allocate_common_primitive_zeroed_inline(size, allocation_mode);
}

void* iso_allocate_common_primitive(size_t size, pas_allocation_mode allocation_mode)
{
    return iso_allocate_common_primitive_inline(size, allocation_mode);
}

void* iso_allocate_common_primitive_with_alignment(size_t size, size_t alignment, pas_allocation_mode allocation_mode)
{
    return iso_allocate_common_primitive_with_alignment_inline(size, alignment, allocation_mode);
}

void* iso_allocate_common_primitive_zeroed(size_t size, pas_allocation_mode allocation_mode)
{
    return iso_allocate_common_primitive_zeroed_inline(size, allocation_mode);
}

void* iso_try_reallocate_common_primitive(void* old_ptr, size_t new_size,
                                          pas_reallocate_free_mode free_mode,
                                          pas_allocation_mode allocation_mode)
{
    return iso_try_reallocate_common_primitive_inline(old_ptr, new_size, free_mode, allocation_mode);
}

void* iso_reallocate_common_primitive(void* old_ptr, size_t new_size,
                                      pas_reallocate_free_mode free_mode,
                                      pas_allocation_mode allocation_mode)
{
    return iso_reallocate_common_primitive_inline(old_ptr, new_size, free_mode, allocation_mode);
}

void* iso_try_allocate_dynamic_primitive(const void* key, size_t size, pas_allocation_mode allocation_mode)
{
    return iso_try_allocate_primitive(
        pas_dynamic_primitive_heap_map_find(
            &iso_primitive_dynamic_heap_map, key, size),
        size, allocation_mode);
}

void* iso_try_allocate_dynamic_primitive_with_alignment(const void* key,
                                                        size_t size,
                                                        size_t alignment,
                                                        pas_allocation_mode allocation_mode)
{
    return iso_try_allocate_primitive_with_alignment(
        pas_dynamic_primitive_heap_map_find(
            &iso_primitive_dynamic_heap_map, key, size),
        size,
        alignment,
        allocation_mode);
}

void* iso_try_allocate_dynamic_primitive_zeroed(const void* key,
                                                size_t size,
                                                pas_allocation_mode allocation_mode)
{
    return iso_try_allocate_primitive_zeroed(
        pas_dynamic_primitive_heap_map_find(
            &iso_primitive_dynamic_heap_map, key, size),
        size,
        allocation_mode);
}

void* iso_try_reallocate_dynamic_primitive(void* old_ptr,
                                           const void* key,
                                           size_t new_size,
                                           pas_reallocate_free_mode free_mode,
                                           pas_allocation_mode allocation_mode)
{
    return iso_try_reallocate_primitive(
        old_ptr,
        pas_dynamic_primitive_heap_map_find(
            &iso_primitive_dynamic_heap_map, key, new_size),
        new_size,
        free_mode,
        allocation_mode);
}

void iso_heap_ref_construct(pas_heap_ref* heap_ref,
                            pas_simple_type type)
{
    heap_ref->type = (const pas_heap_type*)type;
    heap_ref->heap = NULL;
    heap_ref->allocator_index = 0;
}

void* iso_try_allocate(pas_heap_ref* heap_ref, pas_allocation_mode allocation_mode)
{
    return iso_try_allocate_inline(heap_ref, allocation_mode);
}

void* iso_allocate(pas_heap_ref* heap_ref, pas_allocation_mode allocation_mode)
{
    return iso_allocate_inline(heap_ref, allocation_mode);
}

void* iso_try_allocate_array_by_count(pas_heap_ref* heap_ref, size_t count, size_t alignment, pas_allocation_mode allocation_mode)
{
    return iso_try_allocate_array_by_count_inline(heap_ref, count, alignment, allocation_mode);
}

void* iso_allocate_array_by_count(pas_heap_ref* heap_ref, size_t count, size_t alignment, pas_allocation_mode allocation_mode)
{
    return iso_allocate_array_by_count_inline(heap_ref, count, alignment, allocation_mode);
}

void* iso_try_allocate_array_by_count_zeroed(pas_heap_ref* heap_ref, size_t count, size_t alignment, pas_allocation_mode allocation_mode)
{
    return iso_try_allocate_array_by_count_zeroed_inline(heap_ref, count, alignment, allocation_mode);
}

void* iso_allocate_array_by_count_zeroed(pas_heap_ref* heap_ref, size_t count, size_t alignment, pas_allocation_mode allocation_mode)
{
    return iso_allocate_array_by_count_zeroed_inline(heap_ref, count, alignment, allocation_mode);
}

void* iso_try_reallocate_array_by_count(void* old_ptr, pas_heap_ref* heap_ref,
                                        size_t new_count,
                                        pas_reallocate_free_mode free_mode,
                                        pas_allocation_mode allocation_mode)
{
    return iso_try_reallocate_array_by_count_inline(old_ptr, heap_ref, new_count, free_mode, allocation_mode);
}

void* iso_reallocate_array_by_count(void* old_ptr, pas_heap_ref* heap_ref,
                                    size_t new_count,
                                    pas_reallocate_free_mode free_mode,
                                    pas_allocation_mode allocation_mode)
{
    return iso_reallocate_array_by_count_inline(old_ptr, heap_ref, new_count, free_mode, allocation_mode);
}

pas_heap* iso_heap_ref_get_heap(pas_heap_ref* heap_ref)
{
    return pas_ensure_heap(heap_ref, pas_normal_heap_ref_kind,
                           &iso_heap_config, &iso_typed_runtime_config.base);
}

void iso_primitive_heap_ref_construct(pas_primitive_heap_ref* heap_ref,
                                      pas_simple_type type)
{
    PAS_ASSERT(pas_simple_type_size(type) == 1);
    PAS_ASSERT(pas_simple_type_alignment(type) == 1);
    heap_ref->base.type = (const pas_heap_type*)type;
    heap_ref->base.heap = NULL;
    heap_ref->base.allocator_index = 0;
    heap_ref->cached_index = UINT_MAX;
}

void* iso_try_allocate_primitive(pas_primitive_heap_ref* heap_ref,
                                 size_t size,
                                 pas_allocation_mode allocation_mode)
{
    return iso_try_allocate_primitive_inline(heap_ref, size, allocation_mode);
}

void* iso_allocate_primitive(pas_primitive_heap_ref* heap_ref,
                             size_t size,
                             pas_allocation_mode allocation_mode)
{
    return iso_allocate_primitive_inline(heap_ref, size, allocation_mode);
}

void* iso_try_allocate_primitive_zeroed(pas_primitive_heap_ref* heap_ref,
                                        size_t size, pas_allocation_mode allocation_mode)
{
    return iso_try_allocate_primitive_zeroed_inline(heap_ref, size, allocation_mode);
}

void* iso_allocate_primitive_zeroed(pas_primitive_heap_ref* heap_ref,
                                    size_t size, pas_allocation_mode allocation_mode)
{
    return iso_allocate_primitive_zeroed_inline(heap_ref, size, allocation_mode);
}

void* iso_try_allocate_primitive_with_alignment(pas_primitive_heap_ref* heap_ref,
                                                size_t size,
                                                size_t alignment,
                                                pas_allocation_mode allocation_mode)
{
    return iso_try_allocate_primitive_with_alignment_inline(heap_ref, size, alignment, allocation_mode);
}

void* iso_allocate_primitive_with_alignment(pas_primitive_heap_ref* heap_ref,
                                            size_t size,
                                            size_t alignment,
                                            pas_allocation_mode allocation_mode)
{
    return iso_allocate_primitive_with_alignment_inline(heap_ref, size, alignment, allocation_mode);
}

void* iso_try_reallocate_primitive(void* old_ptr,
                                   pas_primitive_heap_ref* heap_ref,
                                   size_t new_size,
                                   pas_reallocate_free_mode free_mode,
                                   pas_allocation_mode allocation_mode)
{
    return iso_try_reallocate_primitive_inline(old_ptr, heap_ref, new_size, free_mode, allocation_mode);
}

void* iso_reallocate_primitive(void* old_ptr,
                               pas_primitive_heap_ref* heap_ref,
                               size_t new_size,
                               pas_reallocate_free_mode free_mode,
                               pas_allocation_mode allocation_mode)
{
    return iso_reallocate_primitive_inline(old_ptr, heap_ref, new_size, free_mode, allocation_mode);
}

void* iso_try_allocate_for_flex(const void* cls, size_t size, pas_allocation_mode allocation_mode)
{
    return iso_try_allocate_for_flex_inline(cls, size, allocation_mode);
}

bool iso_has_object(void* ptr)
{
    return iso_has_object_inline(ptr);
}

size_t iso_get_allocation_size(void* ptr)
{
    return iso_get_allocation_size_inline(ptr);
}

pas_heap* iso_get_heap(void* ptr)
{
    return iso_get_heap_inline(ptr);
}

void iso_deallocate(void* ptr)
{
    iso_deallocate_inline(ptr);
}

pas_heap* iso_force_primitive_heap_into_reserved_memory(pas_primitive_heap_ref* heap_ref,
                                                        uintptr_t begin,
                                                        uintptr_t end)
{
    return pas_ensure_heap_forced_into_reserved_memory(
        &heap_ref->base, pas_primitive_heap_ref_kind, &iso_heap_config,
        &iso_primitive_runtime_config.base, begin, end);
}

#endif /* PAS_ENABLE_ISO */

#endif /* LIBPAS_ENABLED */
