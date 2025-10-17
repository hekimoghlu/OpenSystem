/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 30, 2025.
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
#ifndef BMALLOC_HEAP_H
#define BMALLOC_HEAP_H

#include "bmalloc_heap_ref.h"
#include "pas_allocation_mode.h"
#include "pas_primitive_heap_ref.h"
#include "pas_reallocate_free_mode.h"

#if PAS_ENABLE_BMALLOC

PAS_BEGIN_EXTERN_C;

PAS_API void* bmalloc_try_allocate(size_t size, pas_allocation_mode allocation_mode);
PAS_API void* bmalloc_try_allocate_with_alignment(size_t size,
                                                  size_t alignment,
                                                  pas_allocation_mode allocation_mode);

PAS_API void* bmalloc_try_allocate_zeroed(size_t size, pas_allocation_mode allocation_mode);

PAS_API void* bmalloc_allocate(size_t size, pas_allocation_mode allocation_mode);
PAS_API void* bmalloc_allocate_with_alignment(size_t size,
                                              size_t alignment,
                                              pas_allocation_mode allocation_mode);

PAS_API void* bmalloc_allocate_zeroed(size_t size, pas_allocation_mode allocation_mode);

PAS_API void* bmalloc_try_reallocate(void* old_ptr, size_t new_size,
                                     pas_allocation_mode allocation_mode,
                                     pas_reallocate_free_mode free_mode);

PAS_API void* bmalloc_reallocate(void* old_ptr, size_t new_size,
                                 pas_allocation_mode allocation_mode,
                                 pas_reallocate_free_mode free_mode);

PAS_BAPI void* bmalloc_try_iso_allocate(pas_heap_ref* heap_ref, pas_allocation_mode allocation_mode);
PAS_BAPI void* bmalloc_iso_allocate(pas_heap_ref* heap_ref, pas_allocation_mode allocation_mode);

PAS_BAPI void* bmalloc_try_iso_allocate_array_by_size(pas_heap_ref* heap_ref, size_t size, pas_allocation_mode allocation_mode);
PAS_BAPI void* bmalloc_iso_allocate_array_by_size(pas_heap_ref* heap_ref, size_t size, pas_allocation_mode allocation_mode);

PAS_BAPI void* bmalloc_try_iso_allocate_zeroed_array_by_size(pas_heap_ref* heap_ref, size_t size, pas_allocation_mode allocation_mode);
PAS_BAPI void* bmalloc_iso_allocate_zeroed_array_by_size(pas_heap_ref* heap_ref, size_t size, pas_allocation_mode allocation_mode);

PAS_BAPI void* bmalloc_try_iso_allocate_array_by_size_with_alignment(
    pas_heap_ref* heap_ref, size_t size, size_t alignment, pas_allocation_mode allocation_mode);
PAS_BAPI void* bmalloc_iso_allocate_array_by_size_with_alignment(
    pas_heap_ref* heap_ref, size_t size, size_t alignment, pas_allocation_mode allocation_mode);

PAS_BAPI void* bmalloc_try_iso_reallocate_array_by_size(pas_heap_ref* heap_ref, void* ptr, size_t size, pas_allocation_mode allocation_mode);
PAS_BAPI void* bmalloc_iso_reallocate_array_by_size(pas_heap_ref* heap_ref, void* ptr, size_t size, pas_allocation_mode allocation_mode);

PAS_API void* bmalloc_try_iso_allocate_array_by_count(pas_heap_ref* heap_ref, size_t count, pas_allocation_mode allocation_mode);
PAS_API void* bmalloc_iso_allocate_array_by_count(pas_heap_ref* heap_ref, size_t count, pas_allocation_mode allocation_mode);

PAS_API void* bmalloc_try_iso_allocate_array_by_count_with_alignment(
    pas_heap_ref* heap_ref, size_t count, size_t alignment, pas_allocation_mode allocation_mode);
PAS_API void* bmalloc_iso_allocate_array_by_count_with_alignment(
    pas_heap_ref* heap_ref, size_t count, size_t alignment, pas_allocation_mode allocation_mode);

PAS_API void* bmalloc_try_iso_reallocate_array_by_count(pas_heap_ref* heap_ref, void* ptr, size_t count, pas_allocation_mode allocation_mode);
PAS_API void* bmalloc_iso_reallocate_array_by_count(pas_heap_ref* heap_ref, void* ptr, size_t count, pas_allocation_mode allocation_mode);

PAS_API pas_heap* bmalloc_heap_ref_get_heap(pas_heap_ref* heap_ref);

PAS_BAPI void* bmalloc_try_allocate_flex(pas_primitive_heap_ref* heap_ref, size_t size, pas_allocation_mode allocation_mode);
PAS_BAPI void* bmalloc_allocate_flex(pas_primitive_heap_ref* heap_ref, size_t size, pas_allocation_mode allocation_mode);

PAS_BAPI void* bmalloc_try_allocate_zeroed_flex(pas_primitive_heap_ref* heap_ref, size_t size, pas_allocation_mode allocation_mode);
PAS_BAPI void* bmalloc_allocate_zeroed_flex(pas_primitive_heap_ref* heap_ref, size_t size, pas_allocation_mode allocation_mode);

PAS_API void* bmalloc_try_allocate_flex_with_alignment(
    pas_primitive_heap_ref* heap_ref, size_t size, size_t alignment, pas_allocation_mode allocation_mode);
PAS_API void* bmalloc_allocate_flex_with_alignment(
    pas_primitive_heap_ref* heap_ref, size_t size, size_t alignment, pas_allocation_mode allocation_mode);

PAS_BAPI void* bmalloc_try_reallocate_flex(pas_primitive_heap_ref* heap_ref, void* old_ptr, size_t new_size, pas_allocation_mode allocation_mode);
PAS_BAPI void* bmalloc_reallocate_flex(pas_primitive_heap_ref* heap_ref, void* old_ptr, size_t new_size, pas_allocation_mode allocation_mode);

PAS_API pas_heap* bmalloc_flex_heap_ref_get_heap(pas_primitive_heap_ref* heap_ref);

PAS_API void* bmalloc_try_allocate_auxiliary(pas_primitive_heap_ref* heap_ref,
                                             size_t size,
                                             pas_allocation_mode allocation_mode);
PAS_API void* bmalloc_allocate_auxiliary(pas_primitive_heap_ref* heap_ref,
                                         size_t size,
                                         pas_allocation_mode allocation_mode);

PAS_API void* bmalloc_try_allocate_auxiliary_zeroed(pas_primitive_heap_ref* heap_ref,
                                                    size_t size,
                                                    pas_allocation_mode allocation_mode);
PAS_API void* bmalloc_allocate_auxiliary_zeroed(pas_primitive_heap_ref* heap_ref,
                                                size_t size,
                                                pas_allocation_mode allocation_mode);

PAS_API void* bmalloc_try_allocate_auxiliary_with_alignment(pas_primitive_heap_ref* heap_ref,
                                                            size_t size,
                                                            size_t alignment,
                                                            pas_allocation_mode allocation_mode);
PAS_API void* bmalloc_allocate_auxiliary_with_alignment(pas_primitive_heap_ref* heap_ref,
                                                        size_t size,
                                                        size_t alignment,
                                                        pas_allocation_mode allocation_mode);

PAS_API void* bmalloc_try_reallocate_auxiliary(void* old_ptr,
                                               pas_primitive_heap_ref* heap_ref,
                                               size_t new_size,
                                               pas_allocation_mode allocation_mode,
                                               pas_reallocate_free_mode free_mode);
PAS_API void* bmalloc_reallocate_auxiliary(void* old_ptr,
                                           pas_primitive_heap_ref* heap_ref,
                                           size_t new_size,
                                           pas_allocation_mode allocation_mode,
                                           pas_reallocate_free_mode free_mode);

PAS_API pas_heap* bmalloc_auxiliary_heap_ref_get_heap(pas_primitive_heap_ref* heap_ref);

PAS_API void bmalloc_deallocate(void*);

PAS_API pas_heap* bmalloc_force_auxiliary_heap_into_reserved_memory(pas_primitive_heap_ref* heap_ref,
                                                                    uintptr_t begin,
                                                                    uintptr_t end);

PAS_BAPI size_t bmalloc_heap_ref_get_type_size(pas_heap_ref* heap_ref);
PAS_API pas_heap* bmalloc_get_heap(void* ptr);
PAS_BAPI size_t bmalloc_get_allocation_size(void* ptr);

PAS_END_EXTERN_C;

#endif /* PAS_ENABLE_BMALLOC */

#endif /* BMALLOC_HEAP_H */

