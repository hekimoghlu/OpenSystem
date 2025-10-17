/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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
#ifndef ISO_HEAP_H
#define ISO_HEAP_H

#include "iso_heap_ref.h"
#include "pas_primitive_heap_ref.h"
#include "pas_reallocate_free_mode.h"
#include "pas_allocation_mode.h"

#if PAS_ENABLE_ISO

PAS_BEGIN_EXTERN_C;

PAS_API void* iso_try_allocate_common_primitive(size_t size, pas_allocation_mode allocation_mode);
PAS_API void* iso_try_allocate_common_primitive_with_alignment(size_t size,
                                                               size_t alignment,
                                                               pas_allocation_mode allocation_mode);

PAS_API void* iso_try_allocate_common_primitive_zeroed(size_t size, pas_allocation_mode allocation_mode);

PAS_API void* iso_allocate_common_primitive(size_t size, pas_allocation_mode allocation_mode);
PAS_API void* iso_allocate_common_primitive_with_alignment(size_t size,
                                                           size_t alignment,
                                                           pas_allocation_mode allocation_mode);

PAS_API void* iso_allocate_common_primitive_zeroed(size_t size, pas_allocation_mode allocation_mode);

PAS_API void* iso_try_reallocate_common_primitive(void* old_ptr, size_t new_size,
                                                  pas_reallocate_free_mode free_mode,
                                                  pas_allocation_mode allocation_mode);

PAS_API void* iso_reallocate_common_primitive(void* old_ptr, size_t new_size,
                                              pas_reallocate_free_mode free_mode,
                                              pas_allocation_mode allocation_mode);

PAS_API void* iso_try_allocate_dynamic_primitive(const void* key, size_t size, pas_allocation_mode allocation_mode);
PAS_API void* iso_try_allocate_dynamic_primitive_with_alignment(const void* key,
                                                                size_t size,
                                                                size_t alignment,
                                                                pas_allocation_mode allocation_mode);

PAS_API void* iso_try_allocate_dynamic_primitive_zeroed(const void* key,
                                                        size_t size,
                                                        pas_allocation_mode allocation_mode);

PAS_API void* iso_try_reallocate_dynamic_primitive(void* old_ptr,
                                                   const void* key,
                                                   size_t new_size,
                                                   pas_reallocate_free_mode free_mode,
                                                   pas_allocation_mode allocation_mode);

PAS_API void iso_heap_ref_construct(pas_heap_ref* heap_ref,
                                    pas_simple_type type);

PAS_API void* iso_try_allocate(pas_heap_ref* heap_ref, pas_allocation_mode pas_allocation_mode);
PAS_API void* iso_allocate(pas_heap_ref* heap_ref, pas_allocation_mode pas_allocation_mode);

PAS_API void* iso_try_allocate_array_by_count(pas_heap_ref* heap_ref, size_t count, size_t alignment, pas_allocation_mode allocation_mode);
PAS_API void* iso_allocate_array_by_count(pas_heap_ref* heap_ref, size_t count, size_t alignment, pas_allocation_mode allocation_mode);

PAS_API void* iso_try_allocate_array_by_count_zeroed(pas_heap_ref* heap_ref, size_t count, size_t alignment, pas_allocation_mode allocation_mode);
PAS_API void* iso_allocate_array_by_count_zeroed(pas_heap_ref* heap_ref, size_t count, size_t alignment, pas_allocation_mode allocation_mode);

PAS_API void* iso_try_reallocate_array_by_count(void* old_ptr, pas_heap_ref* heap_ref,
                                                size_t new_count,
                                                pas_reallocate_free_mode free_mode,
                                                pas_allocation_mode allocation_mode);
PAS_API void* iso_reallocate_array_by_count(void* old_ptr, pas_heap_ref* heap_ref,
                                            size_t new_count,
                                            pas_reallocate_free_mode free_mode,
                                            pas_allocation_mode allocation_mode);

PAS_API pas_heap* iso_heap_ref_get_heap(pas_heap_ref* heap_ref);

PAS_API void iso_primitive_heap_ref_construct(pas_primitive_heap_ref* heap_ref,
                                              pas_simple_type type);

PAS_API void* iso_try_allocate_primitive(pas_primitive_heap_ref* heap_ref,
                                         size_t size,
                                         pas_allocation_mode allocation_mode);
PAS_API void* iso_allocate_primitive(pas_primitive_heap_ref* heap_ref,
                                     size_t size,
                                     pas_allocation_mode allocation_mode);

PAS_API void* iso_try_allocate_primitive_zeroed(pas_primitive_heap_ref* heap_ref,
                                                size_t size,
                                                pas_allocation_mode allocation_mode);
PAS_API void* iso_allocate_primitive_zeroed(pas_primitive_heap_ref* heap_ref,
                                            size_t size,
                                            pas_allocation_mode allocation_mode);

PAS_API void* iso_try_allocate_primitive_with_alignment(pas_primitive_heap_ref* heap_ref,
                                                        size_t size,
                                                        size_t alignment,
                                                        pas_allocation_mode allocation_mode);
PAS_API void* iso_allocate_primitive_with_alignment(pas_primitive_heap_ref* heap_ref,
                                                    size_t size,
                                                    size_t alignment,
                                                    pas_allocation_mode allocation_mode);

PAS_API void* iso_try_reallocate_primitive(void* old_ptr,
                                           pas_primitive_heap_ref* heap_ref,
                                           size_t new_size,
                                           pas_reallocate_free_mode free_mode,
                                           pas_allocation_mode allocation_mode);
PAS_API void* iso_reallocate_primitive(void* old_ptr,
                                       pas_primitive_heap_ref* heap_ref,
                                       size_t new_size,
                                       pas_reallocate_free_mode free_mode,
                                       pas_allocation_mode allocation_mode);

PAS_API pas_heap* iso_primitive_heap_ref_get_heap(pas_primitive_heap_ref* heap_ref);

PAS_API void* iso_try_allocate_for_flex(const void* cls, size_t size, pas_allocation_mode allocation_mode);

PAS_API bool iso_has_object(void*);
PAS_API size_t iso_get_allocation_size(void*);
PAS_API pas_heap* iso_get_heap(void*);

PAS_API void iso_deallocate(void*);

PAS_API pas_heap* iso_force_primitive_heap_into_reserved_memory(pas_primitive_heap_ref* heap_ref,
                                                                uintptr_t begin,
                                                                uintptr_t end);

PAS_END_EXTERN_C;

#endif /* PAS_ENABLE_ISO */

#endif /* ISO_HEAP_H */

