/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 4, 2022.
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
__PAS_BEGIN_EXTERN_C;

#pragma mark - Allocator functions

__PAS_API void* __thingy_try_allocate_primitive(__pas_size_t size, __pas_allocation_mode allocation_mode);
__PAS_API void* __thingy_try_allocate_primitive_zeroed(__pas_size_t size, __pas_allocation_mode allocation_mode);
__PAS_API void* __thingy_try_allocate_primitive_with_alignment(__pas_size_t size, __pas_size_t alignment, __pas_allocation_mode allocation_mode);

__PAS_API void* __thingy_try_reallocate_primitive(
    void* old_ptr, __pas_size_t new_size, __pas_allocation_mode allocation_mode);

__attribute__((malloc))
__PAS_API void* __thingy_try_allocate(__pas_heap_ref* heap_ref, __pas_allocation_mode allocation_mode);

__PAS_API void* __thingy_try_allocate_zeroed(__pas_heap_ref* heap_ref, __pas_allocation_mode allocation_mode);

/* FIXME: This should take the size, since the caller calculates it anyway. */
__attribute__((malloc))
__PAS_API void* __thingy_try_allocate_array(__pas_heap_ref* heap_ref,
                                            __pas_size_t count,
                                            __pas_size_t alignment,
                                            __pas_allocation_mode allocation_mode);

__PAS_API void* __thingy_try_allocate_zeroed_array(__pas_heap_ref* heap_ref,
                                                   __pas_size_t count,
                                                   __pas_size_t alignment,
                                                   __pas_allocation_mode allocation_mode);

__PAS_API void* __thingy_try_reallocate_array(void* old_ptr,
                                              __pas_heap_ref* heap_ref,
                                              __pas_size_t new_count,
                                              __pas_allocation_mode allocation_mode);

__PAS_API void __thingy_deallocate(void*);

__PAS_END_EXTERN_C;

