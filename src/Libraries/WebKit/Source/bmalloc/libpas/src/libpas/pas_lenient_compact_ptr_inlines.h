/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 5, 2023.
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
#ifndef PAS_LENIENT_COMPACT_PTR_INLINES_H
#define PAS_LENIENT_COMPACT_PTR_INLINES_H

#include "pas_lenient_compact_ptr.h"
#include "pas_utility_heap.h"

PAS_BEGIN_EXTERN_C;

#define PAS_DEFINE_LENIENT_COMPACT_PTR(type, name) \
    void name ## _destruct(name* ptr) \
    { \
        type* old_value; \
        old_value = name ## _compact_tagged_atomic_ptr_load(&ptr->ptr); \
        if ((uintptr_t)old_value & PAS_LENIENT_COMPACT_PTR_FULL_PTR_BIT) { \
            pas_utility_heap_deallocate( \
                (void*)((uintptr_t)old_value & ~PAS_LENIENT_COMPACT_PTR_FULL_PTR_BIT)); \
        } \
    } \
    \
    void name ## _store(name* ptr, type* value) \
    { \
        PAS_TESTING_ASSERT(!((uintptr_t)value & PAS_LENIENT_COMPACT_PTR_FULL_PTR_BIT)); \
        name ## _destruct(ptr); \
        if ((uintptr_t)value >= PAS_INTERNAL_MIN_ALIGN \
            && (uintptr_t)value - pas_compact_heap_reservation_base >= pas_compact_heap_reservation_size) { \
            type** box; \
            box = pas_utility_heap_allocate(sizeof(type*), #name "/box"); \
            *box = value; \
            value = (type*)((uintptr_t)box | PAS_LENIENT_COMPACT_PTR_FULL_PTR_BIT); \
        } \
        name ## _compact_tagged_atomic_ptr_store(&ptr->ptr, value); \
    } \
    \
    type* name ## _load(name* ptr) \
    { \
        type* result; \
        result = name ## _compact_tagged_atomic_ptr_load(&ptr->ptr); \
        if ((uintptr_t)result & PAS_LENIENT_COMPACT_PTR_FULL_PTR_BIT) \
            return *(type**)((uintptr_t)result & ~PAS_LENIENT_COMPACT_PTR_FULL_PTR_BIT); \
        return result; \
    } \
    \
    type* name ## _load_remote(pas_enumerator* enumerator, name* ptr, size_t size) \
    { \
        type* result; \
        result = name ## _compact_tagged_atomic_ptr_load_remote(enumerator, &ptr->ptr); \
        if ((uintptr_t)result & PAS_LENIENT_COMPACT_PTR_FULL_PTR_BIT) { \
            return (type*)pas_enumerator_read( \
                enumerator, \
                *(type**)((uintptr_t)result & ~PAS_LENIENT_COMPACT_PTR_FULL_PTR_BIT), \
                size); \
        } \
        return result; \
    } \
    \
    struct pas_dummy

PAS_END_EXTERN_C;

#endif /* PAS_LENIENT_COMPACT_PTR_INLINES_H */

