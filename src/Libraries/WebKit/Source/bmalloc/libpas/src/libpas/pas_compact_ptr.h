/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 21, 2022.
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
#ifndef PAS_COMPACT_PTR_H
#define PAS_COMPACT_PTR_H

#include "pas_compact_heap_reservation.h"
#include "pas_enumerator.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

#define PAS_COMPACT_PTR_INITIALIZER { .payload = { 0 } }

#define PAS_DEFINE_COMPACT_PTR_HELPERS(type, name) \
    static inline uintptr_t name ## _index_for_ptr(type* value) \
    { \
        uintptr_t ptr; \
        uintptr_t offset; \
        uintptr_t index; \
        if (!value) \
            return 0; \
        ptr = (uintptr_t)value; \
        offset = ptr - pas_compact_heap_reservation_base; \
        PAS_ASSERT(offset < pas_compact_heap_reservation_size); \
        index = offset / PAS_INTERNAL_MIN_ALIGN; \
        PAS_ASSERT(index * PAS_INTERNAL_MIN_ALIGN == offset); \
        PAS_ASSERT(index); \
        return index; \
    } \
    \
    static inline type* name ## _ptr_for_index(uintptr_t index) \
    { \
        if (!index) \
            return NULL; \
        return (type*)(index * PAS_INTERNAL_MIN_ALIGN + pas_compact_heap_reservation_base); \
    } \
    \
    static inline type* name ## _ptr_for_index_non_null(uintptr_t index) \
    { \
        PAS_TESTING_ASSERT(index); \
        return (type*)(index * PAS_INTERNAL_MIN_ALIGN + pas_compact_heap_reservation_base); \
    } \
    \
    static inline type* name ## _ptr_for_remote_index(pas_enumerator* enumerator, uintptr_t index) \
    { \
        if (!index) \
            return NULL; \
        return (type*)(index * PAS_INTERNAL_MIN_ALIGN + (uintptr_t)enumerator->compact_heap_copy_base); \
    } \
    \
    struct pas_dummy

#define PAS_DEFINE_COMPACT_PTR(type, name) \
    struct name; \
    typedef struct name name; \
    \
    struct name { \
        uint8_t payload[PAS_COMPACT_PTR_SIZE]; \
    }; \
    \
    PAS_DEFINE_COMPACT_PTR_HELPERS(type, name); \
    \
    static inline void name ## _store(name* ptr, type* value) \
    { \
        uintptr_t ptr_as_index; \
        size_t byte_index; \
        ptr_as_index = name ## _index_for_ptr(value); \
        for (byte_index = 0; byte_index < PAS_COMPACT_PTR_SIZE; ++byte_index) { \
            ptr->payload[byte_index] = (uintptr_t)ptr_as_index; \
            ptr_as_index >>= 8; \
        } \
    } \
    \
    static inline type* name ## _load(name* ptr) \
    { \
        uintptr_t ptr_as_index = 0; \
        memcpy(&ptr_as_index, ptr->payload, PAS_COMPACT_PTR_SIZE); \
        ptr_as_index &= PAS_COMPACT_PTR_MASK; \
        return name ## _ptr_for_index(ptr_as_index); \
    } \
    \
    static inline type* name ## _load_non_null(name* ptr) \
    { \
        uintptr_t ptr_as_index = 0; \
        memcpy(&ptr_as_index, ptr->payload, PAS_COMPACT_PTR_SIZE); \
        ptr_as_index &= PAS_COMPACT_PTR_MASK; \
        return name ## _ptr_for_index_non_null(ptr_as_index); \
    } \
    \
    static inline bool name ## _is_null(name* ptr) \
    { \
        uintptr_t ptr_as_index = 0; \
        memcpy(&ptr_as_index, ptr->payload, PAS_COMPACT_PTR_SIZE); \
        ptr_as_index &= PAS_COMPACT_PTR_MASK; \
        return !ptr_as_index; \
    } \
    \
    static inline type* name ## _load_remote(pas_enumerator* enumerator, name* ptr) \
    { \
        uintptr_t ptr_as_index = 0; \
        memcpy(&ptr_as_index, ptr->payload, PAS_COMPACT_PTR_SIZE); \
        ptr_as_index &= PAS_COMPACT_PTR_MASK; \
        return name ## _ptr_for_remote_index(enumerator, ptr_as_index); \
    } \
    \
    struct pas_dummy

PAS_END_EXTERN_C;

#endif /* PAS_COMPACT_PTR_H */

