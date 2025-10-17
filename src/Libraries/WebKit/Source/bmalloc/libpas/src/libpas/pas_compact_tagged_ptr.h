/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 25, 2023.
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
#ifndef PAS_COMPACT_TAGGED_PTR_H
#define PAS_COMPACT_TAGGED_PTR_H

#include "pas_compact_heap_reservation.h"
#include "pas_enumerator.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

#define PAS_COMPACT_TAGGED_PTR_SIZE (PAS_COMPACT_PTR_SIZE + 1)
#define PAS_COMPACT_TAGGED_PTR_BITS (PAS_COMPACT_TAGGED_PTR_SIZE << 3)
#define PAS_COMPACT_TAGGED_PTR_MASK (((uintptr_t)1 << PAS_COMPACT_TAGGED_PTR_BITS) - 1)

#define PAS_COMPACT_TAGGED_PTR_INITIALIZER { .payload = {[0 ... PAS_COMPACT_TAGGED_PTR_SIZE - 1] = 0} }

#define PAS_DEFINE_COMPACT_TAGGED_PTR_HELPERS(type, name) \
    static inline uintptr_t name ## _offset_for_ptr(type value) \
    { \
        uintptr_t ptr; \
        uintptr_t offset; \
        ptr = (uintptr_t)value; \
        if (ptr < PAS_INTERNAL_MIN_ALIGN) \
            return ptr; \
        offset = ptr - pas_compact_heap_reservation_base; \
        PAS_ASSERT(offset < pas_compact_heap_reservation_size); \
        PAS_ASSERT(offset); \
        return offset; \
    } \
    \
    static inline type name ## _ptr_for_offset(uintptr_t offset) \
    { \
        if (offset < PAS_INTERNAL_MIN_ALIGN) \
            return (type)offset; \
        return (type)(offset + pas_compact_heap_reservation_base); \
    } \
    \
    static inline type name ## _ptr_for_offset_non_null(uintptr_t offset) \
    { \
        PAS_ASSERT(offset >= PAS_INTERNAL_MIN_ALIGN); \
        return (type)(offset + pas_compact_heap_reservation_base); \
    } \
    \
    static inline type name ## _ptr_for_remote_offset(pas_enumerator* enumerator, uintptr_t offset) \
    { \
        if (offset < PAS_INTERNAL_MIN_ALIGN) \
            return (type)offset; \
        return (type)(offset + (uintptr_t)enumerator->compact_heap_copy_base); \
    } \
    \
    struct pas_dummy

/* Supports using the low bits as tag bits. */
#define PAS_DEFINE_COMPACT_TAGGED_PTR(type, name) \
    struct name; \
    typedef struct name name; \
    \
    struct name { \
        uint8_t payload[PAS_COMPACT_TAGGED_PTR_SIZE]; \
    }; \
    \
    PAS_DEFINE_COMPACT_TAGGED_PTR_HELPERS(type, name); \
    \
    static inline void name ## _store(name* ptr, type value) \
    { \
        uintptr_t offset; \
        size_t index; \
        offset = name ## _offset_for_ptr(value); \
        for (index = 0; index < PAS_COMPACT_TAGGED_PTR_SIZE; ++index) { \
            ptr->payload[index] = (uintptr_t)offset; \
            offset >>= 8; \
        } \
    } \
    \
    static inline bool name ## _is_null(name* ptr) \
    { \
        uintptr_t ptr_as_offset = 0; \
        memcpy(&ptr_as_offset, ptr->payload, PAS_COMPACT_TAGGED_PTR_SIZE); \
        ptr_as_offset &= PAS_COMPACT_TAGGED_PTR_MASK; \
        return !ptr_as_offset; \
    } \
    \
    static inline type name ## _load(name* ptr) \
    { \
        uintptr_t ptr_as_offset = 0; \
        memcpy(&ptr_as_offset, ptr->payload, PAS_COMPACT_TAGGED_PTR_SIZE); \
        ptr_as_offset &= PAS_COMPACT_TAGGED_PTR_MASK; \
        return name ## _ptr_for_offset(ptr_as_offset); \
    } \
    \
    static inline type name ## _load_non_null(name* ptr) \
    { \
        uintptr_t ptr_as_offset = 0; \
        memcpy(&ptr_as_offset, ptr->payload, PAS_COMPACT_TAGGED_PTR_SIZE); \
        ptr_as_offset &= PAS_COMPACT_TAGGED_PTR_MASK; \
        return name ## _ptr_for_offset_non_null(ptr_as_offset); \
    } \
    \
    static inline type name ## _load_remote(pas_enumerator* enumerator, name* ptr) \
    { \
        uintptr_t ptr_as_offset = 0; \
        memcpy(&ptr_as_offset, ptr->payload, PAS_COMPACT_TAGGED_PTR_SIZE); \
        ptr_as_offset &= PAS_COMPACT_TAGGED_PTR_MASK; \
        return name ## _ptr_for_remote_offset(enumerator, ptr_as_offset); \
    } \
    \
    struct pas_dummy

PAS_END_EXTERN_C;

#endif /* PAS_COMPACT_TAGGED_PTR_H */

