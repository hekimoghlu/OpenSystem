/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 8, 2022.
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
#ifndef PAS_COMPACT_ATOMIC_PTR_H
#define PAS_COMPACT_ATOMIC_PTR_H

#include "pas_compact_ptr.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

#if PAS_COMPACT_PTR_SIZE <= 4
typedef unsigned pas_compact_atomic_ptr_impl;
#define pas_compact_atomic_ptr_impl_weak_cas pas_compare_and_swap_uint32_weak
#elif PAS_COMPACT_PTR_SIZE <= 8
typedef uint64_t pas_compact_atomic_ptr_impl;
#define pas_compact_atomic_ptr_impl_weak_cas pas_compare_and_swap_uint64_weak
#else
#error "Cannot use PAS_COMPACT_PTR_SIZE > 8"
#endif

#define PAS_COMPACT_ATOMIC_PTR_INITIALIZER { .payload = 0 }

#define PAS_DEFINE_COMPACT_ATOMIC_PTR(type, name) \
    struct name; \
    typedef struct name name; \
    \
    struct name { \
        pas_compact_atomic_ptr_impl payload; \
    }; \
    \
    PAS_DEFINE_COMPACT_PTR_HELPERS(type, name); \
    \
    static inline void name ## _store(name* ptr, type* value) \
    { \
        ptr->payload = (pas_compact_atomic_ptr_impl)name ## _index_for_ptr(value); \
    } \
    \
    static inline type* name ## _load(name* ptr) \
    { \
        return name ## _ptr_for_index(ptr->payload); \
    } \
    \
    static inline type* name ## _load_non_null(name* ptr) \
    { \
        return name ## _ptr_for_index_non_null(ptr->payload); \
    } \
    \
    static inline bool name ## _weak_cas(name* ptr, type* old_value, type* new_value) \
    { \
        return pas_compare_and_swap_uint32_weak( \
            &ptr->payload, \
            (pas_compact_atomic_ptr_impl)name ## _index_for_ptr(old_value), \
            (pas_compact_atomic_ptr_impl)name ## _index_for_ptr(new_value)); \
    } \
    \
    static inline type* name ## _strong_cas(name* ptr, type* old_value, type* new_value) \
    { \
        return name ## _ptr_for_index( \
            pas_compare_and_swap_uint32_strong( \
                &ptr->payload, \
                (pas_compact_atomic_ptr_impl)name ## _index_for_ptr(old_value), \
                (pas_compact_atomic_ptr_impl)name ## _index_for_ptr(new_value))); \
    } \
    \
    static inline bool name ## _is_null(name* ptr) \
    { \
        return !ptr->payload; \
    } \
    \
    static inline type* name ## _load_remote(pas_enumerator* enumerator, name* ptr) \
    { \
        return name ## _ptr_for_remote_index(enumerator, ptr->payload); \
    } \
    \
    struct pas_dummy

PAS_END_EXTERN_C;

#endif /* PAS_COMPACT_ATOMIC_PTR_H */

