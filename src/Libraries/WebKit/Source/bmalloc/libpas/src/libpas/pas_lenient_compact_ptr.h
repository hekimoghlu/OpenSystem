/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 2, 2022.
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
#ifndef PAS_LENIENT_COMPACT_PTR_H
#define PAS_LENIENT_COMPACT_PTR_H

#include "pas_compact_tagged_atomic_ptr.h"

PAS_BEGIN_EXTERN_C;

/* You can use this pointer to point at something that is likely to be in the compact heap but that
   sometimes won't be. To make this work, it's necessary to be able to destruct the pointer, and it's
   not legal to pass the pointer around by value. Also, the thing being pointed to must have the lowest
   bit available (i.e. that bit must always be zero). */

#define PAS_LENIENT_COMPACT_PTR_INITIALIZER { .ptr = PAS_COMPACT_TAGGED_ATOMIC_PTR_INITIALIZER }

#define PAS_LENIENT_COMPACT_PTR_FULL_PTR_BIT ((uintptr_t)1)

#define PAS_DECLARE_LENIENT_COMPACT_PTR(type, name) \
    \
    PAS_DEFINE_COMPACT_TAGGED_ATOMIC_PTR(type*, name ## _compact_tagged_atomic_ptr); \
    \
    struct name; \
    typedef struct name name; \
    \
    struct name { \
        name ## _compact_tagged_atomic_ptr ptr; \
    }; \
    \
    PAS_API void name ## _destruct(name* ptr); \
    PAS_API void name ## _store(name* ptr, type* value); \
    PAS_API type* name ## _load(name* ptr); \
    \
    static inline type* name ## _load_compact(name* ptr) \
    { \
        type* result; \
        result = name ## _compact_tagged_atomic_ptr_load(&ptr->ptr); \
        PAS_TESTING_ASSERT(!((uintptr_t)result & PAS_LENIENT_COMPACT_PTR_FULL_PTR_BIT)); \
        return result; \
    } \
    \
    static inline type* name ## _load_compact_non_null(name* ptr) \
    { \
        type* result; \
        result = name ## _compact_tagged_atomic_ptr_load_non_null(&ptr->ptr); \
        PAS_TESTING_ASSERT(!((uintptr_t)result & PAS_LENIENT_COMPACT_PTR_FULL_PTR_BIT)); \
        return result; \
    } \
    \
    PAS_API type* name ## _load_remote(pas_enumerator* enumerator, name* ptr, size_t size); \
    \
    struct pas_dummy

PAS_END_EXTERN_C;

#endif /* PAS_LENIENT_COMPACT_PTR_H */

