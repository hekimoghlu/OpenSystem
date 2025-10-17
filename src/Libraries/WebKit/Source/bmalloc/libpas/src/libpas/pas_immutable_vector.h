/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 21, 2024.
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
#ifndef PAS_IMMUTABLE_VECTOR_H
#define PAS_IMMUTABLE_VECTOR_H

#include "pas_compact_atomic_ptr.h"
#include "pas_immortal_heap.h"
#include "pas_log.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

#define PAS_IMMUTABLE_VECTOR_INITIALIZER \
    { .array = PAS_COMPACT_ATOMIC_PTR_INITIALIZER, .size = 0, .capacity = 0 }

/* If you want perf, you want segment_size to be a power of 2. */
#define PAS_DECLARE_IMMUTABLE_VECTOR(name, type) \
    struct name; \
    typedef struct name name; \
    \
    PAS_DEFINE_COMPACT_ATOMIC_PTR(type, name##_ptr); \
    \
    struct name { \
        name##_ptr array; /* behind this is a next pointer. */ \
        unsigned size; \
        unsigned capacity; \
    }; \
    \
    PAS_UNUSED static inline void name##_construct(name* vector) \
    { \
        pas_zero_memory(vector, sizeof(name)); \
    } \
    \
    PAS_UNUSED static inline void name##_append( \
        name* vector, type value, pas_lock_hold_mode heap_lock_hold_mode) \
    { \
        static const bool verbose = false; \
        \
        type* array; \
        \
        if (verbose) \
            pas_log("doing append, vector = %p.\n", vector); \
        \
        array = name##_ptr_load(&vector->array); \
        \
        if (vector->size >= vector->capacity) { \
            type* new_array; \
            size_t new_array_capacity; \
            \
            PAS_ASSERT(vector->size == vector->capacity); \
            \
            new_array_capacity = (vector->capacity + 1) << 1; \
            PAS_ASSERT(vector->size < new_array_capacity); \
            \
            new_array = (type*)pas_immortal_heap_allocate_with_heap_lock_hold_mode( \
                new_array_capacity * sizeof(type), \
                #name "/array", pas_object_allocation, \
                heap_lock_hold_mode); \
            \
            memcpy(new_array, array, vector->size * sizeof(type)); \
            pas_zero_memory(new_array + vector->size, \
                        (new_array_capacity - vector->size) * sizeof(type)); \
            pas_fence(); \
            \
            name##_ptr_store(&vector->array, new_array); \
            pas_fence(); \
            PAS_ASSERT((unsigned)new_array_capacity == new_array_capacity); \
            vector->capacity = (unsigned)new_array_capacity; \
            array = new_array; \
        } \
        \
        array[vector->size] = value; \
        pas_fence(); \
        vector->size++; \
        PAS_ASSERT(vector->size); \
    } \
    \
    PAS_UNUSED static inline type* name##_get_ptr(name* vector, size_t index) \
    { \
        return name##_ptr_load(&vector->array) + index; \
    } \
    \
    PAS_UNUSED static inline type name##_get(name* vector, size_t index) \
    { \
        return *name##_get_ptr(vector, index); \
    } \
    \
    struct pas_dummy

PAS_END_EXTERN_C;

#endif /* PAS_IMMUTABLE_VECTOR_H */

