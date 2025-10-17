/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 26, 2023.
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
#ifndef PAS_SIMPLE_TYPE_H
#define PAS_SIMPLE_TYPE_H

#include "pas_config.h"

#include "pas_heap_ref.h"
#include "pas_utils.h"
#include <stdio.h>

PAS_BEGIN_EXTERN_C;

#define PAS_SIMPLE_TYPE_DATA_BIT (PAS_NUM_PTR_BITS - 1)
#define PAS_SIMPLE_TYPE_DATA_PTR_MASK (((uintptr_t)1 << PAS_SIMPLE_TYPE_DATA_BIT) - 1)
#define PAS_SIMPLE_TYPE_NUM_ALIGNMENT_BITS 5
#define PAS_SIMPLE_TYPE_ALIGNMENT_SHIFT (PAS_SIMPLE_TYPE_DATA_BIT - \
                                         PAS_SIMPLE_TYPE_NUM_ALIGNMENT_BITS)
#define PAS_SIMPLE_TYPE_ALIGNMENT_MASK ((((uintptr_t)1 << PAS_SIMPLE_TYPE_NUM_ALIGNMENT_BITS) \
                                         - 1) \
                                        << PAS_SIMPLE_TYPE_ALIGNMENT_SHIFT)
#define PAS_SIMPLE_TYPE_SIZE_MASK (((uintptr_t)1 << PAS_SIMPLE_TYPE_ALIGNMENT_SHIFT) - 1)

typedef uintptr_t pas_simple_type;

struct pas_simple_type_with_key_data;
struct pas_stream;
typedef struct pas_simple_type_with_key_data pas_simple_type_with_key_data;
typedef struct pas_stream pas_stream;

struct pas_simple_type_with_key_data {
    uintptr_t simple_type;
    const void* key;
};

/* NOTE: to get default alignment, use alignment=1. This means to defer the alignment decision
   to the the minalign settings of the heap you're allocating in. */
#define PAS_SIMPLE_TYPE_CREATE(size, alignment) \
    ((size) | ((pas_simple_type)__builtin_ctzll(alignment) \
               << PAS_SIMPLE_TYPE_ALIGNMENT_SHIFT))

#define PAS_SIMPLE_TYPE_SIZE(type) \
    ((type) << PAS_SIMPLE_TYPE_NUM_ALIGNMENT_BITS >> PAS_SIMPLE_TYPE_NUM_ALIGNMENT_BITS)

#define PAS_SIMPLE_TYPE_ALIGNMENT(type) \
    ((size_t)1 << ((type) >> PAS_SIMPLE_TYPE_ALIGNMENT_SHIFT))

static inline bool pas_simple_type_has_key(pas_simple_type type)
{
    return type >> PAS_SIMPLE_TYPE_DATA_BIT;
}

static inline const pas_simple_type_with_key_data* pas_simple_type_get_key_data(pas_simple_type type)
{
    PAS_ASSERT(pas_simple_type_has_key(type));
    return (const pas_simple_type_with_key_data*)(type & PAS_SIMPLE_TYPE_DATA_PTR_MASK);
}

static inline pas_simple_type pas_simple_type_unwrap(pas_simple_type type)
{
    if (pas_simple_type_has_key(type))
        return pas_simple_type_get_key_data(type)->simple_type;
    return type;
}

/* It's important that this function can be DCE'd. */
static inline size_t pas_simple_type_size(pas_simple_type type)
{
    return pas_simple_type_unwrap(type) & PAS_SIMPLE_TYPE_SIZE_MASK;
}

/* It's important that this function can be DCE'd. */
static inline size_t pas_simple_type_alignment(pas_simple_type type)
{
    return (size_t)1 << ((pas_simple_type_unwrap(type) & PAS_SIMPLE_TYPE_ALIGNMENT_MASK)
                         >> PAS_SIMPLE_TYPE_ALIGNMENT_SHIFT);
}

static inline const void* pas_simple_type_key(pas_simple_type type)
{
    return pas_simple_type_get_key_data(type)->key;
}

static inline pas_simple_type pas_simple_type_create(size_t size, size_t alignment)
{
    pas_simple_type result;
    
    result = PAS_SIMPLE_TYPE_CREATE(size, alignment);
    
    PAS_ASSERT(pas_simple_type_size(result) == size);
    PAS_ASSERT(pas_simple_type_alignment(result) == alignment);
    PAS_ASSERT(!pas_simple_type_has_key(result));
    
    return result;
}

static inline pas_simple_type pas_simple_type_create_with_key_data(
    const pas_simple_type_with_key_data* data)
{
    pas_simple_type result;
    
    result = ((uintptr_t)1 << PAS_SIMPLE_TYPE_DATA_BIT) | (uintptr_t)data;

    PAS_ASSERT(pas_simple_type_has_key(result));
    PAS_ASSERT(pas_simple_type_get_key_data(result) == data);

    return result;
}

PAS_API void pas_simple_type_dump(pas_simple_type type, pas_stream* stream);

static inline size_t pas_simple_type_as_heap_type_get_type_size(const pas_heap_type* type)
{
    return pas_simple_type_size((pas_simple_type)type);
}

static inline size_t pas_simple_type_as_heap_type_get_type_alignment(const pas_heap_type* type)
{
    return pas_simple_type_alignment((pas_simple_type)type);
}

PAS_API void pas_simple_type_as_heap_type_dump(const pas_heap_type* type, pas_stream* stream);

PAS_END_EXTERN_C;

#endif /* PAS_SIMPLE_TYPE_H */

