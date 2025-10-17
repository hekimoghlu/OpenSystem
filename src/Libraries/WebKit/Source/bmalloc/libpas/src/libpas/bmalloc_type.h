/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 29, 2024.
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
#ifndef BMALLOC_TYPE_H
#define BMALLOC_TYPE_H

#include "pas_simple_type.h"

PAS_BEGIN_EXTERN_C;

struct bmalloc_type;
typedef struct bmalloc_type bmalloc_type;

struct bmalloc_type {
    unsigned size;
    unsigned alignment;
    const char* name;
};

#define BMALLOC_TYPE_INITIALIZER(passed_size, passed_alignment, passed_name) \
    ((bmalloc_type){ \
         .size = (passed_size), \
         .alignment = (passed_alignment), \
         .name = (passed_name) \
     })

/* It's a bit better to use these getters instead of accessing the type struct directly because we want to be
   able to change the shape of the struct. */
static inline size_t bmalloc_type_size(const bmalloc_type* type)
{
    return type->size;
}

static inline size_t bmalloc_type_alignment(const bmalloc_type* type)
{
    return type->alignment;
}

static inline const char* bmalloc_type_name(const bmalloc_type* type)
{
    return type->name;
}

PAS_API bmalloc_type* bmalloc_type_create(size_t size, size_t alignment, const char* name);

PAS_API bool bmalloc_type_try_name_dump(pas_stream* stream, const char* name);
PAS_API void bmalloc_type_name_dump(pas_stream* stream, const char* name);

PAS_API void bmalloc_type_dump(const bmalloc_type* type, pas_stream* stream);

static inline size_t bmalloc_type_as_heap_type_get_type_size(const pas_heap_type* type)
{
    return bmalloc_type_size((const bmalloc_type*)type);
}

static inline size_t bmalloc_type_as_heap_type_get_type_alignment(const pas_heap_type* type)
{
    return bmalloc_type_alignment((const bmalloc_type*)type);
}

PAS_API void bmalloc_type_as_heap_type_dump(const pas_heap_type* type, pas_stream* stream);

PAS_END_EXTERN_C;

#endif /* BMALLOC_TYPE_H */

