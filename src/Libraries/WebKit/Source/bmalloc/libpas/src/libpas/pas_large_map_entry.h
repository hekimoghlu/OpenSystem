/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 5, 2023.
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
#ifndef PAS_LARGE_MAP_ENTRY_H
#define PAS_LARGE_MAP_ENTRY_H

#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_large_heap;
struct pas_large_map_entry;

typedef struct pas_large_heap pas_large_heap;
typedef struct pas_large_map_entry pas_large_map_entry;

typedef uintptr_t pas_large_map_key;

struct pas_large_map_entry {
    uintptr_t begin;
    uintptr_t end;
    pas_large_heap* heap;
};

static inline pas_large_map_entry pas_large_map_entry_create_empty(void)
{
    pas_large_map_entry result;
    result.begin = 0;
    result.end = 0;
    result.heap = NULL;
    return result;
}

static inline pas_large_map_entry pas_large_map_entry_create_deleted(void)
{
    pas_large_map_entry result;
    result.begin = 1;
    result.end = 0;
    result.heap = NULL;
    return result;
}

static inline bool pas_large_map_entry_is_empty_or_deleted(pas_large_map_entry entry)
{
    return !entry.end;
}

static inline bool pas_large_map_entry_is_empty(pas_large_map_entry entry)
{
    return !entry.begin;
}

static inline bool pas_large_map_entry_is_deleted(pas_large_map_entry entry)
{
    return entry.begin == 1;
}

static inline uintptr_t pas_large_map_entry_get_key(pas_large_map_entry entry)
{
    return entry.begin;
}

static inline unsigned pas_large_map_key_get_hash(uintptr_t key)
{
    return pas_large_object_hash(key);
}

static inline bool pas_large_map_key_is_equal(uintptr_t a, uintptr_t b)
{
    return a == b;
}

PAS_END_EXTERN_C;

#endif /* PAS_LARGE_MAP_ENTRY_H */

