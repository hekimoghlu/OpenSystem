/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 23, 2023.
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
#ifndef PAS_SMALL_LARGE_MAP_ENTRY_H
#define PAS_SMALL_LARGE_MAP_ENTRY_H

#include "pas_large_map_entry.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_large_heap;
struct pas_small_large_map_entry;

typedef struct pas_large_heap pas_large_heap;
typedef struct pas_small_large_map_entry pas_small_large_map_entry;

struct pas_small_large_map_entry {
    unsigned encoded_begin;
    unsigned encoded_size;
    unsigned encoded_heap;
};

static inline pas_small_large_map_entry pas_small_large_map_entry_create_empty(void)
{
    pas_small_large_map_entry result;
    result.encoded_begin = 0;
    result.encoded_size = 0;
    result.encoded_heap = 0;
    return result;
}

static inline pas_small_large_map_entry pas_small_large_map_entry_create_deleted(void)
{
    pas_small_large_map_entry result;
    result.encoded_begin = 1;
    result.encoded_size = 0;
    result.encoded_heap = 0;
    return result;
}

static inline bool pas_small_large_map_entry_is_empty_or_deleted(pas_small_large_map_entry entry)
{
    return !entry.encoded_size;
}

static inline bool pas_small_large_map_entry_is_empty(pas_small_large_map_entry entry)
{
    return !entry.encoded_begin;
}

static inline bool pas_small_large_map_entry_is_deleted(pas_small_large_map_entry entry)
{
    return entry.encoded_begin == 1;
}

static inline pas_small_large_map_entry pas_small_large_map_entry_create(pas_large_map_entry entry)
{
    pas_small_large_map_entry result;
    result.encoded_begin = (unsigned)(entry.begin >> PAS_MIN_ALIGN_SHIFT);
    result.encoded_size = (unsigned)((entry.end - entry.begin) >> PAS_MIN_ALIGN_SHIFT);
    result.encoded_heap = (unsigned)((uintptr_t)entry.heap / PAS_INTERNAL_MIN_ALIGN);
    return result;
}

static inline uintptr_t pas_small_large_map_entry_begin(pas_small_large_map_entry entry)
{
    return (uintptr_t)entry.encoded_begin << PAS_MIN_ALIGN_SHIFT;
}

static inline uintptr_t pas_small_large_map_entry_get_key(pas_small_large_map_entry entry)
{
    return pas_small_large_map_entry_begin(entry);
}

static inline uintptr_t pas_small_large_map_entry_end(pas_small_large_map_entry entry)
{
    return pas_small_large_map_entry_begin(entry)
        + ((uintptr_t)entry.encoded_size << PAS_MIN_ALIGN_SHIFT);
}

static inline pas_large_heap* pas_small_large_map_entry_heap(pas_small_large_map_entry entry)
{
    return (pas_large_heap*)((uintptr_t)entry.encoded_heap * PAS_INTERNAL_MIN_ALIGN);
}

static inline pas_large_map_entry pas_small_large_map_entry_get_entry(
    pas_small_large_map_entry entry)
{
    pas_large_map_entry result;
    result.begin = pas_small_large_map_entry_begin(entry);
    result.end = pas_small_large_map_entry_end(entry);
    result.heap = pas_small_large_map_entry_heap(entry);
    return result;
}

static inline bool pas_small_large_map_entry_can_create(pas_large_map_entry entry)
{
    pas_small_large_map_entry result;
    result = pas_small_large_map_entry_create(entry);
    return entry.begin == pas_small_large_map_entry_begin(result)
        && entry.end == pas_small_large_map_entry_end(result)
        && entry.heap == pas_small_large_map_entry_heap(result);
}

PAS_END_EXTERN_C;

#endif /* PAS_SMALL_LARGE_MAP_ENTRY_H */

