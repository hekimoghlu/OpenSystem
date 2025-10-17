/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 18, 2022.
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
#ifndef PAS_FIRST_LEVEL_TINY_LARGE_MAP_ENTRY_H
#define PAS_FIRST_LEVEL_TINY_LARGE_MAP_ENTRY_H

#include "pas_hashtable.h"
#include "pas_internal_config.h"
#include "pas_tiny_large_map_entry.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

PAS_CREATE_HASHTABLE(pas_tiny_large_map_second_level_hashtable,
                     pas_tiny_large_map_entry,
                     pas_large_map_key);

typedef uintptr_t pas_first_level_tiny_large_map_key;

struct pas_first_level_tiny_large_map_entry;
typedef struct pas_first_level_tiny_large_map_entry pas_first_level_tiny_large_map_entry;

struct pas_first_level_tiny_large_map_entry {
    uintptr_t base;
    pas_tiny_large_map_second_level_hashtable* hashtable;
};

static inline pas_first_level_tiny_large_map_entry
pas_first_level_tiny_large_map_entry_create_empty(void)
{
    pas_first_level_tiny_large_map_entry result;
    result.base = 0;
    result.hashtable = NULL;
    return result;
}

static inline pas_first_level_tiny_large_map_entry
pas_first_level_tiny_large_map_entry_create_deleted(void)
{
    pas_first_level_tiny_large_map_entry result;
    result.base = 1;
    result.hashtable = NULL;
    return result;
}

static inline bool
pas_first_level_tiny_large_map_entry_is_empty_or_deleted(
    pas_first_level_tiny_large_map_entry entry)
{
    return !entry.hashtable;
}

static inline bool
pas_first_level_tiny_large_map_entry_is_empty(
    pas_first_level_tiny_large_map_entry entry)
{
    return !entry.base;
}

static inline bool
pas_first_level_tiny_large_map_entry_is_deleted(
    pas_first_level_tiny_large_map_entry entry)
{
    return entry.base == 1;
}

static inline uintptr_t pas_first_level_tiny_large_map_entry_get_key(
    pas_first_level_tiny_large_map_entry entry)
{
    return entry.base;
}

static inline unsigned pas_first_level_tiny_large_map_key_get_hash(uintptr_t key)
{
    return pas_large_object_hash(key);
}

static inline bool pas_first_level_tiny_large_map_key_is_equal(uintptr_t a, uintptr_t b)
{
    return a == b;
}

PAS_END_EXTERN_C;

#endif /* PAS_FIRST_LEVEL_TINY_LARGE_MAP_ENTRY_H */

