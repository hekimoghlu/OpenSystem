/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 6, 2025.
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
#ifndef PAS_PTR_HASH_MAP_H
#define PAS_PTR_HASH_MAP_H

#include "pas_hashtable.h"

PAS_BEGIN_EXTERN_C;

struct pas_ptr_hash_map_entry;
typedef struct pas_ptr_hash_map_entry pas_ptr_hash_map_entry;

struct pas_ptr_hash_map_entry {
    void* key;
    void* value;
};

typedef void* pas_ptr_hash_map_key;

static inline pas_ptr_hash_map_entry pas_ptr_hash_map_entry_create_empty(void)
{
    pas_ptr_hash_map_entry result;
    result.key = (void*)UINTPTR_MAX;
    result.value = NULL;
    return result;
}

static inline pas_ptr_hash_map_entry pas_ptr_hash_map_entry_create_deleted(void)
{
    pas_ptr_hash_map_entry result;
    result.key = (void*)UINTPTR_MAX;
    result.value = (void*)(uintptr_t)1;
    return result;
}

static inline bool pas_ptr_hash_map_entry_is_empty_or_deleted(pas_ptr_hash_map_entry entry)
{
    if (entry.key == (void*)UINTPTR_MAX) {
        PAS_TESTING_ASSERT(entry.value <= (void*)(uintptr_t)1);
        return true;
    }
    return false;
}

static inline bool pas_ptr_hash_map_entry_is_empty(pas_ptr_hash_map_entry entry)
{
    return entry.key == (void*)UINTPTR_MAX
        && !entry.value;
}

static inline bool pas_ptr_hash_map_entry_is_deleted(pas_ptr_hash_map_entry entry)
{
    return entry.key == (void*)UINTPTR_MAX
        && entry.value == (void*)(uintptr_t)1;
}

static inline void* pas_ptr_hash_map_entry_get_key(pas_ptr_hash_map_entry entry)
{
    return entry.key;
}

static inline unsigned pas_ptr_hash_map_key_get_hash(pas_ptr_hash_map_key key)
{
    return pas_hash_ptr(key);
}

static inline bool pas_ptr_hash_map_key_is_equal(pas_ptr_hash_map_key a,
                                                 pas_ptr_hash_map_key b)
{
    return a == b;
}

PAS_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
PAS_CREATE_HASHTABLE(pas_ptr_hash_map,
                     pas_ptr_hash_map_entry,
                     pas_ptr_hash_map_key);
PAS_ALLOW_UNSAFE_BUFFER_USAGE_END

PAS_END_EXTERN_C;

#endif /* PAS_PTR_HASH_MAP_H */
