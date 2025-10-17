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
#ifndef PAS_LOCK_FREE_READ_PTR_PTR_HASHTABLE_H
#define PAS_LOCK_FREE_READ_PTR_PTR_HASHTABLE_H

#include "pas_log.h"
#include "pas_utils.h"
#include <unistd.h>

PAS_BEGIN_EXTERN_C;

struct pas_lock_free_read_ptr_ptr_hashtable;
struct pas_lock_free_read_ptr_ptr_hashtable_table;
typedef struct pas_lock_free_read_ptr_ptr_hashtable pas_lock_free_read_ptr_ptr_hashtable;
typedef struct pas_lock_free_read_ptr_ptr_hashtable_table pas_lock_free_read_ptr_ptr_hashtable_table;

struct pas_lock_free_read_ptr_ptr_hashtable {
    pas_lock_free_read_ptr_ptr_hashtable_table* table;
};

struct PAS_ALIGNED(sizeof(pas_pair)) pas_lock_free_read_ptr_ptr_hashtable_table {
    pas_lock_free_read_ptr_ptr_hashtable_table* previous;
    unsigned table_size;
    unsigned table_mask;
    unsigned key_count;
    pas_pair array[1];
};

#define PAS_LOCK_FREE_READ_PTR_PTR_HASHTABLE_INITIALIZER \
    ((pas_lock_free_read_ptr_ptr_hashtable){ \
         .table = NULL \
     })

#define PAS_LOCK_FREE_READ_PTR_PTR_HASHTABLE_ENABLE_COLLISION_COUNT 0

#if PAS_LOCK_FREE_READ_PTR_PTR_HASHTABLE_ENABLE_COLLISION_COUNT
PAS_API extern uint64_t pas_lock_free_read_ptr_ptr_hashtable_collision_count;
#endif /* PAS_LOCK_FREE_READ_PTR_PTR_HASHTABLE_ENABLE_COLLISION_COUNT */

static PAS_ALWAYS_INLINE void* pas_lock_free_read_ptr_ptr_hashtable_find(
    pas_lock_free_read_ptr_ptr_hashtable* hashtable,
    unsigned (*hash_key)(const void* key, void* arg),
    void* hash_arg,
    const void* key)
{
    pas_lock_free_read_ptr_ptr_hashtable_table* table;

    table = hashtable->table;
    if (!table)
        return NULL;
    
    for (unsigned hash = hash_key(key, hash_arg); ; ++hash) {
        unsigned index;
        pas_pair* entry;
        uintptr_t loaded_key;

        index = hash & table->table_mask;

        entry = table->array + index;

        /* It's crazy, but we *can* load the two words separately. They do have to happen in the
           right order, though. Otherwise it's possible to get a NULL value even though the key
           was already set.
        
           NOTE: Perf would be better if we did an atomic pair read on Apple Silicon. Then we'd
           avoid the synthetic pointer chase. */
        loaded_key = pas_pair_low(*entry);
        if (pas_compare_ptr_opaque(loaded_key, (uintptr_t)key))
            return (void*)pas_pair_high(entry[pas_depend(loaded_key)]);

        if (loaded_key == UINTPTR_MAX)
            return NULL;

#if PAS_LOCK_FREE_READ_PTR_PTR_HASHTABLE_ENABLE_COLLISION_COUNT
        for (;;) {
            uint64_t old_collision_count;
            uint64_t new_collision_count;

            old_collision_count = pas_lock_free_read_ptr_ptr_hashtable_collision_count;
            new_collision_count = old_collision_count + 1;

            if (pas_compare_and_swap_uint64_weak(
                    &pas_lock_free_read_ptr_ptr_hashtable_collision_count,
                    old_collision_count, new_collision_count)) {
                if (!(new_collision_count % 10000))
                    pas_log("%d: Saw %llu collisions.\n", getpid(), new_collision_count);
                break;
            }
        }
#endif /* PAS_LOCK_FREE_READ_PTR_PTR_HASHTABLE_ENABLE_COLLISION_COUNT */
    }
}

enum pas_lock_free_read_ptr_ptr_hashtable_set_mode {
    pas_lock_free_read_ptr_ptr_hashtable_add_new,
    pas_lock_free_read_ptr_ptr_hashtable_set_maybe_existing
};

typedef enum pas_lock_free_read_ptr_ptr_hashtable_set_mode pas_lock_free_read_ptr_ptr_hashtable_set_mode;

PAS_API void pas_lock_free_read_ptr_ptr_hashtable_set(
    pas_lock_free_read_ptr_ptr_hashtable* hashtable,
    unsigned (*hash_key)(const void* key, void* arg),
    void* hash_arg,
    const void* key,
    const void* value,
    pas_lock_free_read_ptr_ptr_hashtable_set_mode set_mode);

static inline unsigned pas_lock_free_read_ptr_ptr_hashtable_size(
    pas_lock_free_read_ptr_ptr_hashtable* hashtable)
{
    return hashtable->table->key_count;
}

PAS_END_EXTERN_C;

#endif /* PAS_LOCK_FREE_READ_PTR_PTR_HASHTABLE_H */

