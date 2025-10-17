/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 16, 2024.
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
#include "pas_config.h"

#if LIBPAS_ENABLED

#include "pas_lock_free_read_ptr_ptr_hashtable.h"

#include "pas_bootstrap_free_heap.h"
#include "pas_hashtable.h"
#include "pas_heap_lock.h"

#if PAS_LOCK_FREE_READ_PTR_PTR_HASHTABLE_ENABLE_COLLISION_COUNT
uint64_t pas_lock_free_read_ptr_ptr_hashtable_collision_count;
#endif /* PAS_LOCK_FREE_READ_PTR_PTR_HASHTABLE_ENABLE_COLLISION_COUNT */

void pas_lock_free_read_ptr_ptr_hashtable_set(
    pas_lock_free_read_ptr_ptr_hashtable* hashtable,
    unsigned (*hash_key)(const void* key, void* arg),
    void* hash_arg,
    const void* key,
    const void* value,
    pas_lock_free_read_ptr_ptr_hashtable_set_mode set_mode)
{
    pas_lock_free_read_ptr_ptr_hashtable_table* table;
    unsigned hash;

    PAS_ASSERT(key);
    pas_heap_lock_assert_held();

    table = hashtable->table;

    if (!table || table->key_count * PAS_HASHTABLE_MAX_LOAD >= table->table_size) {
        unsigned new_size;
        size_t new_byte_size;
        pas_lock_free_read_ptr_ptr_hashtable_table* new_table;
        unsigned old_index;
        unsigned new_table_mask;
        
        if (table)
            new_size = table->table_size * 2;
        else
            new_size = PAS_HASHTABLE_MIN_SIZE;

        PAS_ASSERT(pas_is_power_of_2(new_size));

        new_table_mask = new_size - 1;
        new_byte_size =
            PAS_OFFSETOF(pas_lock_free_read_ptr_ptr_hashtable_table, array) +
            sizeof(pas_pair) * new_size;
        new_table = (void*)pas_bootstrap_free_heap_allocate_with_alignment(
            new_byte_size,
            pas_alignment_create_traditional(sizeof(pas_pair)),
            "pas_lock_free_read_ptr_ptr_hashtable/table",
            pas_object_allocation).begin;

        memset(new_table, -1, new_byte_size);

        new_table->previous = table;

        if (table) {
            for (old_index = 0; old_index < table->table_size; ++old_index) {
                pas_pair* old_entry;
                unsigned hash;
                
                old_entry = table->array + old_index;
                if (pas_pair_low(*old_entry) == UINTPTR_MAX)
                    continue;
                
                for (hash = hash_key((const void*)pas_pair_low(*old_entry), hash_arg); ; ++hash) {
                    unsigned new_index;
                    pas_pair* new_entry;
                    
                    new_index = hash & new_table_mask;
                    new_entry = new_table->array + new_index;
                    
                    if (pas_pair_low(*new_entry) == UINTPTR_MAX) {
                        /* Can do this without atomics because the old table is frozen, the old_entry
                           is frozen if non-null even if the old table wasn't, and the new table is
                           still private to this thread. */
                        *new_entry = *old_entry;
                        break;
                    }
                }
            }
        }
        
        new_table->table_size = new_size;
        new_table->table_mask = new_table_mask;
        new_table->key_count = table ? table->key_count : 0;

        pas_fence();
        
        hashtable->table = new_table;
        table = new_table;
    }

    for (hash = hash_key(key, hash_arg); ; ++hash) {
        unsigned index;
        pas_pair* entry;

        index = hash & table->table_mask;
        entry = table->array + index;

        if (pas_pair_low(*entry) == UINTPTR_MAX) {
            pas_atomic_store_pair(entry, pas_pair_create((uintptr_t)key,
                                                         (uintptr_t)value));
            table->key_count++;
            break;
        }

        if (pas_pair_low(*entry) == (uintptr_t)key) {
            PAS_ASSERT(set_mode == pas_lock_free_read_ptr_ptr_hashtable_set_maybe_existing);
            *entry = pas_pair_create((uintptr_t)key, (uintptr_t)value);
            break;
        }

        PAS_ASSERT((const void*)pas_pair_low(*entry) != key);
    }
}

#endif /* LIBPAS_ENABLED */
