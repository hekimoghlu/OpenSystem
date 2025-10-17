/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 16, 2022.
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

#include "pas_debug_spectrum.h"

#include "pas_heap_lock.h"
#include "pas_immortal_heap.h"
#include "pas_large_utility_free_heap.h"
#include "pas_stream.h"

pas_ptr_hash_map pas_debug_spectrum = PAS_HASHTABLE_INITIALIZER;

void pas_debug_spectrum_add(
    void* key, pas_debug_spectrum_dump_key dump, uint64_t count)
{
    pas_ptr_hash_map_add_result add_result;
    pas_debug_spectrum_entry* entry;
    
    pas_heap_lock_assert_held();

    add_result = pas_ptr_hash_map_add(
        &pas_debug_spectrum, key, NULL, &pas_large_utility_free_heap_allocation_config);

    if (add_result.is_new_entry) {
        entry = pas_immortal_heap_allocate(
            sizeof(pas_debug_spectrum_entry),
            "pas_debug_spectrum_entry",
            pas_object_allocation);
        entry->dump = dump;
        entry->count = count;
        add_result.entry->key = key;
        add_result.entry->value = entry;
        return;
    }

    entry = add_result.entry->value;

    PAS_ASSERT(entry->dump == dump);
    entry->count += count;
}

void pas_debug_spectrum_dump(pas_stream* stream)
{
    unsigned index;

    pas_heap_lock_assert_held();

    for (index = 0; index < pas_debug_spectrum.table_size; ++index) {
        pas_ptr_hash_map_entry hash_entry;
        pas_debug_spectrum_entry* entry;

        hash_entry = pas_debug_spectrum.table[index];
        if (pas_ptr_hash_map_entry_is_empty_or_deleted(hash_entry))
            continue;

        entry = hash_entry.value;

        if (!entry->count)
            continue;

        entry->dump(stream, hash_entry.key);
        pas_stream_printf(stream, ": %llu\n", (unsigned long long)entry->count);
    }
}

void pas_debug_spectrum_reset(void)
{
    unsigned index;

    pas_heap_lock_assert_held();

    for (index = 0; index < pas_debug_spectrum.table_size; ++index) {
        pas_ptr_hash_map_entry hash_entry;
        pas_debug_spectrum_entry* entry;

        hash_entry = pas_debug_spectrum.table[index];
        if (pas_ptr_hash_map_entry_is_empty_or_deleted(hash_entry))
            continue;

        entry = hash_entry.value;
        entry->count = 0;
    }
}

#endif /* LIBPAS_ENABLED */
