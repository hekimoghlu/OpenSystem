/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 11, 2022.
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

#include "pas_basic_heap_config_enumerator_data.h"

#include "pas_page_header_table.h"

bool pas_basic_heap_config_enumerator_data_add_page_header_table(
    pas_basic_heap_config_enumerator_data* data,
    pas_enumerator* enumerator,
    pas_page_header_table* page_header_table)
{
    static const bool verbose = false;
    
    pas_lock_free_read_ptr_ptr_hashtable_table* table;
    size_t index;

    if (!page_header_table)
        return false;
    
    if (!page_header_table->hashtable.table)
        return true;
    
    if (verbose)
        pas_log("Have a page header hashtable at %p.\n", page_header_table->hashtable.table);
    
    table = pas_enumerator_read(
        enumerator, page_header_table->hashtable.table,
        PAS_OFFSETOF(pas_lock_free_read_ptr_ptr_hashtable_table, array));
    if (!table)
        return false;
    
    if (verbose)
        pas_log("The table has size %u.\n", table->table_size);
    
    table = pas_enumerator_read(
        enumerator, page_header_table->hashtable.table,
        PAS_OFFSETOF(pas_lock_free_read_ptr_ptr_hashtable_table, array)
        + sizeof(pas_pair) * table->table_size);
    if (!table)
        return false;
    
    for (index = table->table_size; index--;) {
        pas_pair* pair;
        pas_ptr_hash_map_entry entry;
        
        pair = table->array + index;
        
        if (pas_pair_low(*pair) == UINTPTR_MAX)
            continue;
        
        entry.key = (void*)pas_pair_low(*pair);
        entry.value = (void*)pas_pair_high(*pair);
        
        pas_ptr_hash_map_add_new(
            &data->page_header_table, entry, NULL, &enumerator->allocation_config);
    }
    
    return true;
}

#endif /* LIBPAS_ENABLED */
