/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 23, 2023.
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

#include "pas_page_header_table.h"

#include "pas_log.h"
#include "pas_utility_heap.h"

static const bool verbose = false;

pas_page_base* pas_page_header_table_add(pas_page_header_table* table,
                                         size_t page_size,
                                         size_t header_size,
                                         void* boundary)
{
    pas_page_base* page_base;

    uintptr_t boundary_int = (uintptr_t)boundary;
    PAS_PROFILE(PAGE_HEADER_TABLE_ADD, boundary_int);
    boundary = (void*)boundary_int;

    if (verbose)
        pas_log("Adding page header for boundary = %p.\n", boundary);
    
    PAS_ASSERT(pas_round_down_to_power_of_2((uintptr_t)boundary, page_size)
               == (uintptr_t)boundary);

    PAS_ASSERT(page_size == table->page_size);

    /* This protects against leaks. */
    PAS_ASSERT(!pas_page_header_table_get_for_boundary(table, page_size, boundary));

    /* We allocate two slots before the pas_page_base. The one is used for storing boundary.
       Another is not used, just allocated to align the page with 16byte alignment, which is
       required since it includes 16byte aligned data structures. */
    page_base = (pas_page_base*)(
        (void**)pas_utility_heap_allocate_with_alignment(
            sizeof(void*) * 2 + header_size, 16, "pas_page_header_table/header") + 2);

    if (verbose)
        pas_log("created page header at %p\n", page_base);

    *pas_page_header_table_get_boundary_ptr(table, page_size, page_base) = boundary;

    pas_lock_free_read_ptr_ptr_hashtable_set(
        &table->hashtable,
        pas_page_header_table_hash,
        (void*)page_size,
        boundary,
        page_base,
        pas_lock_free_read_ptr_ptr_hashtable_set_maybe_existing);

    return page_base;
}

void pas_page_header_table_remove(pas_page_header_table* table,
                                  size_t page_size,
                                  pas_page_base* page_base)
{
    void* boundary;
    
    PAS_ASSERT(page_size == table->page_size);

    if (verbose)
        pas_log("destroying page header at %p\n", page_base);

    boundary = pas_page_header_table_get_boundary(table, page_size, page_base);

    if (verbose)
        pas_log("Removing page header for boundary = %p.\n", boundary);

    PAS_ASSERT(pas_round_down_to_power_of_2((uintptr_t)boundary, page_size)
               == (uintptr_t)boundary);

    pas_lock_free_read_ptr_ptr_hashtable_set(
        &table->hashtable,
        pas_page_header_table_hash,
        (void*)page_size,
        boundary,
        NULL,
        pas_lock_free_read_ptr_ptr_hashtable_set_maybe_existing);

    pas_utility_heap_deallocate(
        pas_page_header_table_get_boundary_ptr(table, page_size, page_base));
}

#endif /* LIBPAS_ENABLED */
