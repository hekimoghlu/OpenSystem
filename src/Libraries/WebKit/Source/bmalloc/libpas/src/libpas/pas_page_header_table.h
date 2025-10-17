/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 29, 2022.
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
#ifndef PAS_PAGE_HEADER_TABLE_H
#define PAS_PAGE_HEADER_TABLE_H

#include "pas_lock_free_read_ptr_ptr_hashtable.h"

PAS_BEGIN_EXTERN_C;

/* The page header table is the slow way of getting a page header. We use it for medium pages and
   for pages that need out-of-line headers. It's necessary to have a separate page header table for
   each page size. */

struct pas_page_base;
struct pas_page_header_table;
typedef struct pas_page_base pas_page_base;
typedef struct pas_page_header_table pas_page_header_table;

struct pas_page_header_table {
    size_t page_size;
    pas_lock_free_read_ptr_ptr_hashtable hashtable;
};

#define PAS_PAGE_HEADER_TABLE_INITIALIZER(passed_page_size) \
    { \
         .page_size = (passed_page_size), \
         .hashtable = PAS_LOCK_FREE_READ_PTR_PTR_HASHTABLE_INITIALIZER \
    }

static inline unsigned pas_page_header_table_hash(const void* key, void* arg)
{
    size_t page_size;

    page_size = (size_t)arg;

    return pas_hash32((unsigned)((uintptr_t)key / page_size));
}

PAS_API pas_page_base* pas_page_header_table_add(pas_page_header_table* table,
                                                 size_t page_size,
                                                 size_t header_size,
                                                 void* boundary);

PAS_API void pas_page_header_table_remove(pas_page_header_table* table,
                                          size_t page_size,
                                          pas_page_base* page_base);

static PAS_ALWAYS_INLINE void** pas_page_header_table_get_boundary_ptr(pas_page_header_table* table,
                                                                       size_t page_size,
                                                                       pas_page_base* page_base)
{
    PAS_TESTING_ASSERT(page_size == table->page_size);

    return ((void**)page_base) - 2;
}

static PAS_ALWAYS_INLINE void* pas_page_header_table_get_boundary(pas_page_header_table* table,
                                                                  size_t page_size,
                                                                  pas_page_base* page_base)
{
    return *pas_page_header_table_get_boundary_ptr(table, page_size, page_base);
}

static PAS_ALWAYS_INLINE pas_page_base*
pas_page_header_table_get_for_boundary(pas_page_header_table* table,
                                       size_t page_size,
                                       void* boundary)
{
    uintptr_t begin = (uintptr_t)boundary;

    PAS_TESTING_ASSERT(page_size == table->page_size);
    PAS_TESTING_ASSERT(pas_round_down_to_power_of_2(begin, page_size)
                       == begin);

    PAS_PROFILE(PAGE_HEADER_TABLE_GET, begin);
    boundary = (void*)begin;
    return (pas_page_base*)pas_lock_free_read_ptr_ptr_hashtable_find(
        &table->hashtable, pas_page_header_table_hash, (void*)page_size, boundary);
}

static PAS_ALWAYS_INLINE pas_page_base*
pas_page_header_table_get_for_address(pas_page_header_table* table,
                                      size_t page_size,
                                      void* address)
{
    void* boundary;

    PAS_TESTING_ASSERT(page_size == table->page_size);

    boundary = (void*)pas_round_down_to_power_of_2((uintptr_t)address, page_size);

    return pas_page_header_table_get_for_boundary(table, page_size, boundary);
}

PAS_END_EXTERN_C;

#endif /* PAS_PAGE_HEADER_TABLE_H */


