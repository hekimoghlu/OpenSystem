/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 18, 2024.
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

#include "pas_heap_table.h"

#include "pas_heap_lock.h"
#include "pas_bootstrap_free_heap.h"

pas_large_heap** pas_heap_table = NULL;
unsigned pas_heap_table_bump_index = 0;

void pas_heap_table_try_allocate_index(pas_large_heap* heap)
{
    pas_heap_lock_assert_held();

    if (!pas_heap_table) {
        PAS_ASSERT(!pas_heap_table_bump_index);
        
        pas_heap_table = pas_bootstrap_free_heap_allocate_simple(
            sizeof(pas_large_heap*) * PAS_HEAP_TABLE_SIZE,
            "pas_heap_table",
            pas_delegate_allocation);
    }

    if (pas_heap_table_bump_index >= PAS_HEAP_TABLE_SIZE) {
        PAS_ASSERT(pas_heap_table_bump_index == PAS_HEAP_TABLE_SIZE);
        heap->table_state = pas_heap_table_state_failed;
        return;
    }

    heap->index = pas_heap_table_bump_index++;
    pas_heap_table[heap->index] = heap;
    heap->table_state = pas_heap_table_state_has_index;
}

#endif /* LIBPAS_ENABLED */
