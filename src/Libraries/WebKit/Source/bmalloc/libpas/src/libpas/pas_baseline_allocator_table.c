/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 22, 2021.
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

#include "pas_baseline_allocator_table.h"

#include "pas_heap_lock.h"
#include "pas_immortal_heap.h"
#include "pas_internal_config.h"
#include "pas_random.h"
#include <pthread.h>

pas_baseline_allocator* pas_baseline_allocator_table;
uint64_t pas_num_baseline_allocator_evictions = 0;
unsigned pas_baseline_allocator_table_bound = PAS_NUM_BASELINE_ALLOCATORS;

static void initialize(void)
{
    pas_baseline_allocator* table;
    size_t index;
    pas_heap_lock_lock();
    table = pas_immortal_heap_allocate(
        PAS_NUM_BASELINE_ALLOCATORS * sizeof(pas_baseline_allocator),
        "pas_baseline_allocator_table",
        pas_object_allocation);
    for (index = PAS_NUM_BASELINE_ALLOCATORS; index--;)
        table[index] = PAS_BASELINE_ALLOCATOR_INITIALIZER;
    pas_store_store_fence();
    pas_baseline_allocator_table = table;
    pas_heap_lock_unlock();
}

void pas_baseline_allocator_table_initialize_if_necessary(void)
{
    static pthread_once_t once_control = PTHREAD_ONCE_INIT;
    pthread_once(&once_control, initialize);
}

unsigned pas_baseline_allocator_table_get_random_index(void)
{
    return pas_get_fast_random(PAS_MIN(PAS_NUM_BASELINE_ALLOCATORS, pas_baseline_allocator_table_bound));
}

bool pas_baseline_allocator_table_for_all(pas_allocator_scavenge_action action)
{
    size_t index;
    bool result;

    if (!pas_baseline_allocator_table)
        return false;

    result = false;

    for (index = PAS_NUM_BASELINE_ALLOCATORS; index--;) {
        pas_baseline_allocator* allocator;

        allocator = pas_baseline_allocator_table + index;

        pas_lock_lock(&allocator->lock);
        result |= pas_local_allocator_scavenge(&allocator->u.allocator, action);
        pas_lock_unlock(&allocator->lock);
    }

    return result;
}

#endif /* LIBPAS_ENABLED */
