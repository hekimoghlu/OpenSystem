/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 5, 2024.
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

#include "pas_ptr_worklist.h"

void pas_ptr_worklist_construct(pas_ptr_worklist* worklist)
{
    pas_ptr_hash_set_construct(&worklist->seen);
    worklist->worklist = NULL;
    worklist->worklist_size = 0;
    worklist->worklist_capacity = 0;
}

void pas_ptr_worklist_destruct(pas_ptr_worklist* worklist,
                               const pas_allocation_config* allocation_config)
{
    pas_ptr_hash_set_destruct(&worklist->seen, allocation_config);
    if (worklist->worklist) {
        allocation_config->deallocate(worklist->worklist,
                                      sizeof(void*) * worklist->worklist_capacity,
                                      pas_object_allocation,
                                      allocation_config->arg);
    }
}

bool pas_ptr_worklist_push(pas_ptr_worklist* worklist,
                           void* ptr,
                           const pas_allocation_config* allocation_config)
{
    pas_ptr_hash_set_add_result add_result;

    if (!ptr)
        return false;

    add_result = pas_ptr_hash_set_add(&worklist->seen, ptr, NULL, allocation_config);
    if (!add_result.is_new_entry)
        return false;

    *add_result.entry = ptr;

    if (worklist->worklist_size >= worklist->worklist_capacity) {
        void* new_worklist;
        size_t new_worklist_capacity;
        
        PAS_ASSERT(worklist->worklist_size == worklist->worklist_capacity);

        new_worklist_capacity = (worklist->worklist_capacity + 1) << 1;
        new_worklist = allocation_config->allocate(sizeof(void*) * new_worklist_capacity,
                                                   "pas_ptr_worklist/worklist",
                                                   pas_object_allocation,
                                                   allocation_config->arg);
        memcpy(new_worklist, worklist->worklist, sizeof(void*) * worklist->worklist_size);
        allocation_config->deallocate(worklist->worklist,
                                      sizeof(void*) * worklist->worklist_capacity,
                                      pas_object_allocation,
                                      allocation_config->arg);
        worklist->worklist = new_worklist;
        worklist->worklist_capacity = new_worklist_capacity;

        PAS_ASSERT(worklist->worklist_size < worklist->worklist_capacity);
    }

    worklist->worklist[worklist->worklist_size++] = ptr;
    return true;
}

void* pas_ptr_worklist_pop(pas_ptr_worklist* worklist)
{
    if (!worklist->worklist_size)
        return NULL;

    return worklist->worklist[--worklist->worklist_size];
}

#endif /* LIBPAS_ENABLED */
