/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 16, 2025.
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

#include "pas_physical_memory_transaction.h"

void pas_physical_memory_transaction_construct(pas_physical_memory_transaction* transaction)
{
    pas_zero_memory(transaction, sizeof(pas_physical_memory_transaction));
}

void pas_physical_memory_transaction_begin(pas_physical_memory_transaction* transaction)
{
    PAS_ASSERT(!transaction->lock_held);
    
    if (!transaction->lock_to_acquire_next_time)
        return;
    
    pas_lock_lock(transaction->lock_to_acquire_next_time);
    transaction->lock_held = transaction->lock_to_acquire_next_time;
    transaction->lock_to_acquire_next_time = NULL;
}

bool pas_physical_memory_transaction_end(pas_physical_memory_transaction* transaction)
{
    if (transaction->lock_held) {
        pas_lock_unlock(transaction->lock_held);
        transaction->lock_held = NULL;
    }
    
    return !transaction->lock_to_acquire_next_time;
}

void pas_physical_memory_transaction_did_fail_to_acquire_lock(
    pas_physical_memory_transaction* transaction,
    pas_lock* lock_ptr)
{
    PAS_ASSERT(lock_ptr);
    PAS_ASSERT(lock_ptr != transaction->lock_held);
    
    if (transaction->lock_to_acquire_next_time)
        return;
    
    transaction->lock_to_acquire_next_time = lock_ptr;
}

#endif /* LIBPAS_ENABLED */
