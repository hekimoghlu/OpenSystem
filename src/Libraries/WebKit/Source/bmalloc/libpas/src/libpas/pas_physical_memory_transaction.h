/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 30, 2023.
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
#ifndef PAS_PHYSICAL_MEMORY_TRANSACTION_H
#define PAS_PHYSICAL_MEMORY_TRANSACTION_H

#include "pas_lock.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_physical_memory_transaction;
typedef struct pas_physical_memory_transaction pas_physical_memory_transaction;

/* How to use this: if you are calling API that needs to hold the heap lock and perform commits
   then you need to make sure you wrap your heap lock acquisition with a commit transaction like
   so:
   
   pas_physical_memory_transaction physical_memory_transaction;
   pas_physical_memory_transaction_construct(&physical_memory_transaction);
   do {
       pas_physical_memory_transaction_begin(&physical_memory_transaction);
       pas_heap_lock_lock();
       
       do things
       
       pas_heap_lock_unlock();
   } while (!pas_physical_memory_transaction_end(&physical_memory_transaction));
   
   This ensures that if the things you want to do need to acquire a commit lock, then they can
   arrange for that commit lock to be contended for while the heap lock is not held.

   One of the properties that we get from this is that it's always safe to acquire the heap lock
   if it is not already held. */

struct pas_physical_memory_transaction {
    pas_lock* lock_to_acquire_next_time;
    pas_lock* lock_held;
};

PAS_API void pas_physical_memory_transaction_construct(pas_physical_memory_transaction* transaction);

PAS_API void pas_physical_memory_transaction_begin(pas_physical_memory_transaction* transaction);

PAS_API bool pas_physical_memory_transaction_end(pas_physical_memory_transaction* transaction);

PAS_API void pas_physical_memory_transaction_did_fail_to_acquire_lock(
    pas_physical_memory_transaction* transaction,
    pas_lock* lock_ptr);

PAS_END_EXTERN_C;

#endif /* PAS_PHYSICAL_MEMORY_TRANSACTION_H */

