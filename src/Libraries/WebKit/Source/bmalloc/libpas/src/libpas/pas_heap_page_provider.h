/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 20, 2023.
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
#ifndef PAS_HEAP_PAGE_PROVIDER_H
#define PAS_HEAP_PAGE_PROVIDER_H

#include "pas_alignment.h"
#include "pas_allocation_result.h"
#include "pas_heap_ref.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_physical_memory_transaction;
typedef struct pas_physical_memory_transaction pas_physical_memory_transaction;

/* Heap and transaction can be NULL, in which case, assume the worst. Most implementations of this
   function never need the transaction or the heap. */
typedef pas_allocation_result (*pas_heap_page_provider)(
    size_t size,
    pas_alignment alignment,
    const char* name,
    pas_heap* heap, /* Can be NULL. */
    pas_physical_memory_transaction* transaction, /* Can be NULL. */
    void *arg);

PAS_END_EXTERN_C;

#endif /* PAS_HEAP_PAGE_PROVIDER_H */

