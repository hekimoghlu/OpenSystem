/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 3, 2023.
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
#ifndef PAS_PTR_WORKLIST_H
#define PAS_PTR_WORKLIST_H

#include "pas_ptr_hash_set.h"

PAS_BEGIN_EXTERN_C;

struct pas_ptr_worklist;
typedef struct pas_ptr_worklist pas_ptr_worklist;

struct pas_ptr_worklist {
    pas_ptr_hash_set seen;
    void** worklist;
    size_t worklist_size;
    size_t worklist_capacity;
};

PAS_API void pas_ptr_worklist_construct(pas_ptr_worklist* worklist);

PAS_API void pas_ptr_worklist_destruct(pas_ptr_worklist* worklist,
                                       const pas_allocation_config* allocation_config);

/* Returns true if this is a new worklist entry. Always returns false if you try to push NULL. */
PAS_API bool pas_ptr_worklist_push(pas_ptr_worklist* worklist,
                                   void* ptr,
                                   const pas_allocation_config* allocation_config);

PAS_API void* pas_ptr_worklist_pop(pas_ptr_worklist* worklist);

PAS_END_EXTERN_C;

#endif /* PAS_PTR_WORKLIST_H */

