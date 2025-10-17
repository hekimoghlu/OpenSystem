/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 22, 2023.
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
#ifndef PAS_ALLOCATION_CALLBACKS_H
#define PAS_ALLOCATION_CALLBACKS_H

#include "pas_allocation_kind.h"
#include "pas_heap_kind.h"
#include "pas_log.h"

PAS_BEGIN_EXTERN_C;

typedef void(*pas_allocation_callback_type)(
    void* resulting_base,
    size_t size,
    pas_heap_kind heap_kind,
    const char* name,
    pas_allocation_kind allocation_kind);
typedef void(*pas_deallocation_callback_type)(
    void* base,
    size_t size, /* This is zero for non-free heaps like utility. */
    pas_heap_kind heap_kind,
    pas_allocation_kind allocation_kind);

PAS_API extern pas_allocation_callback_type pas_allocation_callback;
PAS_API extern pas_deallocation_callback_type pas_deallocation_callback;

static inline void pas_did_allocate(
    void* resulting_base,
    size_t size,
    pas_heap_kind heap_kind,
    const char* name,
    pas_allocation_kind allocation_kind)
{
    static const bool verbose = false;

    if (verbose) {
        pas_log("Doing pas_did_allocate with size = %zu, heap_kind = %s, name = %s, "
                "allocation_kind = %s.\n",
                size, pas_heap_kind_get_string(heap_kind), name,
                pas_allocation_kind_get_string(allocation_kind));
    }
    
    if (pas_allocation_callback && resulting_base)
        pas_allocation_callback(resulting_base, size, heap_kind, name, allocation_kind);
}

static inline void pas_will_deallocate(
    void* base,
    size_t size, /* This is zero for non-free heaps like utility. */
    pas_heap_kind heap_kind,
    pas_allocation_kind allocation_kind)
{
    if (pas_deallocation_callback && base)
        pas_deallocation_callback(base, size, heap_kind, allocation_kind);
}

PAS_END_EXTERN_C;

#endif /* PAS_ALLOCATION_CALLBACKS_H */

