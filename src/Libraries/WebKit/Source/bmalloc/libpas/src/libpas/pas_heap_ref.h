/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 2, 2023.
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
#ifndef PAS_HEAP_REF_H
#define PAS_HEAP_REF_H

#include "pas_config.h"
#include "pas_heap_ref_kind.h"
#include "pas_segregated_heap_lookup_kind.h"
#include "pas_utils.h"

#include "pas_heap_ref_prefix.h"

PAS_BEGIN_EXTERN_C;

/* You can use a pas_heap_ref for different kinds of heap configurations. Each of those heap
   configurations will have a distinct set of entrypoints for allocation, deallocation, and
   introspection. For example, thingy_deallocate will deallocate with the thingy_heap_config,
   while pxi_deallocate will deallocate with the pxi_heap_config. */

#define pas_heap __pas_heap
#define pas_heap_ref __pas_heap_ref
#define pas_heap_type __pas_heap_type

struct pas_heap_config;
struct pas_heap_runtime_config;
typedef struct pas_heap_config pas_heap_config;
typedef struct pas_heap_runtime_config pas_heap_runtime_config;

PAS_API pas_heap* pas_ensure_heap_slow(pas_heap_ref* heap_ref,
                                       pas_heap_ref_kind heap_ref_kind,
                                       const pas_heap_config* config,
                                       pas_heap_runtime_config* runtime_config);

static inline pas_heap* pas_ensure_heap(pas_heap_ref* heap_ref,
                                        pas_heap_ref_kind heap_ref_kind,
                                        const pas_heap_config* config,
                                        pas_heap_runtime_config* runtime_config)
{
    pas_heap* heap = heap_ref->heap;
    if (PAS_LIKELY(heap))
        return heap;
    return pas_ensure_heap_slow(heap_ref, heap_ref_kind, config, runtime_config);
}

PAS_END_EXTERN_C;

#endif /* PAS_HEAP_REF_H */
