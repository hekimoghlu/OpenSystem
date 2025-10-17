/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 30, 2025.
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
#ifndef PAS_SIMPLE_FREE_HEAP_HELPERS_H
#define PAS_SIMPLE_FREE_HEAP_HELPERS_H

#include "pas_alignment.h"
#include "pas_allocation_kind.h"
#include "pas_allocation_result.h"
#include "pas_heap_kind.h"

PAS_BEGIN_EXTERN_C;

struct pas_simple_large_free_heap;
struct pas_large_free_heap_config;
typedef struct pas_simple_large_free_heap pas_simple_large_free_heap;
typedef struct pas_large_free_heap_config pas_large_free_heap_config;

PAS_API pas_allocation_result
pas_simple_free_heap_helpers_try_allocate_with_manual_alignment(
    pas_simple_large_free_heap* free_heap,
    void (*initialize_config)(pas_large_free_heap_config* config),
    pas_heap_kind heap_kind,
    size_t size,
    pas_alignment alignment,
    const char* name,
    pas_allocation_kind allocation_kind,
    size_t* num_allocated_object_bytes,
    size_t* num_allocated_object_bytes_peak);

PAS_API void pas_simple_free_heap_helpers_deallocate(
    pas_simple_large_free_heap* free_heap,
    void (*initialize_config)(pas_large_free_heap_config* config),
    pas_heap_kind heap_kind,
    void* ptr,
    size_t size,
    pas_allocation_kind allocation_kind,
    size_t* num_allocated_object_bytes);

PAS_END_EXTERN_C;

#endif /* PAS_SIMPLE_FREE_HEAP_HELPERS_H */

