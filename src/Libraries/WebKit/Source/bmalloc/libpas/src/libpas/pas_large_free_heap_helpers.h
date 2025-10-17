/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 22, 2023.
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
#ifndef PAS_LARGE_FREE_HEAP_HELPERS_H
#define PAS_LARGE_FREE_HEAP_HELPERS_H

#include "pas_alignment.h"
#include "pas_allocation_kind.h"
#include "pas_allocation_result.h"
#include "pas_heap_summary.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

struct pas_fast_large_free_heap;
typedef struct pas_fast_large_free_heap pas_fast_large_free_heap;

PAS_API extern bool pas_large_utility_free_heap_talks_to_large_sharing_pool;

typedef pas_allocation_result (*pas_large_free_heap_helpers_memory_source)(
    size_t size,
    pas_alignment alignment,
    const char* name,
    pas_allocation_kind allocation_kind);

PAS_API void* pas_large_free_heap_helpers_try_allocate_with_alignment(
    pas_fast_large_free_heap* heap,
    pas_large_free_heap_helpers_memory_source memory_source,
    size_t* num_allocated_object_bytes_ptr,
    size_t* num_allocated_object_bytes_peak_ptr,
    size_t size,
    pas_alignment alignment,
    const char* name);

PAS_API void pas_large_free_heap_helpers_deallocate(
    pas_fast_large_free_heap* heap,
    pas_large_free_heap_helpers_memory_source memory_source,
    size_t* num_allocated_object_bytes_ptr,
    void* ptr,
    size_t size);

PAS_API pas_heap_summary pas_large_free_heap_helpers_compute_summary(
    pas_fast_large_free_heap* heap,
    size_t* num_allocated_object_bytes_ptr);

PAS_END_EXTERN_C;

#endif /* PAS_LARGE_FREE_HEAP_HELPERS_H */

