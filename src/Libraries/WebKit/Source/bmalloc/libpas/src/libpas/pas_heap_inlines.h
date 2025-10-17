/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 20, 2023.
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
#ifndef PAS_HEAP_INLINES_H
#define PAS_HEAP_INLINES_H

#include "pas_allocator_counts.h"
#include "pas_config.h"
#include "pas_heap.h"
#include "pas_log.h"
#include "pas_segregated_heap_inlines.h"

PAS_BEGIN_EXTERN_C;

PAS_API pas_segregated_size_directory*
pas_heap_ensure_size_directory_for_size_slow(
    pas_heap* heap,
    size_t size,
    size_t alignment,
    pas_size_lookup_mode force_size_lookup,
    const pas_heap_config* config,
    unsigned* cached_index);

static PAS_ALWAYS_INLINE pas_segregated_size_directory*
pas_heap_ensure_size_directory_for_size(
    pas_heap* heap,
    size_t size,
    size_t alignment,
    pas_size_lookup_mode force_size_lookup,
    pas_heap_config config,
    unsigned* cached_index,
    pas_allocator_counts* counts)
{
    static const bool verbose = PAS_SHOULD_LOG(PAS_LOG_HEAP_INFRASTRUCTURE);

    pas_segregated_size_directory* result;

    PAS_UNUSED_PARAM(counts);
    
    if (verbose) {
        pas_log("%p: getting directory with size = %zu, alignment = %zu.\n",
                heap, size, alignment);
    }
    
    result = pas_segregated_heap_size_directory_for_size(
        &heap->segregated_heap, size, config, cached_index);
    if (result && pas_segregated_size_directory_alignment(result) >= alignment)
        return result;

#if PAS_ENABLE_TESTING
    counts->slow_paths++;
#endif /* PAS_ENABLE_TESTING */
    
    return pas_heap_ensure_size_directory_for_size_slow(
        heap, size, alignment, force_size_lookup, config.config_ptr, cached_index);
}

PAS_END_EXTERN_C;

#endif /* PAS_HEAP_INLINES_H */

