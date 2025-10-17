/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 19, 2024.
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

#include "pas_bootstrap_free_heap.h"

#include "pas_config.h"
#include "pas_heap_lock.h"
#include "pas_large_free_heap_config.h"
#include "pas_enumerable_page_malloc.h"
#include "pas_simple_free_heap_helpers.h"

static pas_aligned_allocation_result bootstrap_source_allocate_aligned(size_t size,
                                                                       pas_alignment alignment,
                                                                       void* arg)
{
    PAS_UNUSED_PARAM(arg);
    static const bool verbose = PAS_SHOULD_LOG(PAS_LOG_BOOTSTRAP_HEAPS);

    if (verbose)
        pas_log("bootstrap heap allocating %zu\n", size);

    pas_aligned_allocation_result retval = pas_enumerable_page_malloc_try_allocate_without_deallocating_padding(size, alignment, false);

    if (verbose)
        pas_log("bootstrap heap done allocating, returning %p.\n", retval.result);

    return retval;
}

static void initialize_config(pas_large_free_heap_config* config)
{
    config->type_size = 1;
    config->min_alignment = 1;
    config->aligned_allocator = bootstrap_source_allocate_aligned;
    config->aligned_allocator_arg = NULL;
    config->deallocator = NULL;
    config->deallocator_arg = NULL;
}

#define PAS_SIMPLE_FREE_HEAP_NAME pas_bootstrap_free_heap
#define PAS_SIMPLE_FREE_HEAP_ID(suffix) pas_bootstrap_free_heap ## suffix
#include "pas_simple_free_heap_definitions.def"
#undef PAS_SIMPLE_FREE_HEAP_NAME
#undef PAS_SIMPLE_FREE_HEAP_ID

#endif /* LIBPAS_ENABLED */
