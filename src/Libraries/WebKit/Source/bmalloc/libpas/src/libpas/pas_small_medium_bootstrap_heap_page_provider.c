/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 23, 2025.
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

#include "pas_small_medium_bootstrap_heap_page_provider.h"

#include "pas_small_medium_bootstrap_free_heap.h"

pas_allocation_result pas_small_medium_bootstrap_heap_page_provider(
    size_t size,
    pas_alignment alignment,
    const char* name,
    pas_heap* heap,
    pas_physical_memory_transaction* transaction,
    void *arg)
{
    PAS_UNUSED_PARAM(arg);
    PAS_UNUSED_PARAM(heap);
    PAS_UNUSED_PARAM(transaction);
    static const bool verbose = PAS_SHOULD_LOG(PAS_LOG_BOOTSTRAP_HEAPS);

    if (verbose)
        pas_log("small/medium bootstrap heap page-provider allocating %zu for %s\n", size, name);

    pas_allocation_result retval = pas_small_medium_bootstrap_free_heap_try_allocate_with_alignment(
        size, alignment, name, pas_delegate_allocation);

    if (verbose)
        pas_log("small/medium bootstrap heap page-provider done allocating\n");

    return retval;
}

#endif /* LIBPAS_ENABLED */
