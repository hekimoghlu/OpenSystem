/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 9, 2022.
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
#include "bmalloc_heap.h"

#if PAS_ENABLE_BMALLOC

#include "pas_scavenger.h"

#if PAS_OS(DARWIN)
#include <malloc/malloc.h>
#endif

void* mbmalloc(size_t size)
{
    return bmalloc_try_allocate(size, pas_non_compact_allocation_mode);
}

void* mbmemalign(size_t alignment, size_t size)
{
    return bmalloc_try_allocate_with_alignment(size, alignment, pas_non_compact_allocation_mode);
}

void* mbrealloc(void* p, size_t ignored_old_size, size_t new_size)
{
    return bmalloc_try_reallocate(p, new_size, pas_non_compact_allocation_mode, pas_reallocate_free_if_successful);
}

void mbfree(void* p, size_t ignored_size)
{
    bmalloc_deallocate(p);
}

void mbscavenge(void)
{
    pas_scavenger_run_synchronously_now();
#if PAS_OS(DARWIN)
    malloc_zone_pressure_relief(NULL, 0);
#endif
}

#endif /* PAS_ENABLE_BMALLOC */
