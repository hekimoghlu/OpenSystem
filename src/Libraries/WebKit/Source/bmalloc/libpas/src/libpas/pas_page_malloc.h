/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 23, 2025.
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
#ifndef PAS_PAGE_MALLOC_H
#define PAS_PAGE_MALLOC_H

#include "pas_aligned_allocation_result.h"
#include "pas_alignment.h"
#include "pas_mmap_capability.h"
#include "pas_utils.h"

PAS_BEGIN_EXTERN_C;

PAS_API extern size_t pas_page_malloc_num_allocated_bytes;
PAS_API extern size_t pas_page_malloc_cached_alignment;
PAS_API extern size_t pas_page_malloc_cached_alignment_shift;
#if PAS_OS(DARWIN)
PAS_API extern bool pas_page_malloc_decommit_zero_fill;
#endif /* PAS_OS(DARWIN) */

PAS_API PAS_NEVER_INLINE size_t pas_page_malloc_alignment_slow(void);

static inline size_t pas_page_malloc_alignment(void)
{
    if (!pas_page_malloc_cached_alignment)
        pas_page_malloc_cached_alignment = pas_page_malloc_alignment_slow();
    return pas_page_malloc_cached_alignment;
}

PAS_API PAS_NEVER_INLINE size_t pas_page_malloc_alignment_shift_slow(void);

static inline size_t pas_page_malloc_alignment_shift(void)
{
    if (!pas_page_malloc_cached_alignment_shift)
        pas_page_malloc_cached_alignment_shift = pas_page_malloc_alignment_shift_slow();
    return pas_page_malloc_cached_alignment_shift;
}

PAS_API pas_aligned_allocation_result
pas_page_malloc_try_allocate_without_deallocating_padding(
    size_t size, pas_alignment alignment, bool may_contain_small_or_medium);

PAS_API void pas_page_malloc_deallocate(void* base, size_t size);

PAS_API void pas_page_malloc_zero_fill(void* base, size_t size);

/* This even works if size < pas_page_malloc_alignment so long as the range [base, base+size) is
   entirely within a page according to pas_page_malloc_alignment. */
PAS_API void pas_page_malloc_commit(void* base, size_t size, pas_mmap_capability mmap_capability);
PAS_API void pas_page_malloc_decommit(void* base, size_t size, pas_mmap_capability mmap_capability);

/* In testing mode, we have commit/decommit mprotect the memory as a way of helping us see if we are
   accidentally reading or writing that memory. But sometimes we use commit/decommit in a way that prevents
   us from making such assertions, like if we allow some reads and writes to decommitted memory in rare
   cases. */
PAS_API void pas_page_malloc_commit_without_mprotect(
    void* base, size_t size, pas_mmap_capability mmap_capability);
PAS_API void pas_page_malloc_decommit_without_mprotect(
    void* base, size_t size, pas_mmap_capability mmap_capability);

PAS_END_EXTERN_C;

#endif /* PAS_PAGE_MALLOC_H */

