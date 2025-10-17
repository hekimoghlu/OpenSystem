/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 8, 2021.
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
#include "BPlatform.h"
#include "IsoHeap.h"

#if !BUSE(TZONE)
#include "AllocationCounts.h"

#if BUSE(LIBPAS)

#include "bmalloc_heap_inlines.h"
#include "bmalloc_heap_internal.h"
#include "pas_allocation_mode.h"

#include "IsoMallocFallback.h"

namespace bmalloc { namespace api {

void* isoAllocate(pas_heap_ref& heapRef)
{
    // FIXME: libpas should know how to do the fallback thing.
    // https://bugs.webkit.org/show_bug.cgi?id=227177
    
    auto typeSize = pas_simple_type_size(reinterpret_cast<pas_simple_type>(heapRef.type));
    if (IsoMallocFallback::shouldTryToFallBack()) {
        IsoMallocFallback::MallocResult result = IsoMallocFallback::tryMalloc(typeSize, CompactAllocationMode::NonCompact);
        if (result.didFallBack) {
            RELEASE_BASSERT(result.ptr);
            BPROFILE_ALLOCATION(NON_JS_CELL, result.ptr, typeSize);
            return result.ptr;
        }
    }

    void* result = bmalloc_iso_allocate_inline(&heapRef, pas_non_compact_allocation_mode);
    BPROFILE_ALLOCATION(NON_JS_CELL, result, typeSize);
    return result;
}

void* isoTryAllocate(pas_heap_ref& heapRef)
{
    auto typeSize = pas_simple_type_size(reinterpret_cast<pas_simple_type>(heapRef.type));
    if (IsoMallocFallback::shouldTryToFallBack()) {
        IsoMallocFallback::MallocResult result = IsoMallocFallback::tryMalloc(typeSize, CompactAllocationMode::NonCompact);
        if (result.didFallBack) {
            BPROFILE_TRY_ALLOCATION(NON_JS_CELL, result.ptr, typeSize);
            return result.ptr;
        }
    }

    void* result = bmalloc_try_iso_allocate_inline(&heapRef, pas_non_compact_allocation_mode);
    BPROFILE_TRY_ALLOCATION(NON_JS_CELL, result, typeSize);
    return result;
}

void* isoAllocateCompact(pas_heap_ref& heapRef)
{
    // FIXME: libpas should know how to do the fallback thing.
    // https://bugs.webkit.org/show_bug.cgi?id=227177

    auto typeSize = pas_simple_type_size(reinterpret_cast<pas_simple_type>(heapRef.type));
    if (IsoMallocFallback::shouldTryToFallBack()) {
        IsoMallocFallback::MallocResult result = IsoMallocFallback::tryMalloc(typeSize, CompactAllocationMode::Compact);
        if (result.didFallBack) {
            RELEASE_BASSERT(result.ptr);
            BPROFILE_ALLOCATION(COMPACTIBLE, result.ptr, typeSize);
            return result.ptr;
        }
    }

    void* result = bmalloc_iso_allocate_inline(&heapRef, pas_maybe_compact_allocation_mode);
    BPROFILE_ALLOCATION(COMPACTIBLE, result, typeSize);
    return result;
}

void* isoTryAllocateCompact(pas_heap_ref& heapRef)
{
    auto typeSize = pas_simple_type_size(reinterpret_cast<pas_simple_type>(heapRef.type));
    if (IsoMallocFallback::shouldTryToFallBack()) {
        IsoMallocFallback::MallocResult result = IsoMallocFallback::tryMalloc(typeSize, CompactAllocationMode::Compact);
        if (result.didFallBack) {
            BPROFILE_TRY_ALLOCATION(NON_JS_CELL, result.ptr, typeSize);
            return result.ptr;
        }
    }

    void* result = bmalloc_try_iso_allocate_inline(&heapRef, pas_maybe_compact_allocation_mode);
    BPROFILE_TRY_ALLOCATION(COMPACTIBLE, result, typeSize);
    return result;
}

void isoDeallocate(void* ptr)
{
    if (IsoMallocFallback::shouldTryToFallBack()
        && IsoMallocFallback::tryFree(ptr))
        return;

    bmalloc_deallocate_inline(ptr);
}

} } // namespace bmalloc::api

#endif // BUSE(LIBPAS)
#endif // !BUSE(TZONE)
