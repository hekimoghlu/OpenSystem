/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 18, 2023.
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
#include "TZoneHeap.h"

#if BUSE(TZONE)

#include "IsoHeap.h"
#include "IsoMallocFallback.h"
#include "TZoneHeapManager.h"
#include "bmalloc.h"
#include "bmalloc_heap_internal.h"
#include "bmalloc_heap_ref.h"

#if !BUSE(LIBPAS)
#error TZONE implementation requires LIBPAS
#endif

namespace bmalloc { namespace api {

#define TO_PAS_HEAPREF(heapRef) std::bit_cast<pas_heap_ref*>(heapRef)

// HeapRef is an opaque alias for pas_heap_ref* in the underlying implementation.
static_assert(sizeof(HeapRef) == sizeof(pas_heap_ref*));

void* tzoneAllocateNonCompactSlow(size_t requestedSize, const TZoneSpecification& spec)
{
    HeapRef heapRef = *spec.addressOfHeapRef;
    if (BUNLIKELY(tzoneMallocFallback != TZoneMallocFallback::DoNotFallBack)) {
        if (BUNLIKELY(tzoneMallocFallback == TZoneMallocFallback::Undecided)) {
            TZoneHeapManager::ensureSingleton();
            return tzoneAllocateNonCompactSlow(requestedSize, spec);
        }

        RELEASE_BASSERT(tzoneMallocFallback == TZoneMallocFallback::ForceDebugMalloc);
        return api::malloc(requestedSize, CompactAllocationMode::NonCompact);
    }

    // Handle TZoneMallocFallback::DoNotFallBack.
    if (BUNLIKELY(requestedSize != spec.size))
        heapRef = tzoneHeapManager->heapRefForTZoneTypeDifferentSize(requestedSize, spec);

    if (!heapRef) {
        heapRef = tzoneHeapManager->heapRefForTZoneType(spec);
        *spec.addressOfHeapRef = heapRef;
    }
    return bmalloc_iso_allocate_inline(TO_PAS_HEAPREF(heapRef), pas_non_compact_allocation_mode);
}

void* tzoneAllocateCompactSlow(size_t requestedSize, const TZoneSpecification& spec)
{
    HeapRef heapRef = *spec.addressOfHeapRef;
    if (BUNLIKELY(tzoneMallocFallback != TZoneMallocFallback::DoNotFallBack)) {
        if (BUNLIKELY(tzoneMallocFallback == TZoneMallocFallback::Undecided)) {
            TZoneHeapManager::ensureSingleton();
            return tzoneAllocateCompactSlow(requestedSize, spec);
        }

        RELEASE_BASSERT(tzoneMallocFallback == TZoneMallocFallback::ForceDebugMalloc);
        return api::malloc(requestedSize, CompactAllocationMode::Compact);
    }

    // Handle TZoneMallocFallback::DoNotFallBack.
    if (BUNLIKELY(requestedSize != spec.size))
        heapRef = tzoneHeapManager->heapRefForTZoneTypeDifferentSize(requestedSize, spec);

    if (!heapRef) {
        heapRef = tzoneHeapManager->heapRefForTZoneType(spec);
        *spec.addressOfHeapRef = heapRef;
    }
    return bmalloc_iso_allocate_inline(TO_PAS_HEAPREF(heapRef), pas_maybe_compact_allocation_mode);
}

void* tzoneAllocateCompact(HeapRef heapRef)
{
    return bmalloc_iso_allocate_inline(TO_PAS_HEAPREF(heapRef), pas_maybe_compact_allocation_mode);
}

void* tzoneAllocateNonCompact(HeapRef heapRef)
{
    return bmalloc_iso_allocate_inline(TO_PAS_HEAPREF(heapRef), pas_non_compact_allocation_mode);
}

void tzoneFree(void* p)
{
    bmalloc_deallocate_inline(p);
}

#undef TO_PAS_HEAPREF

} } // namespace bmalloc::api

#endif // BUSE(TZONE)
