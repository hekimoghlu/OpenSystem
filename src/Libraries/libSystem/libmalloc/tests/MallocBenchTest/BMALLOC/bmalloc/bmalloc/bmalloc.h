/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 3, 2022.
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
#pragma once

#include "AvailableMemory.h"
#include "Cache.h"
#include "Gigacage.h"
#include "Heap.h"
#include "IsoTLS.h"
#include "Mutex.h"
#include "PerHeapKind.h"
#include "Scavenger.h"

namespace bmalloc {
namespace api {

// Returns null on failure.
inline void* tryMalloc(size_t size, HeapKind kind = HeapKind::Primary)
{
    return Cache::tryAllocate(kind, size);
}

// Crashes on failure.
inline void* malloc(size_t size, HeapKind kind = HeapKind::Primary)
{
    return Cache::allocate(kind, size);
}

BEXPORT void* mallocOutOfLine(size_t size, HeapKind kind = HeapKind::Primary);

// Returns null on failure.
inline void* tryMemalign(size_t alignment, size_t size, HeapKind kind = HeapKind::Primary)
{
    return Cache::tryAllocate(kind, alignment, size);
}

// Crashes on failure.
inline void* memalign(size_t alignment, size_t size, HeapKind kind = HeapKind::Primary)
{
    return Cache::allocate(kind, alignment, size);
}

// Returns null on failure.
inline void* tryRealloc(void* object, size_t newSize, HeapKind kind = HeapKind::Primary)
{
    return Cache::tryReallocate(kind, object, newSize);
}

// Crashes on failure.
inline void* realloc(void* object, size_t newSize, HeapKind kind = HeapKind::Primary)
{
    return Cache::reallocate(kind, object, newSize);
}

// Returns null on failure.
// This API will give you zeroed pages that are ready to be used. These pages
// will page fault on first access. It returns to you memory that initially only
// uses up virtual address space, not `size` bytes of physical memory.
BEXPORT void* tryLargeZeroedMemalignVirtual(size_t alignment, size_t size, HeapKind kind = HeapKind::Primary);

inline void free(void* object, HeapKind kind = HeapKind::Primary)
{
    Cache::deallocate(kind, object);
}

BEXPORT void freeOutOfLine(void* object, HeapKind kind = HeapKind::Primary);

BEXPORT void freeLargeVirtual(void* object, size_t, HeapKind kind = HeapKind::Primary);

inline void scavengeThisThread()
{
    for (unsigned i = numHeaps; i--;)
        Cache::scavenge(static_cast<HeapKind>(i));
    IsoTLS::scavenge();
}

BEXPORT void scavenge();

BEXPORT bool isEnabled(HeapKind kind = HeapKind::Primary);

// ptr must be aligned to vmPageSizePhysical and size must be divisible 
// by vmPageSizePhysical.
BEXPORT void decommitAlignedPhysical(void* object, size_t, HeapKind = HeapKind::Primary);
BEXPORT void commitAlignedPhysical(void* object, size_t, HeapKind = HeapKind::Primary);
    
inline size_t availableMemory()
{
    return bmalloc::availableMemory();
}
    
#if BPLATFORM(IOS_FAMILY)
inline size_t memoryFootprint()
{
    return bmalloc::memoryFootprint();
}

inline double percentAvailableMemoryInUse()
{
    return bmalloc::percentAvailableMemoryInUse();
}
#endif

#if BOS(DARWIN)
BEXPORT void setScavengerThreadQOSClass(qos_class_t overrideClass);
#endif

BEXPORT void enableMiniMode();

} // namespace api
} // namespace bmalloc
