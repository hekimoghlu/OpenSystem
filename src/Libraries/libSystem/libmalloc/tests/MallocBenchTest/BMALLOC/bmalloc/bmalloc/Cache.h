/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 31, 2023.
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
#ifndef Cache_h
#define Cache_h

#include "Allocator.h"
#include "BExport.h"
#include "Deallocator.h"
#include "HeapKind.h"
#include "PerThread.h"

namespace bmalloc {

// Per-thread allocation / deallocation cache, backed by a per-process Heap.

class Cache {
public:
    static void* tryAllocate(HeapKind, size_t);
    static void* allocate(HeapKind, size_t);
    static void* tryAllocate(HeapKind, size_t alignment, size_t);
    static void* allocate(HeapKind, size_t alignment, size_t);
    static void deallocate(HeapKind, void*);
    static void* tryReallocate(HeapKind, void*, size_t);
    static void* reallocate(HeapKind, void*, size_t);

    static void scavenge(HeapKind);

    Cache(HeapKind);

    Allocator& allocator() { return m_allocator; }
    Deallocator& deallocator() { return m_deallocator; }

private:
    BEXPORT static void* tryAllocateSlowCaseNullCache(HeapKind, size_t);
    BEXPORT static void* allocateSlowCaseNullCache(HeapKind, size_t);
    BEXPORT static void* allocateSlowCaseNullCache(HeapKind, size_t alignment, size_t);
    BEXPORT static void deallocateSlowCaseNullCache(HeapKind, void*);
    BEXPORT static void* reallocateSlowCaseNullCache(HeapKind, void*, size_t);

    Deallocator m_deallocator;
    Allocator m_allocator;
};

inline void* Cache::tryAllocate(HeapKind heapKind, size_t size)
{
    PerHeapKind<Cache>* caches = PerThread<PerHeapKind<Cache>>::getFastCase();
    if (!caches)
        return tryAllocateSlowCaseNullCache(heapKind, size);
    return caches->at(mapToActiveHeapKindAfterEnsuringGigacage(heapKind)).allocator().tryAllocate(size);
}

inline void* Cache::allocate(HeapKind heapKind, size_t size)
{
    PerHeapKind<Cache>* caches = PerThread<PerHeapKind<Cache>>::getFastCase();
    if (!caches)
        return allocateSlowCaseNullCache(heapKind, size);
    return caches->at(mapToActiveHeapKindAfterEnsuringGigacage(heapKind)).allocator().allocate(size);
}

inline void* Cache::tryAllocate(HeapKind heapKind, size_t alignment, size_t size)
{
    PerHeapKind<Cache>* caches = PerThread<PerHeapKind<Cache>>::getFastCase();
    if (!caches)
        return allocateSlowCaseNullCache(heapKind, alignment, size);
    return caches->at(mapToActiveHeapKindAfterEnsuringGigacage(heapKind)).allocator().tryAllocate(alignment, size);
}

inline void* Cache::allocate(HeapKind heapKind, size_t alignment, size_t size)
{
    PerHeapKind<Cache>* caches = PerThread<PerHeapKind<Cache>>::getFastCase();
    if (!caches)
        return allocateSlowCaseNullCache(heapKind, alignment, size);
    return caches->at(mapToActiveHeapKindAfterEnsuringGigacage(heapKind)).allocator().allocate(alignment, size);
}

inline void Cache::deallocate(HeapKind heapKind, void* object)
{
    PerHeapKind<Cache>* caches = PerThread<PerHeapKind<Cache>>::getFastCase();
    if (!caches)
        return deallocateSlowCaseNullCache(heapKind, object);
    return caches->at(mapToActiveHeapKindAfterEnsuringGigacage(heapKind)).deallocator().deallocate(object);
}

inline void* Cache::tryReallocate(HeapKind heapKind, void* object, size_t newSize)
{
    PerHeapKind<Cache>* caches = PerThread<PerHeapKind<Cache>>::getFastCase();
    if (!caches)
        return reallocateSlowCaseNullCache(heapKind, object, newSize);
    return caches->at(mapToActiveHeapKindAfterEnsuringGigacage(heapKind)).allocator().tryReallocate(object, newSize);
}

inline void* Cache::reallocate(HeapKind heapKind, void* object, size_t newSize)
{
    PerHeapKind<Cache>* caches = PerThread<PerHeapKind<Cache>>::getFastCase();
    if (!caches)
        return reallocateSlowCaseNullCache(heapKind, object, newSize);
    return caches->at(mapToActiveHeapKindAfterEnsuringGigacage(heapKind)).allocator().reallocate(object, newSize);
}

} // namespace bmalloc

#endif // Cache_h
