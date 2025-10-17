/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 9, 2024.
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
#include "BInline.h"
#include "Cache.h"
#include "DebugHeap.h"
#include "Environment.h"
#include "Heap.h"
#include "PerProcess.h"

#if !BUSE(LIBPAS)

namespace bmalloc {

void Cache::scavenge(HeapKind heapKind)
{
    PerHeapKind<Cache>* caches = PerThread<PerHeapKind<Cache>>::getFastCase();
    if (!caches)
        return;
    if (!isActiveHeapKind(heapKind))
        return;

    caches->at(heapKind).allocator().scavenge();
    caches->at(heapKind).deallocator().scavenge();
}

Cache::Cache(HeapKind heapKind)
    : m_deallocator(PerProcess<PerHeapKind<Heap>>::get()->at(heapKind))
    , m_allocator(PerProcess<PerHeapKind<Heap>>::get()->at(heapKind), m_deallocator)
{
    BASSERT(!Environment::get()->isDebugHeapEnabled());
}

BNO_INLINE void* Cache::tryAllocateSlowCaseNullCache(HeapKind heapKind, size_t size)
{
    if (auto* debugHeap = DebugHeap::tryGet())
        return debugHeap->malloc(size, FailureAction::ReturnNull);
    return PerThread<PerHeapKind<Cache>>::getSlowCase()->at(mapToActiveHeapKind(heapKind)).allocator().tryAllocate(size);
}

BNO_INLINE void* Cache::allocateSlowCaseNullCache(HeapKind heapKind, size_t size)
{
    if (auto* debugHeap = DebugHeap::tryGet())
        return debugHeap->malloc(size, FailureAction::Crash);
    return PerThread<PerHeapKind<Cache>>::getSlowCase()->at(mapToActiveHeapKind(heapKind)).allocator().allocate(size);
}

BNO_INLINE void* Cache::tryAllocateSlowCaseNullCache(HeapKind heapKind, size_t alignment, size_t size)
{
    if (auto* debugHeap = DebugHeap::tryGet())
        return debugHeap->memalign(alignment, size, FailureAction::ReturnNull);
    return PerThread<PerHeapKind<Cache>>::getSlowCase()->at(mapToActiveHeapKind(heapKind)).allocator().tryAllocate(alignment, size);
}

BNO_INLINE void* Cache::allocateSlowCaseNullCache(HeapKind heapKind, size_t alignment, size_t size)
{
    if (auto* debugHeap = DebugHeap::tryGet())
        return debugHeap->memalign(alignment, size, FailureAction::Crash);
    return PerThread<PerHeapKind<Cache>>::getSlowCase()->at(mapToActiveHeapKind(heapKind)).allocator().allocate(alignment, size);
}

BNO_INLINE void Cache::deallocateSlowCaseNullCache(HeapKind heapKind, void* object)
{
    if (auto* debugHeap = DebugHeap::tryGet()) {
        debugHeap->free(object);
        return;
    }
    PerThread<PerHeapKind<Cache>>::getSlowCase()->at(mapToActiveHeapKind(heapKind)).deallocator().deallocate(object);
}

BNO_INLINE void* Cache::tryReallocateSlowCaseNullCache(HeapKind heapKind, void* object, size_t newSize)
{
    if (auto* debugHeap = DebugHeap::tryGet())
        return debugHeap->realloc(object, newSize, FailureAction::ReturnNull);
    return PerThread<PerHeapKind<Cache>>::getSlowCase()->at(mapToActiveHeapKind(heapKind)).allocator().tryReallocate(object, newSize);
}

BNO_INLINE void* Cache::reallocateSlowCaseNullCache(HeapKind heapKind, void* object, size_t newSize)
{
    if (auto* debugHeap = DebugHeap::tryGet())
        return debugHeap->realloc(object, newSize, FailureAction::Crash);
    return PerThread<PerHeapKind<Cache>>::getSlowCase()->at(mapToActiveHeapKind(heapKind)).allocator().reallocate(object, newSize);
}

} // namespace bmalloc

#endif
