/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 21, 2025.
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
#ifndef Heap_h
#define Heap_h

#include "BumpRange.h"
#include "Chunk.h"
#include "HeapKind.h"
#include "LargeMap.h"
#include "LineMetadata.h"
#include "List.h"
#include "Map.h"
#include "Mutex.h"
#include "Object.h"
#include "PerHeapKind.h"
#include "PerProcess.h"
#include "PhysicalPageMap.h"
#include "SmallLine.h"
#include "SmallPage.h"
#include "Vector.h"
#include <array>
#include <condition_variable>
#include <mutex>
#include <vector>

namespace bmalloc {

class BeginTag;
class BulkDecommit;
class BumpAllocator;
class DebugHeap;
class EndTag;
class Scavenger;

class Heap {
public:
    Heap(HeapKind, std::lock_guard<Mutex>&);
    
    static Mutex& mutex() { return PerProcess<PerHeapKind<Heap>>::mutex(); }
    
    HeapKind kind() const { return m_kind; }
    
    DebugHeap* debugHeap() { return m_debugHeap; }

    void allocateSmallBumpRanges(std::unique_lock<Mutex>&, size_t sizeClass,
        BumpAllocator&, BumpRangeCache&, LineCache&);
    void derefSmallLine(std::unique_lock<Mutex>&, Object, LineCache&);
    void deallocateLineCache(std::unique_lock<Mutex>&, LineCache&);

    void* allocateLarge(std::unique_lock<Mutex>&, size_t alignment, size_t);
    void* tryAllocateLarge(std::unique_lock<Mutex>&, size_t alignment, size_t);
    void deallocateLarge(std::unique_lock<Mutex>&, void*);

    bool isLarge(std::unique_lock<Mutex>&, void*);
    size_t largeSize(std::unique_lock<Mutex>&, void*);
    void shrinkLarge(std::unique_lock<Mutex>&, const Range&, size_t);

    void scavenge(std::lock_guard<Mutex>&, BulkDecommit&);
    void scavenge(std::lock_guard<Mutex>&, BulkDecommit&, size_t& freed, size_t goal);
    void scavengeToHighWatermark(std::lock_guard<Mutex>&, BulkDecommit&);

    size_t freeableMemory(std::lock_guard<Mutex>&);
    size_t footprint();

    void externalDecommit(void* ptr, size_t);
    void externalDecommit(std::unique_lock<Mutex>&, void* ptr, size_t);
    void externalCommit(void* ptr, size_t);
    void externalCommit(std::unique_lock<Mutex>&, void* ptr, size_t);

    void markAllLargeAsEligibile(std::lock_guard<Mutex>&);

private:
    void decommitLargeRange(std::lock_guard<Mutex>&, LargeRange&, BulkDecommit&);

    struct LargeObjectHash {
        static unsigned hash(void* key)
        {
            return static_cast<unsigned>(
                reinterpret_cast<uintptr_t>(key) / smallMax);
        }
    };

    ~Heap() = delete;
    
    bool usingGigacage();
    void* gigacageBasePtr(); // May crash if !usingGigacage().
    size_t gigacageSize();
    
    void initializeLineMetadata();
    void initializePageMetadata();

    void allocateSmallBumpRangesByMetadata(std::unique_lock<Mutex>&,
        size_t sizeClass, BumpAllocator&, BumpRangeCache&, LineCache&);
    void allocateSmallBumpRangesByObject(std::unique_lock<Mutex>&,
        size_t sizeClass, BumpAllocator&, BumpRangeCache&, LineCache&);

    SmallPage* allocateSmallPage(std::unique_lock<Mutex>&, size_t sizeClass, LineCache&);
    void deallocateSmallLine(std::unique_lock<Mutex>&, Object, LineCache&);

    void allocateSmallChunk(std::unique_lock<Mutex>&, size_t pageClass);
    void deallocateSmallChunk(Chunk*, size_t pageClass);

    void mergeLarge(BeginTag*&, EndTag*&, Range&);
    void mergeLargeLeft(EndTag*&, BeginTag*&, Range&, bool& inVMHeap);
    void mergeLargeRight(EndTag*&, BeginTag*&, Range&, bool& inVMHeap);

    LargeRange splitAndAllocate(std::unique_lock<Mutex>&, LargeRange&, size_t alignment, size_t);

    HeapKind m_kind;
    
    size_t m_vmPageSizePhysical;
    Vector<LineMetadata> m_smallLineMetadata;
    std::array<size_t, sizeClassCount> m_pageClasses;

    LineCache m_lineCache;
    std::array<List<Chunk>, pageClassCount> m_freePages;
    std::array<List<Chunk>, pageClassCount> m_chunkCache;

    Map<void*, size_t, LargeObjectHash> m_largeAllocated;
    LargeMap m_largeFree;

    Map<Chunk*, ObjectType, ChunkHash> m_objectTypes;

    Scavenger* m_scavenger { nullptr };
    DebugHeap* m_debugHeap { nullptr };

    size_t m_footprint { 0 };
    size_t m_freeableMemory { 0 };

    bool m_hasPendingDecommits { false };
    std::condition_variable_any m_condition;

#if ENABLE_PHYSICAL_PAGE_MAP 
    PhysicalPageMap m_physicalPageMap;
#endif

    void* m_highWatermark { nullptr };
};

inline void Heap::allocateSmallBumpRanges(
    std::unique_lock<Mutex>& lock, size_t sizeClass,
    BumpAllocator& allocator, BumpRangeCache& rangeCache,
    LineCache& lineCache)
{
    if (sizeClass < bmalloc::sizeClass(smallLineSize))
        return allocateSmallBumpRangesByMetadata(lock, sizeClass, allocator, rangeCache, lineCache);
    return allocateSmallBumpRangesByObject(lock, sizeClass, allocator, rangeCache, lineCache);
}

inline void Heap::derefSmallLine(std::unique_lock<Mutex>& lock, Object object, LineCache& lineCache)
{
    if (!object.line()->deref(lock))
        return;
    deallocateSmallLine(lock, object, lineCache);
}

} // namespace bmalloc

#endif // Heap_h
