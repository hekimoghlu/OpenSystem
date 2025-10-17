/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 10, 2024.
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
#include "Allocator.h"
#include "BAssert.h"
#include "Chunk.h"
#include "Deallocator.h"
#include "DebugHeap.h"
#include "Heap.h"
#include "PerProcess.h"
#include "Sizes.h"
#include <algorithm>
#include <cstdlib>

namespace bmalloc {

Allocator::Allocator(Heap& heap, Deallocator& deallocator)
    : m_heap(heap)
    , m_debugHeap(heap.debugHeap())
    , m_deallocator(deallocator)
{
    for (size_t sizeClass = 0; sizeClass < sizeClassCount; ++sizeClass)
        m_bumpAllocators[sizeClass].init(objectSize(sizeClass));
}

Allocator::~Allocator()
{
    scavenge();
}

void* Allocator::tryAllocate(size_t size)
{
    if (m_debugHeap)
        return m_debugHeap->malloc(size);

    if (size <= smallMax)
        return allocate(size);

    std::unique_lock<Mutex> lock(Heap::mutex());
    return m_heap.tryAllocateLarge(lock, alignment, size);
}

void* Allocator::allocate(size_t alignment, size_t size)
{
    bool crashOnFailure = true;
    return allocateImpl(alignment, size, crashOnFailure);
}

void* Allocator::tryAllocate(size_t alignment, size_t size)
{
    bool crashOnFailure = false;
    return allocateImpl(alignment, size, crashOnFailure);
}

void* Allocator::allocateImpl(size_t alignment, size_t size, bool crashOnFailure)
{
    BASSERT(isPowerOfTwo(alignment));

    if (m_debugHeap)
        return m_debugHeap->memalign(alignment, size, crashOnFailure);

    if (!size)
        size = alignment;

    if (size <= smallMax && alignment <= smallMax)
        return allocate(roundUpToMultipleOf(alignment, size));

    std::unique_lock<Mutex> lock(Heap::mutex());
    if (crashOnFailure)
        return m_heap.allocateLarge(lock, alignment, size);
    return m_heap.tryAllocateLarge(lock, alignment, size);
}

void* Allocator::reallocate(void* object, size_t newSize)
{
    bool crashOnFailure = true;
    return reallocateImpl(object, newSize, crashOnFailure);
}

void* Allocator::tryReallocate(void* object, size_t newSize)
{
    bool crashOnFailure = false;
    return reallocateImpl(object, newSize, crashOnFailure);
}

void* Allocator::reallocateImpl(void* object, size_t newSize, bool crashOnFailure)
{
    if (m_debugHeap)
        return m_debugHeap->realloc(object, newSize, crashOnFailure);

    size_t oldSize = 0;
    switch (objectType(m_heap.kind(), object)) {
    case ObjectType::Small: {
        BASSERT(objectType(m_heap.kind(), nullptr) == ObjectType::Small);
        if (!object)
            break;

        size_t sizeClass = Object(object).page()->sizeClass();
        oldSize = objectSize(sizeClass);
        break;
    }
    case ObjectType::Large: {
        std::unique_lock<Mutex> lock(Heap::mutex());
        oldSize = m_heap.largeSize(lock, object);

        if (newSize < oldSize && newSize > smallMax) {
            m_heap.shrinkLarge(lock, Range(object, oldSize), newSize);
            return object;
        }
        break;
    }
    }

    void* result = nullptr;
    if (crashOnFailure)
        result = allocate(newSize);
    else {
        result = tryAllocate(newSize);
        if (!result)
            return nullptr;
    }
    size_t copySize = std::min(oldSize, newSize);
    memcpy(result, object, copySize);
    m_deallocator.deallocate(object);
    return result;
}

void Allocator::scavenge()
{
    for (size_t sizeClass = 0; sizeClass < sizeClassCount; ++sizeClass) {
        BumpAllocator& allocator = m_bumpAllocators[sizeClass];
        BumpRangeCache& bumpRangeCache = m_bumpRangeCaches[sizeClass];

        while (allocator.canAllocate())
            m_deallocator.deallocate(allocator.allocate());

        while (bumpRangeCache.size()) {
            allocator.refill(bumpRangeCache.pop());
            while (allocator.canAllocate())
                m_deallocator.deallocate(allocator.allocate());
        }

        allocator.clear();
    }
}

BNO_INLINE void Allocator::refillAllocatorSlowCase(BumpAllocator& allocator, size_t sizeClass)
{
    BumpRangeCache& bumpRangeCache = m_bumpRangeCaches[sizeClass];

    std::unique_lock<Mutex> lock(Heap::mutex());
    m_deallocator.processObjectLog(lock);
    m_heap.allocateSmallBumpRanges(lock, sizeClass, allocator, bumpRangeCache, m_deallocator.lineCache(lock));
}

BINLINE void Allocator::refillAllocator(BumpAllocator& allocator, size_t sizeClass)
{
    BumpRangeCache& bumpRangeCache = m_bumpRangeCaches[sizeClass];
    if (!bumpRangeCache.size())
        return refillAllocatorSlowCase(allocator, sizeClass);
    return allocator.refill(bumpRangeCache.pop());
}

BNO_INLINE void* Allocator::allocateLarge(size_t size)
{
    std::unique_lock<Mutex> lock(Heap::mutex());
    return m_heap.allocateLarge(lock, alignment, size);
}

BNO_INLINE void* Allocator::allocateLogSizeClass(size_t size)
{
    size_t sizeClass = bmalloc::sizeClass(size);
    BumpAllocator& allocator = m_bumpAllocators[sizeClass];
    if (!allocator.canAllocate())
        refillAllocator(allocator, sizeClass);
    return allocator.allocate();
}

void* Allocator::allocateSlowCase(size_t size)
{
    if (m_debugHeap)
        return m_debugHeap->malloc(size);

    if (size <= maskSizeClassMax) {
        size_t sizeClass = bmalloc::maskSizeClass(size);
        BumpAllocator& allocator = m_bumpAllocators[sizeClass];
        refillAllocator(allocator, sizeClass);
        return allocator.allocate();
    }

    if (size <= smallMax)
        return allocateLogSizeClass(size);

    return allocateLarge(size);
}

} // namespace bmalloc
