/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 15, 2024.
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
#include "Environment.h"
#include "Heap.h"
#include "PerProcess.h"
#include "Sizes.h"
#include <algorithm>
#include <cstdlib>

#if !BUSE(LIBPAS)

namespace bmalloc {

Allocator::Allocator(Heap& heap, Deallocator& deallocator)
    : m_heap(heap)
    , m_deallocator(deallocator)
{
    BASSERT(!Environment::get()->isDebugHeapEnabled());
    for (size_t sizeClass = 0; sizeClass < sizeClassCount; ++sizeClass)
        m_bumpAllocators[sizeClass].init(objectSize(sizeClass));
}

Allocator::~Allocator()
{
    scavenge();
}

void* Allocator::allocateImpl(size_t alignment, size_t size, FailureAction action)
{
    BASSERT(isPowerOfTwo(alignment));

    if (!size)
        size = alignment;

    if (size <= smallMax && alignment <= smallMax)
        return allocateImpl(roundUpToMultipleOf(alignment, size), action);

    return allocateLarge(size, action);
}

void* Allocator::reallocateImpl(void* object, size_t newSize, FailureAction action)
{
    if (!object)
        return allocateImpl(newSize, action);

    size_t oldSize = 0;
    switch (objectType(m_heap, object)) {
    case ObjectType::Small: {
        size_t sizeClass = Object(object).page()->sizeClass();
        oldSize = objectSize(sizeClass);
        break;
    }
    case ObjectType::Large: {
        UniqueLockHolder lock(Heap::mutex());
        oldSize = m_heap.largeSize(lock, object);

        if (newSize < oldSize && newSize > smallMax) {
            m_heap.shrinkLarge(lock, Range(object, oldSize), newSize);
            return object;
        }
        break;
    }
    }

    void* result = nullptr;
    result = allocateImpl(newSize, action);
    if (!result) {
        BASSERT(action == FailureAction::ReturnNull);
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

BNO_INLINE void Allocator::refillAllocatorSlowCase(BumpAllocator& allocator, size_t sizeClass, FailureAction action)
{
    BumpRangeCache& bumpRangeCache = m_bumpRangeCaches[sizeClass];

    UniqueLockHolder lock(Heap::mutex());
    m_deallocator.processObjectLog(lock);
    m_heap.allocateSmallBumpRanges(lock, sizeClass, allocator, bumpRangeCache, m_deallocator.lineCache(lock), action);
}

BINLINE void Allocator::refillAllocator(BumpAllocator& allocator, size_t sizeClass, FailureAction action)
{
    BumpRangeCache& bumpRangeCache = m_bumpRangeCaches[sizeClass];
    if (!bumpRangeCache.size())
        return refillAllocatorSlowCase(allocator, sizeClass, action);
    return allocator.refill(bumpRangeCache.pop());
}

BNO_INLINE void* Allocator::allocateLarge(size_t size, FailureAction action)
{
    UniqueLockHolder lock(Heap::mutex());
    return m_heap.allocateLarge(lock, alignment, size, action);
}

BNO_INLINE void* Allocator::allocateLogSizeClass(size_t size, FailureAction action)
{
    size_t sizeClass = bmalloc::sizeClass(size);
    BumpAllocator& allocator = m_bumpAllocators[sizeClass];
    if (!allocator.canAllocate())
        refillAllocator(allocator, sizeClass, action);
    if (action == FailureAction::ReturnNull && !allocator.canAllocate())
        return nullptr;
    return allocator.allocate();
}

void* Allocator::allocateSlowCase(size_t size, FailureAction action)
{
    if (size <= maskSizeClassMax) {
        size_t sizeClass = bmalloc::maskSizeClass(size);
        BumpAllocator& allocator = m_bumpAllocators[sizeClass];
        refillAllocator(allocator, sizeClass, action);
        if (action == FailureAction::ReturnNull && !allocator.canAllocate())
            return nullptr;
        return allocator.allocate();
    }

    if (size <= smallMax)
        return allocateLogSizeClass(size, action);

    return allocateLarge(size, action);
}

} // namespace bmalloc

#endif
