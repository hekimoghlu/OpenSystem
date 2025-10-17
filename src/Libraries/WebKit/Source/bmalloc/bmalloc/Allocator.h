/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 14, 2024.
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

#include "BExport.h"
#include "BumpAllocator.h"
#include "FailureAction.h"
#include <array>

#if !BUSE(LIBPAS)

namespace bmalloc {

class Deallocator;
class Heap;

// Per-cache object allocator.

class Allocator {
public:
    Allocator(Heap&, Deallocator&);
    ~Allocator();

    void* tryAllocate(size_t size) { return allocateImpl(size, FailureAction::ReturnNull); }
    void* allocate(size_t size) { return allocateImpl(size, FailureAction::Crash); }
    void* tryAllocate(size_t alignment, size_t size) { return allocateImpl(alignment, size, FailureAction::ReturnNull); }
    void* allocate(size_t alignment, size_t size) { return allocateImpl(alignment, size, FailureAction::Crash); }
    void* tryReallocate(void* object, size_t newSize) { return reallocateImpl(object, newSize, FailureAction::ReturnNull); }
    void* reallocate(void* object, size_t newSize) { return reallocateImpl(object, newSize, FailureAction::Crash); }

    void scavenge();

private:
    void* allocateImpl(size_t, FailureAction);
    BEXPORT void* allocateImpl(size_t alignment, size_t, FailureAction);
    BEXPORT void* reallocateImpl(void*, size_t, FailureAction);

    bool allocateFastCase(size_t, void*&);
    BEXPORT void* allocateSlowCase(size_t, FailureAction);

    void* allocateLogSizeClass(size_t, FailureAction);
    void* allocateLarge(size_t, FailureAction);
    
    inline void refillAllocator(BumpAllocator&, size_t sizeClass, FailureAction);
    void refillAllocatorSlowCase(BumpAllocator&, size_t sizeClass, FailureAction);
    
    std::array<BumpAllocator, sizeClassCount> m_bumpAllocators;
    std::array<BumpRangeCache, sizeClassCount> m_bumpRangeCaches;

    Heap& m_heap;
    Deallocator& m_deallocator;
};

inline bool Allocator::allocateFastCase(size_t size, void*& object)
{
    if (size > maskSizeClassMax)
        return false;

    BumpAllocator& allocator = m_bumpAllocators[maskSizeClass(size)];
    if (!allocator.canAllocate())
        return false;

    object = allocator.allocate();
    return true;
}

inline void* Allocator::allocateImpl(size_t size, FailureAction action)
{
    void* object;
    if (!allocateFastCase(size, object))
        return allocateSlowCase(size, action);
    return object;
}

} // namespace bmalloc

#endif
