/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 25, 2023.
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
#ifndef Allocator_h
#define Allocator_h

#include "BExport.h"
#include "BumpAllocator.h"
#include <array>

namespace bmalloc {

class Deallocator;
class DebugHeap;
class Heap;

// Per-cache object allocator.

class Allocator {
public:
    Allocator(Heap&, Deallocator&);
    ~Allocator();

    BEXPORT void* tryAllocate(size_t);
    void* allocate(size_t);
    void* tryAllocate(size_t alignment, size_t);
    void* allocate(size_t alignment, size_t);
    void* tryReallocate(void*, size_t);
    void* reallocate(void*, size_t);

    void scavenge();

private:
    void* allocateImpl(size_t alignment, size_t, bool crashOnFailure);
    void* reallocateImpl(void*, size_t, bool crashOnFailure);

    bool allocateFastCase(size_t, void*&);
    BEXPORT void* allocateSlowCase(size_t);
    
    void* allocateLogSizeClass(size_t);
    void* allocateLarge(size_t);
    
    void refillAllocator(BumpAllocator&, size_t sizeClass);
    void refillAllocatorSlowCase(BumpAllocator&, size_t sizeClass);
    
    std::array<BumpAllocator, sizeClassCount> m_bumpAllocators;
    std::array<BumpRangeCache, sizeClassCount> m_bumpRangeCaches;

    Heap& m_heap;
    DebugHeap* m_debugHeap;
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

inline void* Allocator::allocate(size_t size)
{
    void* object;
    if (!allocateFastCase(size, object))
        return allocateSlowCase(size);
    return object;
}

} // namespace bmalloc

#endif // Allocator_h
