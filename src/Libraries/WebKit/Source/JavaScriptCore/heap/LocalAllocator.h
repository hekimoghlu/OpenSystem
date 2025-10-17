/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 19, 2022.
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

#include "AllocationFailureMode.h"
#include "FreeList.h"
#include "MarkedBlock.h"
#include <wtf/Noncopyable.h>

namespace JSC {

class BlockDirectory;
class GCDeferralContext;
class Heap;

class LocalAllocator : public BasicRawSentinelNode<LocalAllocator> {
    WTF_MAKE_NONCOPYABLE(LocalAllocator);
    
public:
    LocalAllocator(BlockDirectory*);
    JS_EXPORT_PRIVATE ~LocalAllocator();
    
    void* allocate(Heap&, size_t cellSize, GCDeferralContext*, AllocationFailureMode);
    
    unsigned cellSize() const { return m_freeList.cellSize(); }

    void stopAllocating();
    void prepareForAllocation();
    void resumeAllocating();
    void stopAllocatingForGood();
    
    static constexpr ptrdiff_t offsetOfFreeList();
    static constexpr ptrdiff_t offsetOfCellSize();

    BlockDirectory& directory() const { return *m_directory; }

private:
    friend class BlockDirectory;
    
    void reset();
    JS_EXPORT_PRIVATE void* allocateSlowCase(Heap&, size_t, GCDeferralContext*, AllocationFailureMode);
    void didConsumeFreeList();
    void* tryAllocateWithoutCollecting(size_t);
    void* tryAllocateIn(MarkedBlock::Handle*, size_t);
    void* allocateIn(MarkedBlock::Handle*, size_t cellSize);
    ALWAYS_INLINE void doTestCollectionsIfNeeded(Heap&, GCDeferralContext*);

    BlockDirectory* m_directory;
    FreeList m_freeList;

    MarkedBlock::Handle* m_currentBlock { nullptr };
    MarkedBlock::Handle* m_lastActiveBlock { nullptr };
    
    // After you do something to a block based on one of these cursors, you clear the bit in the
    // corresponding bitvector and leave the cursor where it was.
    unsigned m_allocationCursor { 0 }; // Points to the next block that is a candidate for allocation.
};

inline constexpr ptrdiff_t LocalAllocator::offsetOfFreeList()
{
    return OBJECT_OFFSETOF(LocalAllocator, m_freeList);
}

inline constexpr ptrdiff_t LocalAllocator::offsetOfCellSize()
{
    return OBJECT_OFFSETOF(LocalAllocator, m_freeList) + FreeList::offsetOfCellSize();
}

} // namespace JSC

