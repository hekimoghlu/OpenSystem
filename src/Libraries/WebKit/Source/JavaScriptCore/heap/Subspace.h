/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 3, 2023.
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
#include "AllocatorForMode.h"
#include "Allocator.h"
#include "MarkedBlock.h"
#include "MarkedSpace.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/text/CString.h>

namespace JSC {

class AlignedMemoryAllocator;
class HeapCellType;

// The idea of subspaces is that you can provide some custom behavior for your objects if you
// allocate them from a custom Subspace in which you override some of the virtual methods. This
// class is the baseclass of all subspaces e.g. CompleteSubspace, IsoSubspace.
class Subspace {
    WTF_MAKE_NONCOPYABLE(Subspace);
    WTF_MAKE_TZONE_ALLOCATED(Subspace);
public:
    JS_EXPORT_PRIVATE virtual ~Subspace();

    const char* name() const { return m_name.data(); }
    unsigned nameHash() const { return m_name.hash(); } // FIXME: rdar://139998916
    MarkedSpace& space() const { return m_space; }

    CellAttributes attributes() const;
    const HeapCellType* heapCellType() const { return m_heapCellType; }
    AlignedMemoryAllocator* alignedMemoryAllocator() const { return m_alignedMemoryAllocator; }
    
    void finishSweep(MarkedBlock::Handle&, FreeList*);
    void destroy(VM&, JSCell*);

    void prepareForAllocation();
    
    void didCreateFirstDirectory(BlockDirectory* directory) { m_directoryForEmptyAllocation = directory; }
    
    // Finds an empty block from any Subspace that agrees to trade blocks with us.
    MarkedBlock::Handle* findEmptyBlockToSteal();
    
    template<typename Func>
    void forEachDirectory(const Func&);
    
    Ref<SharedTask<BlockDirectory*()>> parallelDirectorySource();
    
    template<typename Func>
    void forEachMarkedBlock(const Func&);
    
    template<typename Func>
    void forEachNotEmptyMarkedBlock(const Func&);
    
    JS_EXPORT_PRIVATE Ref<SharedTask<MarkedBlock::Handle*()>> parallelNotEmptyMarkedBlockSource();
    
    template<typename Func>
    void forEachPreciseAllocation(const Func&);
    
    template<typename Func>
    void forEachMarkedCell(const Func&);
    
    template<typename Visitor, typename Func>
    Ref<SharedTask<void(Visitor&)>> forEachMarkedCellInParallel(const Func&);

    template<typename Func>
    void forEachLiveCell(const Func&);
    
    void sweepBlocks();
    
    Subspace* nextSubspaceInAlignedMemoryAllocator() const { return m_nextSubspaceInAlignedMemoryAllocator; }
    void setNextSubspaceInAlignedMemoryAllocator(Subspace* subspace) { m_nextSubspaceInAlignedMemoryAllocator = subspace; }
    
    virtual void didResizeBits(unsigned newSize);
    virtual void didRemoveBlock(unsigned blockIndex);
    virtual void didBeginSweepingToFreeList(MarkedBlock::Handle*);

    bool isIsoSubspace() const { return m_isIsoSubspace; }
    bool isPreciseOnly() const { return m_isPreciseOnly; }

protected:
    Subspace(CString name, Heap&);

    void initialize(const HeapCellType&, AlignedMemoryAllocator*);
    
    MarkedSpace& m_space;
    
    const HeapCellType* m_heapCellType { nullptr };
    AlignedMemoryAllocator* m_alignedMemoryAllocator { nullptr };
    
    BlockDirectory* m_firstDirectory { nullptr };
    BlockDirectory* m_directoryForEmptyAllocation { nullptr }; // Uses the MarkedSpace linked list of blocks.
    SentinelLinkedList<PreciseAllocation, BasicRawSentinelNode<PreciseAllocation>> m_preciseAllocations;

    bool m_isIsoSubspace { false };
    bool m_isPreciseOnly { false };
    uint8_t m_remainingLowerTierPreciseCount { 0 }; // Lower tier is a precise allocation but we use the term lower to avoid confusion with precise-only.

    Subspace* m_nextSubspaceInAlignedMemoryAllocator { nullptr };

    CString m_name;
};

} // namespace JSC

