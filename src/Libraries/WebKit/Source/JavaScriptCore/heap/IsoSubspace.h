/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 31, 2025.
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

#include "BlockDirectory.h"
#include "IsoMemoryAllocatorBase.h"
#include "Subspace.h"
#include "SubspaceAccess.h"
#include <wtf/SinglyLinkedListWithTail.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {

class IsoCellSet;

namespace GCClient {
class IsoSubspace;
}

class IsoSubspace : public Subspace {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(IsoSubspace, JS_EXPORT_PRIVATE);
public:
    JS_EXPORT_PRIVATE IsoSubspace(CString name, Heap&, const HeapCellType&, size_t size, bool preciseOnly, uint8_t numberOfLowerTierPreciseCells, std::unique_ptr<IsoMemoryAllocatorBase>&& = nullptr);
    JS_EXPORT_PRIVATE ~IsoSubspace() override;

    size_t cellSize() { return m_directory.cellSize(); }

    void sweepLowerTierPreciseCell(PreciseAllocation*);
    void clearIsoCellSetBit(PreciseAllocation*);

    void* tryAllocatePreciseOrLowerTierPrecise(size_t cellSize);
    void destroyLowerTierPreciseFreeList();

    void sweep();

    template<typename Func> void forEachLowerTierPreciseFreeListedPreciseAllocation(const Func&);

private:
    friend class IsoCellSet;
    friend class GCClient::IsoSubspace;
    
    void didResizeBits(unsigned newSize) override;
    void didRemoveBlock(unsigned blockIndex) override;
    void didBeginSweepingToFreeList(MarkedBlock::Handle*) override;

    BlockDirectory m_directory;
    std::unique_ptr<IsoMemoryAllocatorBase> m_isoAlignedMemoryAllocator;
    SentinelLinkedList<PreciseAllocation, BasicRawSentinelNode<PreciseAllocation>> m_lowerTierPreciseFreeList;
    SentinelLinkedList<IsoCellSet, BasicRawSentinelNode<IsoCellSet>> m_cellSets;
};


namespace GCClient {

class IsoSubspace {
    WTF_MAKE_NONCOPYABLE(IsoSubspace);
    WTF_MAKE_FAST_ALLOCATED;
public:
    JS_EXPORT_PRIVATE IsoSubspace(JSC::IsoSubspace&);
    JS_EXPORT_PRIVATE ~IsoSubspace() = default;

    size_t cellSize() { return m_localAllocator.cellSize(); }

    Allocator allocatorFor(size_t, AllocatorForMode);

    void* allocate(VM&, size_t, GCDeferralContext*, AllocationFailureMode);

private:
    LocalAllocator m_localAllocator;
};

ALWAYS_INLINE Allocator IsoSubspace::allocatorFor(size_t size, AllocatorForMode)
{
    RELEASE_ASSERT(size <= cellSize());
    return Allocator(&m_localAllocator);
}

} // namespace GCClient

#define ISO_SUBSPACE_INIT(heap, heapCellType, type) ("IsoSpace " #type, (heap), (heapCellType), sizeof(type), type::usePreciseAllocationsOnly, type::numberOfLowerTierPreciseCells)

} // namespace JSC

